import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, AutoImageProcessor
from torch.optim import AdamW
from torchvision import datasets, transforms
import argparse
import math
from fla.models import KDAModel, KDAConfig


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


@dataclass
class ModelConfig:
    patch_size: int = 2
    latent_size: int = 128
    latent_channels: int = 32
    num_layers: int = 4
    attention_head_dim: int = 32
    num_attention_heads: int = 16
    hidden_size: int = 128  # latent_channels * patch_size**2


@dataclass
class TrainingConfig:
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    vae_model: str = "black-forest-labs/FLUX.1-dev"
    image_size: int = 1024
    num_train_timesteps: int = 1000
    dataset_root: str = "/path/to/imagenet"
    output_dir: str = "./output"


class TransformerDenoiser(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.kda_config = KDAConfig(
            hidden_size=config.hidden_size,
            use_short_conv=True,
            head_dim=config.attention_head_dim,
            num_heads=config.num_attention_heads,
            num_hidden_layers=config.num_layers,
        )
        self.model = KDAModel(self.kda_config)
        self.time_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, timesteps):
        t_emb = timestep_embedding(timesteps, self.config.hidden_size)
        t_emb = self.time_proj(t_emb)
        x = x + t_emb[:, None, :]
        attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        outputs = self.model(inputs_embeds=x, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        return self.out(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/path/to/imagenet")
    parser.add_argument("--output_dir", type=str, default="./denoiser_checkpoint")
    args = parser.parse_args()

    training_config = TrainingConfig(dataset_root=args.dataset_root, output_dir=args.output_dir)
    model_config = ModelConfig()

    accelerator = Accelerator(mixed_precision="fp16")


    train_dataset = datasets.ImageNet(root=training_config.dataset_root, split="train")
    val_dataset = datasets.ImageNet(root=training_config.dataset_root, split="val")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_config.train_batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=training_config.eval_batch_size, shuffle=False, num_workers=4
    )

    vae = AutoencoderKL.from_pretrained(training_config.vae_model, subfolder="vae").to(accelerator.device)
    vae.eval()
    image_processor = AutoImageProcessor.from_pretrained(training_config.vae_model, subfolder="vae")

    scheduler = DDPMScheduler(num_train_timesteps=training_config.num_train_timesteps)

    model = TransformerDenoiser(model_config)
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    for epoch in range(training_config.num_epochs):
        model.train()
        for batch in train_dataloader:
            images = batch[0]  # (images, labels)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                packed_latents = _pack_latents(
                    latents,
                    latents.shape[0],
                    model_config.latent_channels,
                    model_config.latent_size,
                    model_config.latent_size,
                )

            noise = torch.randn_like(packed_latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (latents.shape[0],), device=packed_latents.device
            ).long()
            noisy_latents = scheduler.add_noise(packed_latents, noise, timesteps)

            noise_pred = model(noisy_latents, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_samples = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch[0]
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                packed_latents = _pack_latents(
                    latents,
                    latents.shape[0],
                    model_config.latent_channels,
                    model_config.latent_size,
                    model_config.latent_size,
                )

                noise = torch.randn_like(packed_latents)
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps, (latents.shape[0],), device=packed_latents.device
                ).long()
                noisy_latents = scheduler.add_noise(packed_latents, noise, timesteps)

                noise_pred = model(noisy_latents, timesteps)
                loss = F.mse_loss(noise_pred, noise)

                val_loss += loss.item() * latents.shape[0]
                num_val_samples += latents.shape[0]

        val_loss /= num_val_samples

        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1}/{training_config.num_epochs} - Val loss: {val_loss:.4f}")
            accelerator.save_state(training_config.output_dir)


if __name__ == "__main__":
    main()