import torch
import torch.nn.functional as F
from fla.models import KDAModel, KDAConfig
from fla.modules import GatedMLP
from fla.models.utils import Cache
from transformers import AutoModelForCausalLM
from torch import nn


def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
    return latents

def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents

class FlaEncoder(torch.nn.Module):
    def __init__(
        self,
        pretrained_feature_extractor: str,
        vae_latent_wh: int,
        vae_latent_c: int,
        n_kda_layers: int = 4,
    ):
        super().__init__()
        self.feature_extractor = AutoModelForCausalLM.from_pretrained(
            pretrained_feature_extractor
        )
        self.config = self.feature_extractor.config
        self.norm1 = torch.nn.LayerNorm(self.config.hidden_size)
        self.feature_transformation = nn.Sequential(
            GatedMLP(
                hidden_size=self.config.hidden_size,
                hidden_ratio=4,
                hidden_act="swish",
                fuse_swiglu=True,
            ),
            nn.Linear(self.config.hidden_size, vae_latent_c * vae_latent_wh),
        )
        self.norm2 = torch.nn.LayerNorm(vae_latent_c * vae_latent_wh)
        self.kda_config = KDAConfig(
            hidden_size=vae_latent_c * vae_latent_wh,
            use_short_conv=True,
            head_dim=vae_latent_wh,
            num_heads=vae_latent_c,
            num_hidden_layers=n_kda_layers,
        )
        self.kda_model = KDAModel(self.kda_config)

    def forward(self, input_ids, attention_mask):
        outputs = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.feature_transformation(hidden_states)
        hidden_states = self.norm2(hidden_states)
        cache = Cache()
        outputs = self.kda_model(
            inputs_embeds=hidden_states, attention_mask=attention_mask, past_key_values=cache, use_cache=True
        )
        states = outputs.past_key_values[-1]  # shape: (B, C, H, W)

        return states["recurrent_state"]


if __name__ == "__main__":
    # Example usage
    from diffusers import AutoencoderKL
    encoder = FlaEncoder(
        pretrained_feature_extractor="gpt2",
        vae_latent_wh=128,
        vae_latent_c=32,
        n_kda_layers=2,
    ).cuda().train()
    input_ids = torch.randint(0, 50257, (1, 10)).cuda()  # Example input
    attention_mask = torch.ones((1, 10)).cuda()  # Example attention mask
    with torch.no_grad():
        output = encoder(input_ids, attention_mask)
        output = _pack_latents(output, *output.shape)
    print("Output shape:", output.shape)  # Expected shape: (2, 128, 32, 32)

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae").cuda().eval()

    image = torch.zeros((1, 3, 1024, 1024)).cuda()  # Example image
    latent = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    latent = _pack_latents(latent, *latent.shape)
    print("Latent shape:", latent.shape)  #
    print(output.mean(), latent.mean())
    print("Difference:", (output - latent).abs().mean())

    from .denoiser import TransformerDenoiser

    denoiser = TransformerDenoiser(
        patch_size=2,
        in_channels=128,
        out_channels=32,
        num_layers=4,
        attention_head_dim=32,
        num_attention_heads=16,
    ).cuda().train()

    noise_pred = denoiser(output, torch.tensor([10]).cuda())
    print("Noise prediction shape:", noise_pred.shape)

    