import torch
import torch.nn.functional as F
from fla.models import KDAModel, KDAConfig
from fla.modules import GatedMLP
from fla.models.utils import Cache
from transformers import AutoModelForCausalLM
from torch import nn


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
            hidden_size=self.config.hidden_size,
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
        cache = Cache()
        outputs = self.kda_model(
            hidden_states, attention_mask=attention_mask, cache=cache
        )
        states = outputs.past_key_values[-1]  # shape: (B, C, H, W)

        return states


if __name__ == "__main__":
    # Example usage
    encoder = FlaEncoder(
        pretrained_feature_extractor="gpt2",
        vae_latent_wh=16,
        vae_latent_c=4,
        n_kda_layers=2,
    )
    input_ids = torch.randint(0, 50257, (2, 10))  # Example input
    attention_mask = torch.ones((2, 10))  # Example attention mask
    output = encoder(input_ids, attention_mask)
    print(output.shape)  # Expected output shape: (B, C, H, W)
