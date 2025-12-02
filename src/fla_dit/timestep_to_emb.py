import torch.nn as nn
import torch
import math


def timestep_to_emb(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000, scale: int = 1000
):
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half_dim, dtype=torch.float32)
        / half_dim
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None] * scale
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1024,
        emb_dim: int = 256,
        max_period: int = 10000,
        scale: int = 1000,
    ):
        super().__init__()
        self.dim = emb_dim
        self.max_period = max_period
        self.scale = scale
        self.embedder = nn.Sequential(
            nn.Linear(emb_dim, output_dim * 4),
            nn.SiLU(),
            nn.Linear(output_dim * 4, output_dim),
        )

    def forward(self, timesteps: torch.Tensor):
        emb = timestep_to_emb(
            timesteps, dim=self.dim, max_period=self.max_period, scale=self.scale
        )
        emb = self.embedder(emb)
        return emb


if __name__ == "__main__":
    timestep = torch.tensor([0.1])  # shape (1,)
    t_embed = TimestepEmbedder(output_dim=512, emb_dim=256)
    emb = t_embed(timestep)
    print(emb.shape)  # should be (1, 512)
    print(emb[0, :16].tolist())
