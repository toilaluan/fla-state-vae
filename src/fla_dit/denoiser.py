import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0,
    scale: float = 1000,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d tensor"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate, approximate="tanh")


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()
        inner_dim = int(mult * dim)
        dim_out = dim_out or dim
        self.net = nn.Sequential(GEGLU(dim, inner_dim), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.scale_mlp = nn.Linear(cond_dim, embedding_dim, bias=True)
        self.shift_mlp = nn.Linear(cond_dim, embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor):
        emb = self.silu(conditioning)
        scale = self.scale_mlp(emb)[:, None, :]
        shift = self.shift_mlp(emb)[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor):
        timesteps_proj = get_timestep_embedding(timestep, 256)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_states.dtype)
        )
        conditioning = timesteps_emb
        return conditioning


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.norm_q = (
            RMSNorm(attention_head_dim, eps=eps) if qk_norm == "rms_norm" else None
        )
        self.norm_k = (
            RMSNorm(attention_head_dim, eps=eps) if qk_norm == "rms_norm" else None
        )
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate"
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor):
        scale, shift, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        mod_params = self.modulation(temb)
        mod1, mod2 = mod_params.chunk(2, dim=-1)

        normed = self.norm1(hidden_states)
        modulated, gate1 = self._modulate(normed, mod1)

        q = self.to_q(modulated)
        k = self.to_k(modulated)
        v = self.to_v(modulated)

        q = q.view(
            q.shape[0], q.shape[1], self.num_attention_heads, self.attention_head_dim
        )
        k = k.view(
            k.shape[0], k.shape[1], self.num_attention_heads, self.attention_head_dim
        )
        v = v.view(
            v.shape[0], v.shape[1], self.num_attention_heads, self.attention_head_dim
        )

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb_qwen(q, image_rotary_emb, use_real=False)
            k = apply_rotary_emb_qwen(k, image_rotary_emb, use_real=False)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .contiguous()
            .view(attn_output.shape[0], attn_output.shape[2], self.dim)
        )

        attn_output = self.to_out(attn_output)

        hidden_states = hidden_states + gate1 * attn_output

        norm_hidden = self.norm2(hidden_states)
        mod_hidden, gate2 = self._modulate(norm_hidden, mod2)
        img_mlp_output = self.img_mlp(mod_hidden)
        hidden_states = hidden_states + gate2 * img_mlp_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels
        self.proj_in = nn.Linear(in_channels, self.inner_dim)
        self.time_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=1e-6)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True
        )
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor = None):
        hidden_states = self.proj_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        temb = self.time_embed(timestep, hidden_states)

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(block), hidden_states, temb
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


if __name__ == "__main__":
    denoiser = TransformerDenoiser(
        in_channels=128,
        attention_head_dim=12,
        num_attention_heads=8,
        num_layers=2,
        out_channels=32,
    )

    x = torch.randn((1, 4096, 128))
    t = torch.randint(0, 1000, (1,))
    output = denoiser(x, t)
    print(output.shape)
