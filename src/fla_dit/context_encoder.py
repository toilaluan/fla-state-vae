import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from fla.layers import KimiDeltaAttention
from fla.models.kda.modeling_kda import KDAMLP
from fla.models.utils import Cache


class KDABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        hidden_ratio: int = 4,
        hidden_act: str = "swish",
        fuse_swiglu: bool = True,
    ):
        super().__init__()
        self.attention = KimiDeltaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            use_short_conv=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.mlp = KDAMLP(
            hidden_size=hidden_size,
            hidden_ratio=hidden_ratio,
            hidden_act=hidden_act,
            fuse_swiglu=fuse_swiglu,
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cache = Cache()
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        hidden_states, _, past_key_values = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, past_key_values


class ContextEncoder(nn.Module):
    def __init__(
        self,
        pretrained_feature_name: str = "gpt2",
        num_attention_heads: int = 8,
        num_kda_layers: int = 2,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_feature_name)
        lm_config = self.model.config
        assert lm_config.hidden_size % num_attention_heads == 0, (
            "The hidden size must be divisible by the number of attention heads."
        )
        self.head_dim = lm_config.hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.norm_a = nn.LayerNorm(lm_config.hidden_size)
        self.project_a = nn.Linear(lm_config.hidden_state, lm_config.hidden_size)

        self.layers = nn.ModuleList(
            [
                KDABlock(
                    hidden_size=lm_config.hidden_size,
                    num_heads=num_attention_heads,
                    head_dim=self.head_dim,
                )
                for _ in range(num_kda_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        hidden_states = self.norm_a(hidden_states)
        hidden_states = self.project_a(hidden_states)

        recurrent_states = []
        for i, layer in enumerate(self.layers):
            hidden_states, past_key_values = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            recurrent_states.append(past_key_values[i]["recurrent_state"])

        return hidden_states, recurrent_states


if __name__ == "__main__":
    model = ContextEncoder(
        pretrained_feature_name="HuggingFaceTB/SmolLM2-135M",
        num_attention_heads=8,
        num_kda_layers=2,
    )
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones((2, 16))
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states, recurrent_states = outputs
    print("Hidden states shape:", hidden_states.shape)
    for i, state in enumerate(recurrent_states):
        print(f"Recurrent state {i} shape:", state.shape)
