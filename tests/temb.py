import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# Your original code
dim = 32
max_period = 10000
timestep = torch.tensor([0.1])  # shape (1,)
scale = 1000

exponent = (
    -math.log(max_period)
    * torch.arange(start=0, end=dim // 2, dtype=torch.float32)
    / (dim // 2)
)
print("exponent:", exponent)

emb = torch.exp(exponent)  # these are the base frequencies 1/p_i
print("emb (frequencies):", emb)

# Multiply timestep by frequencies
emb = timestep[:, None].float() * emb[None, :]  # shape (1, 16)
print("emb (t * 1/p):", emb)

# Scale by 1000 as in the original PE paper
emb = emb * scale
print("emb (scaled):", emb)
# Final sinusoidal embedding
emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # shape (1, 32)
print("Final embedding:", emb.shape, emb)

# ========================
# Visualization with plt
# ========================

emb_np = emb.squeeze(0).numpy()  # (32,)

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# 1. Final positional encoding vector (32 dimensions)
axs[0].bar(range(dim), emb_np, color="tab:blue", alpha=0.8)
axs[0].set_title(f"Positional Encoding at t = {timestep.item()} (dim={dim})")
axs[0].set_xlabel("Embedding dimension")
axs[0].set_ylabel("Value")
axs[0].grid(True, axis="y", alpha=0.3)
axs[0].set_xlim(-0.5, dim - 0.5)

# 2. Cosine part only (first 16 dims)
cos_part = emb_np[:16]
sin_part = emb_np[16:]

x = np.arange(16)
axs[1].plot(x, cos_part, "o-", label="cos( t / p_i )", color="tab:green")
axs[1].plot(x, sin_part, "s-", label="sin( t / p_i )", color="tab:red")
axs[1].set_title("Cosine and Sine components separately")
axs[1].set_xlabel("Frequency index i (0 to 15)")
axs[1].set_ylabel("Value")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# 3. Show the actual frequencies used (1/period)
freqs = torch.exp(exponent).numpy()
periods = 1.0 / freqs

axs[2].semilogx(periods, cos_part, "o-", label="cos component", color="tab:green")
axs[2].semilogx(periods, sin_part, "s-", label="sin component", color="tab:red")
axs[2].set_title("Sinusoid values vs Period (log scale)")
axs[2].set_xlabel("Period (higher = lower frequency)")
axs[2].set_ylabel("cos/sin value")
axs[2].legend()
axs[2].grid(True, which="both", ls="--", alpha=0.5)
axs[
    2
].invert_xaxis()  # higher periods on the left (low freq) → matches original PE paper style

plt.tight_layout()
plt.show()
