from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from PIL import Image
import torch


def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(
        batch_size, num_channels_latents * 4, height // 2, width // 2
    )
    return latents


def _unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
    )
    return latents


device = "mps"

vae = (
    AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    .to(device)
    .eval()
)
print(vae.config)
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)
image = Image.open("tests/dog.jpg").convert("RGB")
print(image)
image = image_processor.preprocess(
    image=image, width=256, height=256
)
image = image.to(device)
print(image.shape)  # torch.Size([1, 3, 1024, 1024])

# encode
with torch.no_grad():
    latents = vae.encode(image).latent_dist.sample()
    b, c, h, w = latents.shape
    latents = _patchify_latents(latents)

    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    )
    latent = _unpatchify_latents(latents)
    latents = latents * latents_bn_std + latents_bn_mean
    decoded = vae.decode(latent)

    image = image_processor.postprocess(decoded.sample)
    image[0].show()
