from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from PIL import Image
import torch
from torch import nn
from typing import Union, List


def _prepare_image_ids(
    image_latents: List[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
    scale: int = 10,
):
    r"""
    Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

    This function creates a unique coordinate for every pixel/patch across all input latent with different
    dimensions.

    Args:
        image_latents (List[torch.Tensor]):
            A list of image latent feature tensors, typically of shape (C, H, W).
        scale (int, optional):
            A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
            latent is: 'scale + scale * i'. Defaults to 10.

    Returns:
        torch.Tensor:
            The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
            input latents.

    Coordinate Components (Dimension 4):
        - T (Time): The unique index indicating which latent image the coordinate belongs to.
        - H (Height): The row index within that latent image.
        - W (Width): The column index within that latent image.
        - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
    """

    if not isinstance(image_latents, list):
        raise ValueError(
            f"Expected `image_latents` to be a list, got {type(image_latents)}."
        )

    # create time offset for each reference image
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape

        x_ids = torch.cartesian_prod(
            t, torch.arange(height), torch.arange(width), torch.arange(1)
        )
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    image_latent_ids = image_latent_ids.unsqueeze(0)

    return image_latent_ids


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


def _pack_latents(latents):
    """
    pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
    """

    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    return latents


def _unpack_latents_with_ids(
    x: torch.Tensor, x_ids: torch.Tensor
) -> list[torch.Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)


class PretrainedVAE(nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        width: int = 1024,
        height: int = 1024,
    ):
        super().__init__()
        self.vae = (
            AutoencoderKLFlux2.from_pretrained(
                "black-forest-labs/FLUX.2-dev", subfolder="vae"
            )
            .to(device)
            .eval()
        )
        vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = Flux2ImageProcessor(
            vae_scale_factor=vae_scale_factor * 2
        )
        self.latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        )
        self.latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1)
        self.width = width
        self.height = height
        self.device = device

    @torch.no_grad()
    def encode(self, image: Image.Image):
        image = self.image_processor.preprocess(
            image=image, width=self.width, height=self.height
        )
        image = image.to(self.device)
        latents = self.vae.encode(image).latent_dist.sample()
        latents = _patchify_latents(latents)
        ids = _prepare_image_ids([latents]).to(self.device)
        latents = (latents - self.latents_bn_mean) / self.latents_bn_std
        latents = _pack_latents(latents)
        return latents, ids

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, latents_ids: torch.Tensor):
        """
        latents: (B, T, D)
        """
        latents = _unpack_latents_with_ids(latents, latents_ids)
        latents = latents * self.latents_bn_std + self.latents_bn_mean
        latents = _unpatchify_latents(latents)
        decoded = self.vae.decode(latents)
        image = self.image_processor.postprocess(decoded.sample)
        return image


if __name__ == "__main__":
    vae = PretrainedVAE(device="mps", width=256, height=256)
    image = Image.open("tests/dog.jpg").convert("RGB")
    latents, latents_ids = vae.encode(image)
    print("Latents shape:", latents.shape)
    reconstructed_image = vae.decode(latents, latents_ids)
    reconstructed_image[0].show()
