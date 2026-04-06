"""
Doodle-to-Realistic Image Pipeline

Uses FLUX.1-dev + ControlNet (lineart) + IP-Adapter to transform
children's doodles into realistic images.

Models:
  - Base: black-forest-labs/FLUX.1-dev
  - ControlNet: promeai/FLUX.1-controlnet-lineart-promeai
  - IP-Adapter: XLabs-AI/flux-ip-adapter (CLIP ViT-L/14 encoder)
"""

import torch
import numpy as np
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import load_image


BASE_MODEL = "black-forest-labs/FLUX.1-dev"
CONTROLNET_MODEL = "promeai/FLUX.1-controlnet-lineart-promeai"
IP_ADAPTER_REPO = "XLabs-AI/flux-ip-adapter"
IP_ADAPTER_WEIGHTS = "ip_adapter.safetensors"
IMAGE_ENCODER = "openai/clip-vit-large-patch14"


def preprocess_doodle(
    image: Image.Image,
    target_size: tuple[int, int] = (1024, 1024),
    invert: bool = True,
) -> Image.Image:
    """Convert a child's doodle into a ControlNet-ready lineart image.

    Args:
        image: Input doodle (any mode).
        target_size: Output resolution (width, height). Will be snapped to multiples of 16.
        invert: If True, auto-invert white-background doodles to white-on-black.
    """
    img = image.convert("L")

    # Snap to multiples of 16
    w, h = target_size
    w -= w % 16
    h -= h % 16
    img = img.resize((w, h), Image.LANCZOS)

    arr = np.array(img)

    if invert and arr.mean() > 127:
        arr = 255 - arr

    return Image.fromarray(arr).convert("RGB")


def build_pipeline(
    device: str = "cuda",
    enable_cpu_offload: bool = True,
    enable_ip_adapter: bool = True,
    ip_adapter_scale: float = 0.7,
) -> FluxControlNetPipeline:
    """Build the FLUX ControlNet pipeline with optional IP-Adapter.

    Args:
        device: Target device.
        enable_cpu_offload: Use model CPU offloading to save VRAM (~24 GB needed without).
        enable_ip_adapter: Whether to load the IP-Adapter for reference-image guidance.
        ip_adapter_scale: Strength of IP-Adapter influence (0.0–1.0).
    """
    controlnet = FluxControlNetModel.from_pretrained(
        CONTROLNET_MODEL,
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )

    if enable_ip_adapter:
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            weight_name=IP_ADAPTER_WEIGHTS,
            image_encoder_pretrained_model_name_or_path=IMAGE_ENCODER,
        )
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    if enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


def generate(
    pipe: FluxControlNetPipeline,
    doodle: Image.Image,
    prompt: str,
    reference_image: Image.Image | None = None,
    negative_prompt: str = "blurry, low quality, distorted, deformed",
    controlnet_conditioning_scale: float = 0.6,
    control_guidance_end: float = 0.8,
    ip_adapter_scale: float | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
) -> Image.Image:
    """Generate a realistic image from a doodle.

    Args:
        pipe: The loaded pipeline.
        doodle: Raw doodle image (will be preprocessed).
        prompt: Text description of desired output.
        reference_image: Optional style/content reference for IP-Adapter.
        negative_prompt: What to avoid.
        controlnet_conditioning_scale: ControlNet strength (0.0–1.0).
        control_guidance_end: Stop applying ControlNet at this fraction of steps.
        ip_adapter_scale: Override IP-Adapter scale for this generation.
        num_inference_steps: Diffusion steps.
        guidance_scale: CFG scale.
        width: Output width (snapped to multiple of 16).
        height: Output height (snapped to multiple of 16).
        seed: Random seed (-1 for random).
    """
    # Snap dimensions
    width -= width % 16
    height -= height % 16

    control_image = preprocess_doodle(doodle, target_size=(width, height))

    if ip_adapter_scale is not None:
        pipe.set_ip_adapter_scale(ip_adapter_scale)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=0.0,
        control_guidance_end=control_guidance_end,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    if reference_image is not None:
        ref = reference_image.resize((width, height), Image.LANCZOS)
        kwargs["ip_adapter_image"] = ref

    result = pipe(**kwargs).images[0]
    return result
