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
from PIL import Image, ImageOps
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
) -> Image.Image:
    """Convert a child's doodle into a ControlNet-ready lineart image.

    The ControlNet expects bright/white lines on a pure black background.
    Children's doodles are typically dark lines on white paper, so we invert
    and boost contrast to produce a strong control signal.

    Args:
        image: Input doodle (any mode — RGB, RGBA, L, etc.).
        target_size: Output resolution (width, height). Snapped to multiples of 16.
    """
    # Handle RGBA: composite onto white background first
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg

    img = image.convert("L")

    # Snap to multiples of 16
    w, h = target_size
    w -= w % 16
    h -= h % 16
    img = img.resize((w, h), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)

    # Determine if this is dark-on-light (typical doodle) or light-on-dark
    is_light_background = arr.mean() > 127

    if is_light_background:
        # Invert: dark lines on white → white lines on black
        arr = 255.0 - arr

    # Boost contrast: stretch the line values to full 0-255 range
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-3:
        arr = (arr - lo) / (hi - lo) * 255.0
    arr = arr.clip(0, 255).astype(np.uint8)

    # Apply a threshold to clean up noise and make lines crisp
    # Pixels above threshold become full white (lines), rest become black
    threshold = 30  # low threshold to keep faint strokes
    arr = np.where(arr > threshold, 255, 0).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def build_pipeline(
    device: str = "cuda",
    enable_cpu_offload: bool = True,
    enable_ip_adapter: bool = True,
    ip_adapter_scale: float = 0.5,
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
    controlnet_conditioning_scale: float = 0.9,
    ip_adapter_scale: float | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
) -> tuple[Image.Image, Image.Image]:
    """Generate a realistic image from a doodle.

    Returns:
        Tuple of (generated_image, preprocessed_control_image) so the user
        can verify the control signal looks correct.
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
    return result, control_image
