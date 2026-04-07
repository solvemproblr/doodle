"""
Doodle-to-Video Pipeline

1. Remove paper background from the doodle (rembg)
2. Place on a neutral background
3. Feed to Wan2.1 14B I2V with a fixed prompt to generate animated video
"""

import io
import torch
import numpy as np
from PIL import Image
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from rembg import remove


WAN_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

FIXED_PROMPT = (
    "a children's hand-drawn character comes to life in a beautiful natural setting, "
    "the original drawing style is fully preserved, soft golden hour lighting, "
    "gentle ambient motion in the background, leaves and particles floating softly, "
    "studio ghibli atmosphere, cinematic, whimsical"
)


def remove_background(image: Image.Image) -> Image.Image:
    """Remove the paper/white background from a doodle, returning RGBA."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    result = remove(image)
    return result


def prepare_input_image(
    doodle: Image.Image,
    target_size: tuple[int, int] = (832, 480),
    bg_color: tuple[int, int, int] = (135, 206, 235),
) -> Image.Image:
    """Remove background from doodle and place on a colored background.

    Args:
        doodle: Raw doodle image.
        target_size: (width, height) for Wan input. Must be multiples of 16.
        bg_color: Background color RGB. Default is sky blue.
    """
    # Remove paper background
    character = remove_background(doodle)

    # Create background canvas
    w, h = target_size
    canvas = Image.new("RGBA", (w, h), (*bg_color, 255))

    # Fit character into canvas (80% of canvas height, centered)
    char_w, char_h = character.size
    max_h = int(h * 0.8)
    max_w = int(w * 0.8)
    scale = min(max_w / char_w, max_h / char_h)
    new_w = int(char_w * scale)
    new_h = int(char_h * scale)
    character = character.resize((new_w, new_h), Image.LANCZOS)

    # Center on canvas
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    canvas.paste(character, (x, y), mask=character)

    return canvas.convert("RGB")


def build_pipeline(device: str = "cuda") -> WanImageToVideoPipeline:
    """Load Wan2.1 14B I2V pipeline."""
    vae = AutoencoderKLWan.from_pretrained(
        WAN_MODEL,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        WAN_MODEL,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)

    return pipe


def generate_video(
    pipe: WanImageToVideoPipeline,
    doodle: Image.Image,
    prompt: str = FIXED_PROMPT,
    bg_color: tuple[int, int, int] = (135, 206, 235),
    num_frames: int = 81,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    seed: int = -1,
) -> tuple[str, Image.Image]:
    """Generate a video from a doodle.

    Returns:
        Tuple of (video_path, prepared_input_image).
    """
    input_image = prepare_input_image(doodle, bg_color=bg_color)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    output = pipe(
        image=input_image,
        prompt=prompt,
        negative_prompt="blurry, distorted, deformed, ugly, low quality",
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    video_path = "/tmp/doodle_video.mp4"
    export_to_video(output.frames[0], video_path, fps=16)

    return video_path, input_image
