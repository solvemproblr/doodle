"""
Gradio UI for Doodle-to-Video.

Upload a child's doodle → get an animated video with the character in a scene.

Launch with:
    python app.py
"""

import gradio as gr
from PIL import Image
from pipeline import build_pipeline, generate_video, FIXED_PROMPT


pipe = None


def load_pipeline():
    global pipe
    if pipe is None:
        gr.Info("Loading Wan2.1 14B — this may take a few minutes on first run...")
        pipe = build_pipeline()
        gr.Info("Pipeline ready!")
    return pipe


BG_COLORS = {
    "Sky Blue": (135, 206, 235),
    "Meadow Green": (124, 185, 100),
    "Sunset Orange": (255, 183, 120),
    "Sandy Beach": (237, 221, 178),
    "Snowy White": (230, 235, 240),
    "Night Sky": (25, 25, 60),
}


def run(
    doodle: Image.Image | None,
    bg_choice: str,
    num_frames: int,
    steps: int,
    guidance: float,
    seed: int,
):
    if doodle is None:
        raise gr.Error("Please upload a doodle first.")

    p = load_pipeline()
    bg_color = BG_COLORS.get(bg_choice, (135, 206, 235))

    video_path, input_preview = generate_video(
        pipe=p,
        doodle=doodle,
        bg_color=bg_color,
        num_frames=int(num_frames),
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        seed=int(seed),
    )
    return video_path, input_preview


def create_ui():
    with gr.Blocks(title="Doodle to Video") as demo:
        gr.Markdown(
            "# Doodle to Video\n"
            "Upload your child's drawing and watch it come to life in a magical scene!"
        )

        with gr.Row():
            with gr.Column():
                doodle_input = gr.Image(
                    label="Upload doodle",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                bg_choice = gr.Dropdown(
                    choices=list(BG_COLORS.keys()),
                    value="Sky Blue",
                    label="Scene background",
                )

                with gr.Accordion("Advanced settings", open=False):
                    num_frames = gr.Slider(33, 121, value=81, step=8,
                                            label="Number of frames (more = longer video)")
                    steps = gr.Slider(10, 50, value=30, step=1,
                                       label="Inference steps")
                    guidance = gr.Slider(1.0, 10.0, value=5.0, step=0.5,
                                          label="Guidance scale")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

                generate_btn = gr.Button("Generate Video", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="Result")
                input_preview = gr.Image(
                    label="Input to Wan (character on background)",
                    type="pil",
                )

        generate_btn.click(
            fn=run,
            inputs=[doodle_input, bg_choice, num_frames, steps, guidance, seed],
            outputs=[output_video, input_preview],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
