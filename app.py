"""
Gradio UI for Doodle-to-Realistic Image generation.

Launch with:
    python app.py
"""

import gradio as gr
from PIL import Image
from pipeline import build_pipeline, generate


pipe = None


def load_pipeline(use_ip_adapter: bool = True):
    global pipe
    if pipe is None:
        gr.Info("Loading models — this may take a few minutes on first run...")
        pipe = build_pipeline(
            enable_ip_adapter=use_ip_adapter,
            enable_cpu_offload=True,
        )
        gr.Info("Pipeline ready!")
    return pipe


def run(
    doodle: Image.Image | None,
    reference_image: Image.Image | None,
    prompt: str,
    controlnet_scale: float,
    control_guidance_end: float,
    ip_adapter_scale: float,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int,
):
    if doodle is None:
        raise gr.Error("Please upload or draw a doodle first.")

    if not prompt.strip():
        raise gr.Error("Please enter a text prompt describing the desired output.")

    p = load_pipeline(use_ip_adapter=reference_image is not None)

    result = generate(
        pipe=p,
        doodle=doodle,
        prompt=prompt,
        reference_image=reference_image,
        controlnet_conditioning_scale=controlnet_scale,
        control_guidance_end=control_guidance_end,
        ip_adapter_scale=ip_adapter_scale if reference_image else None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        seed=seed,
    )
    return result


def create_ui():
    with gr.Blocks(title="Doodle to Realistic Image") as demo:
        gr.Markdown("# Doodle to Realistic Image\n"
                     "Upload a child's doodle and get a realistic version back. "
                     "Optionally add a reference photo to guide style/appearance.")

        with gr.Row():
            with gr.Column():
                doodle_input = gr.Image(
                    label="Doodle (upload or draw)",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                reference_input = gr.Image(
                    label="Reference image (optional — guides style via IP-Adapter)",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="a fluffy orange cat sitting on grass, photorealistic, detailed",
                    lines=2,
                )

                with gr.Accordion("Advanced settings", open=False):
                    controlnet_scale = gr.Slider(0.0, 1.5, value=0.6, step=0.05,
                                                  label="ControlNet strength")
                    control_end = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                             label="ControlNet guidance end")
                    ip_scale = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                          label="IP-Adapter strength")
                    steps = gr.Slider(10, 50, value=28, step=1,
                                       label="Inference steps")
                    guidance = gr.Slider(1.0, 10.0, value=3.5, step=0.5,
                                          label="Guidance scale")
                    with gr.Row():
                        width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                        height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Result", type="pil")

        generate_btn.click(
            fn=run,
            inputs=[
                doodle_input, reference_input, prompt_input,
                controlnet_scale, control_end, ip_scale,
                steps, guidance, width, height, seed,
            ],
            outputs=output_image,
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
