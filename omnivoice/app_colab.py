import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design

# Minimal CSS
custom_css = """
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
"""

def create_app():
    with gr.Blocks(title="Qwen3-TTS") as demo:
        # Main Title
        gr.Markdown("# 🎙️ Qwen3-TTS")
        gr.Markdown("Advanced Text-to-Speech AI | Voice Cloning, Custom Voice & Voice Design")

        with gr.Tab("Voice Cloning"):
            gr.Markdown("### Clone any voice with 3+ seconds of audio")
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    clone_audio = gr.Audio(label="Reference Audio (3+ seconds)", type="filepath")
                    clone_transcript = gr.Textbox(label="Reference Transcript (Optional)", placeholder="What's said in the reference audio... Highly recommended for best quality.", lines=3)
                    clone_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    clone_output = gr.Audio(label="Generated Speech")
                    gr.Markdown("""
                    **Note**: For best results, provide the **Reference Transcript**. 
                    If left empty, the model will fallback to a faster but lower-quality cloning method.
                    """)

            clone_btn.click(voice_clone, inputs=[clone_text, clone_audio, clone_transcript], outputs=clone_output)

        with gr.Tab("Custom Voice"):
            gr.Markdown("### Use 9 preset character voices with style control")
            with gr.Row():
                with gr.Column():
                    custom_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    custom_voice_name = gr.Dropdown(
                        choices=["serena", "vivian", "ono_anna", "sohee", "aiden", "dylan", "eric", "ryan", "uncle_fu"],
                        label="Voice Character", value="serena"
                    )
                    custom_instruction = gr.Textbox(label="Style Instruction (Optional)", placeholder="e.g., 'speak slowly and cheerfully'", lines=2)
                    custom_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    custom_output = gr.Audio(label="Generated Speech")

            custom_btn.click(custom_voice, inputs=[custom_text, custom_voice_name, custom_instruction], outputs=custom_output)

        with gr.Tab("Voice Design"):
            gr.Markdown("### Design a unique voice from text description")
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    design_description = gr.Textbox(label="Voice Description", placeholder="A young female, cheerful, speaking clearly", lines=4)
                    design_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    design_output = gr.Audio(label="Generated Speech")

            design_btn.click(voice_design, inputs=[design_text, design_description], outputs=design_output)

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True, css=custom_css)
