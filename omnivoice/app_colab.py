import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design

# Custom CSS for clean branding
custom_css = """
/* Creator Badge */
.creator-badge {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 12px 20px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.creator-badge p {
    color: white;
    margin: 0;
    font-size: 1em;
    font-weight: 500;
}

.creator-badge strong {
    font-weight: 700;
    font-size: 1.1em;
}

/* Social Buttons */
.social-buttons {
    display: flex;
    gap: 12px;
    justify-content: center;
    margin: 15px 0 25px 0;
    flex-wrap: wrap;
}

.social-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

.social-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.youtube-btn {
    background: #FF0000;
    color: white !important;
}

.twitter-btn {
    background: #000000;
    color: white !important;
}

/* Footer */
.aiquest-footer {
    text-align: center;
    padding: 15px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 8px;
    margin-top: 20px;
    font-size: 0.9em;
    color: #555;
}

.aiquest-footer strong {
    color: #667eea;
}

/* Button styling */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
"""

def create_app():
    with gr.Blocks(title="Qwen3-TTS - By AIQuest Academy", css=custom_css) as demo:
        # Creator Badge
        gr.HTML("""
            <div class="creator-badge">
                <p>📺 App created by <strong>AIQuest Academy</strong> | Your AI Learning Hub</p>
            </div>
        """)

        # Main Title
        gr.Markdown("# 🎙️ Qwen3-TTS: Voice Clone, Custom Voice & Voice Design")
        gr.Markdown("### Advanced Text-to-Speech AI | Using 1.7B models with SDPA optimization")

        # Social Media Buttons
        gr.HTML("""
            <div class="social-buttons">
                <a href="https://www.youtube.com/@AIQuestAcademy?sub_confirmation=1" target="_blank" class="social-btn youtube-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                    </svg>
                    Subscribe on YouTube
                </a>
                <a href="https://twitter.com/intent/follow?screen_name=AIQuestAcademy" target="_blank" class="social-btn twitter-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                    </svg>
                    Follow on X
                </a>
            </div>
        """)

        with gr.Tab("🎤 Voice Cloning"):
            gr.Markdown("### Clone any voice with 3+ seconds of audio")
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    clone_audio = gr.Audio(label="Reference Audio (3+ seconds)", type="filepath")
                    clone_transcript = gr.Textbox(label="Transcript (Optional)", placeholder="What's said in the audio...", lines=3)
                    clone_fast_mode = gr.Checkbox(label="Fast Mode (skip transcript)", value=True)
                    clone_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    clone_output = gr.Audio(label="Generated Speech")

            clone_btn.click(voice_clone, inputs=[clone_text, clone_audio, clone_transcript, clone_fast_mode], outputs=clone_output)

        with gr.Tab("🎭 Custom Voice"):
            gr.Markdown("### Use 9 preset character voices with style control")
            with gr.Row():
                with gr.Column():
                    custom_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    custom_voice_name = gr.Dropdown(
                        choices=["serena", "vivian", "ono_anna", "sohee", "aiden", "dylan", "eric", "ryan", "uncle_fu"],
                        label="Voice Character", value="serena"
                    )
                    custom_instruction = gr.Textbox(label="Style Instruction (Optional)", placeholder="e.g., 'speak slowly and cheerfully'", lines=2)
                    custom_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    custom_output = gr.Audio(label="Generated Speech")

            custom_btn.click(custom_voice, inputs=[custom_text, custom_voice_name, custom_instruction], outputs=custom_output)

        with gr.Tab("🎨 Voice Design"):
            gr.Markdown("### Design a unique voice from text description")
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=4)
                    design_description = gr.Textbox(label="Voice Description", placeholder="A young female, cheerful, speaking clearly", lines=4)
                    design_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                with gr.Column():
                    design_output = gr.Audio(label="Generated Speech")

            design_btn.click(voice_design, inputs=[design_text, design_description], outputs=design_output)

        # Footer
        gr.HTML("""
            <div class="aiquest-footer">
                <p><strong>🎓 Powered by AIQuest Academy</strong></p>
                <p style="margin-top: 10px; font-size: 0.85em; color: #888;">
                    Powered by Qwen3-TTS | Free & Open Source
                </p>
            </div>
        """)
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
