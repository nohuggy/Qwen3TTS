import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design, transcribe_ref, compile_role
import time
import re

def count_words(text):
    """Smart word count: CJK characters count as 1, Latin words count as 1"""
    if not text: return 0
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    latin_words = len(re.findall(r"[a-zA-Z0-9\']+", text))
    return cjk_count + latin_words

# Minimal CSS
custom_css = """
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}
.status-msg {
    font-size: 0.9em;
    color: #a0aec0;
    margin-bottom: 8px;
    font-family: monospace;
}
/* Small gap above Status panels */
#vc_status, #cv_status, #vd_status {
    margin-top: 10px !important;
}
/* Load Role button overlaid inside the Role Name textbox */
.role-name-wrap {
    position: relative !important;
}
.role-name-wrap .load-role-btn {
    position: absolute !important;
    right: 6px !important;
    bottom: 6px !important;
    z-index: 100 !important;
    width: fit-content !important;
    max-width: fit-content !important;
}
.role-name-wrap .load-role-btn button {
    font-size: 0.75rem !important;
    padding: 3px 10px !important;
    height: 26px !important;
    min-width: auto !important;
    width: auto !important;
    opacity: 0.8;
    border-radius: 4px !important;
}
.role-name-wrap .load-role-btn button:hover {
    opacity: 1;
}
"""

def package_zip(text, audio_path, srt_content):
    """Package audio and SRT into a ZIP for download using slug-based naming"""
    if not audio_path: return None
    import zipfile
    from omnivoice.omni_engine_colab import get_slug
    slug = get_slug(text)
    os.makedirs("outputs", exist_ok=True)
    zip_path = os.path.join("outputs", f"{slug}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(audio_path, f"{slug}.wav")
        if srt_content:
            base = os.path.splitext(audio_path)[0]
            srt_path = f"{base}.srt"
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            zipf.write(srt_path, f"{slug}.srt")
    return zip_path

def _adv_accordion():
    """Returns (temperature, top_p, repetition_penalty, subtalker_temperature, gen_srt, conv_punc)
    inside a collapsed Advanced TTS Settings accordion. Reused across tabs."""
    with gr.Accordion("Advanced TTS Settings", open=False):
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature",
                                    info="Controls randomness. Rec: 0.8")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P",
                              info="Nucleus sampling. Rec: 0.9")
        with gr.Row():
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty",
                                           info="Reduces repetition. Rec: 1.1")
            subtalker_temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Subtalker Temperature",
                                              info="Secondary voice tokens")
        with gr.Row():
            gen_srt = gr.Checkbox(label="Generate Subtitles", value=True)
            conv_punc = gr.Checkbox(label="Smart Punctuation", value=True)
    return temperature, top_p, repetition_penalty, subtalker_temperature, gen_srt, conv_punc

def create_app():
    with gr.Blocks(title="Qwen3-TTS", css=custom_css) as demo:
        # Main Title
        gr.Markdown("# Qwen3-TTS")
        gr.Markdown("Advanced Text-to-Speech AI | Voice Cloning, Custom Voice & Voice Design")

        # ── shared helpers (defined once, reused across tabs) ──────────────────
        def on_transcribe(audio):
            if not audio: return ""
            return transcribe_ref(audio)

        def process_ref_zip(zip_file):
            if not zip_file: return None, ""
            import zipfile, shutil, uuid
            out_dir = os.path.abspath("outputs/refs")
            os.makedirs(out_dir, exist_ok=True)
            audio_path, text_content = None, ""
            with zipfile.ZipFile(zip_file.name, 'r') as z:
                for f in z.namelist():
                    if f.endswith('/') or f.startswith('__MACOSX') or os.path.basename(f).startswith('.'): continue
                    uid = str(uuid.uuid4())[:8]
                    if f.lower().endswith(('.wav', '.mp3', '.flac')):
                        dest = os.path.join(out_dir, f"{uid}_{os.path.basename(f)}")
                        with z.open(f) as src, open(dest, "wb") as tgt:
                            shutil.copyfileobj(src, tgt)
                        if os.path.exists(dest) and os.path.getsize(dest) > 0:
                            audio_path = dest
                    if f.lower().endswith('.txt'):
                        with z.open(f) as tf:
                            try: text_content = tf.read().decode('utf-8')
                            except: text_content = tf.read().decode('gbk', errors='ignore')
            return audio_path, text_content

        def process_ref_txt(txt_file):
            if not txt_file: return ""
            try:
                with open(txt_file.name, 'r', encoding='utf-8') as tf: return tf.read()
            except UnicodeDecodeError:
                with open(txt_file.name, 'r', encoding='gbk') as tf: return tf.read()

        def auto_name_from_file(f):
            if f: 
                return os.path.splitext(os.path.basename(f))[0], gr.update(visible=False)
            return gr.update(), gr.update()

        # ── VOICE CLONING TAB ──────────────────────────────────────────────────
        with gr.Tab("Voice Cloning"):

            def on_clone(text,
                         name1, audio1, text1, qwen3ts1,
                         name2, audio2, text2, qwen3ts2,
                         name3, audio3, text3, qwen3ts3,
                         temperature, top_p, repetition_penalty, subtalker_temperature,
                         gen_srt, conv_punc):
                start_time = time.time()
                yield None, "", gr.update(visible=False), "Initializing..."
                role_bank_data = []
                if audio1 or qwen3ts1:
                    role_bank_data.append({'name': name1, 'audio': audio1, 'text': text1, 'qwen3tts_path': qwen3ts1})
                if audio2 or qwen3ts2:
                    role_bank_data.append({'name': name2, 'audio': audio2, 'text': text2, 'qwen3tts_path': qwen3ts2})
                if audio3 or qwen3ts3:
                    role_bank_data.append({'name': name3, 'audio': audio3, 'text': text3, 'qwen3tts_path': qwen3ts3})

                tts_start = time.time()
                audio_path, last_status = None, ""
                for status in voice_clone(text, role_bank_data, gen_srt=False, convert_punc=conv_punc,
                                          temperature=temperature, top_p=top_p,
                                          repetition_penalty=repetition_penalty,
                                          subtalker_temperature=subtalker_temperature,
                                          status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                tts_dur = time.time() - tts_start
                if not audio_path:
                    if not last_status.startswith("❌"):
                        yield None, gr.update(), gr.update(visible=False), "Generation failed."
                    return
                yield audio_path, gr.update(), gr.update(visible=False), "Audio ready. Aligning subtitles..."
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt, clean_script
                    for status in generate_srt(clean_script(text), audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                total_dur = time.time() - start_time
                perf_msg = f"Done! Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {count_words(text)}"
                yield audio_path, srt, gr.update(value=package_zip(text, audio_path, srt), visible=True), perf_msg

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text",
                        placeholder="## Alex ## [shout] Hello! [pause:0.5] How are you?\n## Sara ## [happy] I'm fine!",
                        lines=6)

                    def _role_panel(label, open_=True, name_ph="e.g. Alex"):
                        with gr.Accordion(label, open=open_):
                            with gr.Column(elem_classes="role-name-wrap"):
                                rname = gr.Textbox(label="Role Name", placeholder=name_ph, value="")
                                rload = gr.UploadButton("Load Role", file_types=[".qwen3tts"],
                                                        variant="secondary", size="sm",
                                                        elem_classes="load-role-btn")
                            
                            with gr.Column(visible=True) as extra_fields:
                                raudio = gr.Audio(label="Reference Audio", type="filepath")
                                rtext  = gr.Textbox(label="Reference Transcript", placeholder="Text from the audio...")
                                with gr.Row():
                                    rtrans = gr.Button("Trans Ref", variant="primary", size="sm")
                                    rzip   = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                                    rtxt   = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                            
                            rclear = gr.Button("Clear", variant="secondary", size="sm")
                        return rname, rload, raudio, rtext, rtrans, rzip, rtxt, rclear, extra_fields

                    r1_name, r1_load, r1_audio, r1_text, r1_trans, r1_zip, r1_txt, r1_clear, r1_extra = \
                        _role_panel("Role 1 (Primary)", open_=True, name_ph="e.g. Alex")
                    r2_name, r2_load, r2_audio, r2_text, r2_trans, r2_zip, r2_txt, r2_clear, r2_extra = \
                        _role_panel("Role 2", open_=False, name_ph="e.g. Sara")
                    r3_name, r3_load, r3_audio, r3_text, r3_trans, r3_zip, r3_txt, r3_clear, r3_extra = \
                        _role_panel("Role 3", open_=False, name_ph="e.g. Bob")

                with gr.Column():
                    audio_out  = gr.Audio(label="Generated Speech")
                    srt_out    = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    status_out = gr.Textbox(label="Status", value="", interactive=False, lines=2, elem_id="vc_status")
                    temperature, top_p, repetition_penalty, subtalker_temperature, gen_srt, conv_punc = _adv_accordion()
                    btn     = gr.Button("Generate Audio", variant="primary", size="lg")
                    zip_out = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            # Role callbacks
            for rname, rload, raudio, rtext, rtrans, rzip, rtxt, rclear, rextra in [
                (r1_name, r1_load, r1_audio, r1_text, r1_trans, r1_zip, r1_txt, r1_clear, r1_extra),
                (r2_name, r2_load, r2_audio, r2_text, r2_trans, r2_zip, r2_txt, r2_clear, r2_extra),
                (r3_name, r3_load, r3_audio, r3_text, r3_trans, r3_zip, r3_txt, r3_clear, r3_extra),
            ]:
                rtrans.click(on_transcribe, inputs=[raudio], outputs=[rtext])
                rzip.upload(process_ref_zip, inputs=[rzip], outputs=[raudio, rtext])
                rtxt.upload(process_ref_txt, inputs=[rtxt], outputs=[rtext])
                rclear.click(lambda: (None, None, "", gr.update(visible=True)), outputs=[rname, raudio, rtext, rextra])
                rload.upload(auto_name_from_file, inputs=[rload], outputs=[rname, rextra])

            btn.click(
                on_clone,
                inputs=[input_text,
                        r1_name, r1_audio, r1_text, r1_load,
                        r2_name, r2_audio, r2_text, r2_load,
                        r3_name, r3_audio, r3_text, r3_load,
                        temperature, top_p, repetition_penalty, subtalker_temperature,
                        gen_srt, conv_punc],
                outputs=[audio_out, srt_out, zip_out, status_out]
            )

        # ── CUSTOM VOICE TAB ───────────────────────────────────────────────────
        with gr.Tab("Custom Voice"):
            with gr.Row():
                with gr.Column():
                    custom_text = gr.Textbox(label="Input Text", placeholder="Enter text...", lines=6)
                    custom_name = gr.Dropdown(
                        label="Character Voice",
                        choices=["amanda", "denis", "jessica", "kevin", "lewis", "pippa", "stella", "tess", "vivienne"],
                        value="amanda")
                    custom_instr = gr.Textbox(label="Instruction",
                                              placeholder="e.g. happy, sad, whispered, shouting...", value="Standard")
                    c_temp, c_top_p, c_rep, c_sub, c_gen_srt, c_conv_punc = _adv_accordion()
                    custom_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                with gr.Column():
                    custom_audio  = gr.Audio(label="Generated Speech")
                    custom_srt    = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    custom_status = gr.Textbox(label="Status", interactive=False, lines=2, elem_id="cv_status")
                    custom_zip    = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            def on_custom(text, name, instr, temperature, top_p, repetition_penalty, subtalker_temperature, gen_srt, conv_punc):
                start_time = time.time()
                yield None, "", gr.update(visible=False), "Initializing..."
                tts_start = time.time()
                audio_path, last_status = None, ""
                for status in custom_voice(text, name, instr, gen_srt=False, convert_punc=conv_punc,
                                           temperature=temperature, top_p=top_p,
                                           repetition_penalty=repetition_penalty,
                                           subtalker_temperature=subtalker_temperature,
                                           status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                tts_dur = time.time() - tts_start
                if not audio_path:
                    if not last_status.startswith("Error"):
                        yield None, gr.update(), gr.update(visible=False), "Generation failed."
                    return
                yield audio_path, gr.update(), gr.update(visible=False), "Audio ready. Aligning subtitles..."
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt, clean_script
                    for status in generate_srt(clean_script(text), audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                total_dur = time.time() - start_time
                perf_msg = f"Done! Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {count_words(text)}"
                yield audio_path, srt, gr.update(value=package_zip(text, audio_path, srt), visible=True), perf_msg

            custom_btn.click(
                on_custom,
                inputs=[custom_text, custom_name, custom_instr,
                        c_temp, c_top_p, c_rep, c_sub, c_gen_srt, c_conv_punc],
                outputs=[custom_audio, custom_srt, custom_zip, custom_status]
            )

        # ── VOICE DESIGN TAB ───────────────────────────────────────────────────
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column():
                    design_text = gr.Textbox(label="Input Text", placeholder="Enter text...", lines=6)
                    design_desc = gr.Textbox(
                        label="Voice Description",
                        placeholder="e.g. A middle-aged man with a deep, raspy voice and a calm tone.",
                        lines=3)
                    d_temp, d_top_p, d_rep, d_sub, d_gen_srt, d_conv_punc = _adv_accordion()
                    design_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                with gr.Column():
                    design_audio  = gr.Audio(label="Generated Speech")
                    design_srt    = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    design_status = gr.Textbox(label="Status", interactive=False, lines=2, elem_id="vd_status")
                    design_zip    = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            def on_design(text, desc, temperature, top_p, repetition_penalty, subtalker_temperature, gen_srt, conv_punc):
                start_time = time.time()
                yield None, "", gr.update(visible=False), "Initializing..."
                tts_start = time.time()
                audio_path, last_status = None, ""
                for status in voice_design(text, desc, gen_srt=False, convert_punc=conv_punc,
                                           temperature=temperature, top_p=top_p,
                                           repetition_penalty=repetition_penalty,
                                           subtalker_temperature=subtalker_temperature,
                                           status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(status, str):
                        last_status = status
                        yield None, "", gr.update(visible=False), status
                    else:
                        audio_path, _ = status
                tts_dur = time.time() - tts_start
                if not audio_path:
                    if not last_status.startswith("Error"):
                        yield None, gr.update(), gr.update(visible=False), "Generation failed."
                    return
                yield audio_path, gr.update(), gr.update(visible=False), "Audio ready. Aligning subtitles..."
                asr_start = time.time()
                srt = ""
                if gen_srt:
                    from omnivoice.omni_engine_colab import generate_srt, clean_script
                    for status in generate_srt(clean_script(text), audio_path, total_start_time=start_time):
                        if isinstance(status, str) and not status.startswith("1\n"):
                            yield audio_path, gr.update(), gr.update(visible=False), status
                        else:
                            srt = status
                asr_dur = time.time() - asr_start
                total_dur = time.time() - start_time
                perf_msg = f"Done! Total: {total_dur:.1f}s | Gen: {tts_dur:.1f}s | Asr: {asr_dur:.1f}s | Words: {count_words(text)}"
                yield audio_path, srt, gr.update(value=package_zip(text, audio_path, srt), visible=True), perf_msg

            design_btn.click(
                on_design,
                inputs=[design_text, design_desc,
                        d_temp, d_top_p, d_rep, d_sub, d_gen_srt, d_conv_punc],
                outputs=[design_audio, design_srt, design_zip, design_status]
            )

        # ── ROLE MAKER TAB ─────────────────────────────────────────────────────
        with gr.Tab("Role Maker"):
            gr.Markdown("""
            ### Compile a Voice → `.qwen3tts`
            Upload a reference audio clip and its transcript, give the role a name, then hit **Compile**.  
            The resulting `.qwen3tts` file can be loaded into any Role panel in the **Voice Cloning** tab via **Load Role** — no need to re-process reference audio each session.
            """)
            with gr.Row():
                with gr.Column():
                    maker_audio = gr.Audio(label="Reference Audio", type="filepath")
                    maker_text  = gr.Textbox(label="Transcript",
                                             placeholder="Type transcript or use Trans Ref...", lines=4)
                    with gr.Row():
                        maker_trans = gr.Button("Trans Ref", variant="primary", size="sm")
                        maker_zip   = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                        maker_txt   = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                        maker_clear = gr.Button("Clear", variant="secondary", size="sm")
                    maker_name    = gr.Textbox(label="Role Name (used as filename)", placeholder="e.g. Natasha")
                    maker_compile = gr.Button("Compile", variant="primary", size="lg")

                with gr.Column():
                    maker_download = gr.File(label="Download Compiled Voice (.qwen3tts)")
                    maker_status   = gr.Textbox(label="Status", value="", interactive=False, lines=2)

            maker_trans.click(on_transcribe, inputs=[maker_audio], outputs=[maker_text])
            maker_zip.upload(process_ref_zip, inputs=[maker_zip], outputs=[maker_audio, maker_text])
            maker_txt.upload(process_ref_txt, inputs=[maker_txt], outputs=[maker_text])
            maker_clear.click(lambda: (None, "", ""), outputs=[maker_audio, maker_text, maker_name])

            def on_compile(audio, text, name):
                status_msgs = []
                def _cb(msg): status_msgs.append(msg)
                yield gr.update(), f"Starting compilation for '{name}'..."
                filepath, final_msg = compile_role(audio, text, name, status_callback=_cb)
                log = "\n".join(status_msgs[-8:]) + "\n" + final_msg
                if filepath and os.path.exists(filepath):
                    yield gr.update(value=filepath, visible=True), log
                else:
                    yield gr.update(visible=False), log

            maker_compile.click(
                on_compile,
                inputs=[maker_audio, maker_text, maker_name],
                outputs=[maker_download, maker_status]
            )

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
