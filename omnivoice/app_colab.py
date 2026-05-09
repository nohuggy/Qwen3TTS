import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
import gradio as gr
from omnivoice.omni_engine_colab import voice_clone, custom_voice, voice_design, transcribe_ref, compile_role
import time
import re

def count_words(text):
    if not text: return 0
    cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))
    lat = len(re.findall(r"[a-zA-Z0-9\']+", text))
    return cjk + lat

custom_css = """
.load-role-btn { min-width: 90px !important; max-width: 90px !important; }
.panel-gap { margin-top: 12px !important; }
"""

def package_zip(text, audio_path, srt_content):
    if not audio_path: return None
    import zipfile
    from omnivoice.omni_engine_colab import get_slug
    slug = get_slug(text)
    os.makedirs("outputs", exist_ok=True)
    zip_path = os.path.join("outputs", f"{slug}.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(audio_path, f"{slug}.wav")
        if srt_content:
            srt_p = os.path.splitext(audio_path)[0] + ".srt"
            with open(srt_p, 'w', encoding='utf-8') as f: f.write(srt_content)
            zf.write(srt_p, f"{slug}.srt")
    return zip_path

def _adv_accordion():
    """Shared Advanced TTS Settings accordion with SRT/Punc toggles inside."""
    with gr.Accordion("Advanced TTS Settings", open=False):
        with gr.Row():
            temp  = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature", info="Rec: 0.8")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top P", info="Rec: 0.9")
        with gr.Row():
            rep   = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty", info="Rec: 1.1")
            sub   = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Subtalker Temperature")
        with gr.Row():
            srt   = gr.Checkbox(label="Generate Subtitles", value=True)
            punc  = gr.Checkbox(label="Smart Punctuation", value=True)
    return temp, top_p, rep, sub, srt, punc

def _run_srt(text, audio_path, start_time, gen_srt):
    """Generator: yields (status_str | srt_str) for SRT phase."""
    if not gen_srt:
        return
    from omnivoice.omni_engine_colab import generate_srt, clean_script
    for s in generate_srt(clean_script(text), audio_path, total_start_time=start_time):
        yield s

def create_app():
    with gr.Blocks(title="Qwen3-TTS", css=custom_css) as demo:
        gr.Markdown("# Qwen3-TTS")
        gr.Markdown("Advanced Text-to-Speech AI | Voice Cloning · Custom Voice · Voice Design")

        # ── shared helpers ──────────────────────────────────────────────────────
        def on_transcribe(audio):
            return transcribe_ref(audio) if audio else ""

        def process_ref_zip(zf):
            if not zf: return None, ""
            import zipfile, shutil, uuid
            out = os.path.abspath("outputs/refs"); os.makedirs(out, exist_ok=True)
            ap, tc = None, ""
            with zipfile.ZipFile(zf.name, 'r') as z:
                for f in z.namelist():
                    if f.endswith('/') or f.startswith('__MACOSX') or os.path.basename(f).startswith('.'): continue
                    uid = str(uuid.uuid4())[:8]
                    if f.lower().endswith(('.wav','.mp3','.flac')):
                        dest = os.path.join(out, f"{uid}_{os.path.basename(f)}")
                        with z.open(f) as s, open(dest,"wb") as t: shutil.copyfileobj(s,t)
                        if os.path.exists(dest) and os.path.getsize(dest)>0: ap = dest
                    if f.lower().endswith('.txt'):
                        with z.open(f) as t:
                            try: tc = t.read().decode('utf-8')
                            except: tc = t.read().decode('gbk', errors='ignore')
            return ap, tc

        def process_ref_txt(tf):
            if not tf: return ""
            try:
                with open(tf.name,'r',encoding='utf-8') as f: return f.read()
            except UnicodeDecodeError:
                with open(tf.name,'r',encoding='gbk') as f: return f.read()

        # ── VOICE CLONING TAB ──────────────────────────────────────────────────
        with gr.Tab("Voice Cloning"):

            # Each role panel returns: name, path_state, audio, text, ref_group, trans, zip, txt, clear
            def _role_panel(label, open_=True, ph="e.g. Alex"):
                with gr.Accordion(label, open=open_):
                    # Row: Role Name + Load Role button (small)
                    with gr.Row():
                        rname  = gr.Textbox(label="Role Name", placeholder=ph, value="", scale=5)
                        rload  = gr.UploadButton("Load Role", file_types=[".qwen3tts"],
                                                  variant="secondary", size="sm", scale=1,
                                                  elem_classes=["load-role-btn"])
                    # Ref section — hidden when a .qwen3tts is loaded
                    with gr.Group(visible=True) as ref_grp:
                        raudio = gr.Audio(label="Reference Audio", type="filepath")
                        rtext  = gr.Textbox(label="Reference Transcript",
                                            placeholder="Text from the audio...")
                        with gr.Row():
                            rtrans = gr.Button("Trans Ref", variant="primary", size="sm")
                            rzip   = gr.UploadButton("Ref Zip", file_types=[".zip"],
                                                      variant="primary", size="sm")
                            rtxt   = gr.UploadButton("Ref Txt", file_types=[".txt"],
                                                      variant="primary", size="sm")
                    rclear = gr.Button("Clear", variant="secondary", size="sm")
                    # Hidden state: stores the .qwen3tts filepath for on_clone
                    rpath  = gr.Textbox(visible=False, value="")
                return rname, rload, raudio, rtext, ref_grp, rtrans, rzip, rtxt, rclear, rpath

            def on_clone(text,
                         n1, p1, a1, t1,
                         n2, p2, a2, t2,
                         n3, p3, a3, t3,
                         temp, top_p, rep, sub, gen_srt, punc):
                start = time.time()
                yield None, "", gr.update(visible=False), "Initializing..."
                rb = []
                if a1 or p1: rb.append({'name':n1,'audio':a1,'text':t1,'qwen3tts_path':p1 or None})
                if a2 or p2: rb.append({'name':n2,'audio':a2,'text':t2,'qwen3tts_path':p2 or None})
                if a3 or p3: rb.append({'name':n3,'audio':a3,'text':t3,'qwen3tts_path':p3 or None})
                ts = time.time(); ap = None; last = ""
                for s in voice_clone(text, rb, gen_srt=False, convert_punc=punc,
                                     temperature=temp, top_p=top_p,
                                     repetition_penalty=rep, subtalker_temperature=sub,
                                     status_callback=lambda m: print(f"UI: {m}")):
                    if isinstance(s, str): last=s; yield None, "", gr.update(visible=False), s
                    else: ap,_ = s
                td = time.time()-ts
                if not ap:
                    if not last.startswith("❌"): yield None, "", gr.update(visible=False), "Generation failed."
                    return
                yield ap, "", gr.update(visible=False), "Audio ready. Aligning subtitles..."
                srt=""; as_=time.time()
                for s in _run_srt(text, ap, start, gen_srt):
                    if isinstance(s,str) and not s.startswith("1\n"):
                        yield ap, "", gr.update(visible=False), s
                    else: srt=s
                ad=time.time()-as_; tot=time.time()-start
                pm=f"Done! Total:{tot:.1f}s | Gen:{td:.1f}s | Asr:{ad:.1f}s | Words:{count_words(text)}"
                yield ap, srt, gr.update(value=package_zip(text,ap,srt),visible=True), pm

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text",
                        placeholder="## Alex ## [shout] Hello!\n## Sara ## [happy] I'm fine!",
                        lines=6)
                    n1,rl1,a1,t1,rg1,rt1,rz1,rx1,rc1,rp1 = _role_panel("Role 1 (Primary)", True, "e.g. Alex")
                    n2,rl2,a2,t2,rg2,rt2,rz2,rx2,rc2,rp2 = _role_panel("Role 2", False, "e.g. Sara")
                    n3,rl3,a3,t3,rg3,rt3,rz3,rx3,rc3,rp3 = _role_panel("Role 3", False, "e.g. Bob")

                with gr.Column():
                    audio_out  = gr.Audio(label="Generated Speech")
                    srt_out    = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    with gr.Group(elem_classes=["panel-gap"]):
                        status_out = gr.Textbox(label="Status", value="", interactive=False, lines=2)
                    temp,top_p,rep,sub,gen_srt,punc = _adv_accordion()
                    btn     = gr.Button("Generate Audio", variant="primary", size="lg")
                    zip_out = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            # Load Role: fill name, store path, hide ref section
            def on_load_role(filepath):
                if not filepath: return gr.update(), "", gr.update(visible=True)
                name = os.path.splitext(os.path.basename(filepath))[0]
                return gr.update(visible=False), name, filepath

            for rl, rg, rname, rpath in [(rl1,rg1,n1,rp1),(rl2,rg2,n2,rp2),(rl3,rg3,n3,rp3)]:
                rl.upload(on_load_role, inputs=[rl], outputs=[rg, rname, rpath])

            # Trans / Zip / Txt per role
            for raudio, rtext, rtrans, rzip, rtxt in [(a1,t1,rt1,rz1,rx1),(a2,t2,rt2,rz2,rx2),(a3,t3,rt3,rz3,rx3)]:
                rtrans.click(on_transcribe, inputs=[raudio], outputs=[rtext])
                rzip.upload(process_ref_zip, inputs=[rzip], outputs=[raudio, rtext])
                rtxt.upload(process_ref_txt, inputs=[rtxt], outputs=[rtext])

            # Clear: restore ref section, wipe everything
            def on_clear(): return gr.update(visible=True), "", None, "", ""
            rc1.click(on_clear, outputs=[rg1, n1, a1, t1, rp1])
            rc2.click(on_clear, outputs=[rg2, n2, a2, t2, rp2])
            rc3.click(on_clear, outputs=[rg3, n3, a3, t3, rp3])

            btn.click(
                on_clone,
                inputs=[input_text,
                        n1,rp1,a1,t1, n2,rp2,a2,t2, n3,rp3,a3,t3,
                        temp,top_p,rep,sub,gen_srt,punc],
                outputs=[audio_out, srt_out, status_out, zip_out]
            )

        # ── CUSTOM VOICE TAB ───────────────────────────────────────────────────
        with gr.Tab("Custom Voice"):
            with gr.Row():
                with gr.Column():
                    ct = gr.Textbox(label="Input Text", placeholder="Enter text...", lines=6)
                    cn = gr.Dropdown(
                        label="Character Voice",
                        choices=["amanda","denis","jessica","kevin","lewis","pippa","stella","tess","vivienne"],
                        value="amanda")
                    ci = gr.Textbox(label="Instruction",
                                    placeholder="e.g. happy, sad, whispered, shouting...", value="Standard")
                    c_temp,c_top,c_rep,c_sub,c_srt,c_punc = _adv_accordion()
                    cb = gr.Button("Generate Audio", variant="primary", size="lg")
                with gr.Column():
                    ca   = gr.Audio(label="Generated Speech")
                    cs   = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    with gr.Group(elem_classes=["panel-gap"]):
                        cst  = gr.Textbox(label="Status", interactive=False, lines=2)
                    czp  = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            def on_custom(text, name, instr, temp, top_p, rep, sub, gen_srt, punc):
                start=time.time(); yield None,"",gr.update(visible=False),"Initializing..."
                ts=time.time(); ap=None; last=""
                for s in custom_voice(text,name,instr,gen_srt=False,convert_punc=punc,
                                      temperature=temp,top_p=top_p,repetition_penalty=rep,
                                      subtalker_temperature=sub,status_callback=lambda m:print(f"UI:{m}")):
                    if isinstance(s,str): last=s; yield None,"",gr.update(visible=False),s
                    else: ap,_=s
                td=time.time()-ts
                if not ap:
                    if not last.startswith("Error"): yield None,"",gr.update(visible=False),"Generation failed."
                    return
                yield ap,"",gr.update(visible=False),"Audio ready. Aligning subtitles..."
                srt=""; as_=time.time()
                for s in _run_srt(text,ap,start,gen_srt):
                    if isinstance(s,str) and not s.startswith("1\n"): yield ap,"",gr.update(visible=False),s
                    else: srt=s
                ad=time.time()-as_; tot=time.time()-start
                pm=f"Done! Total:{tot:.1f}s | Gen:{td:.1f}s | Asr:{ad:.1f}s | Words:{count_words(text)}"
                yield ap,srt,gr.update(value=package_zip(text,ap,srt),visible=True),pm

            cb.click(on_custom,
                     inputs=[ct,cn,ci,c_temp,c_top,c_rep,c_sub,c_srt,c_punc],
                     outputs=[ca,cs,czp,cst])

        # ── VOICE DESIGN TAB ───────────────────────────────────────────────────
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column():
                    dt  = gr.Textbox(label="Input Text", placeholder="Enter text...", lines=6)
                    dd  = gr.Textbox(label="Voice Description",
                                     placeholder="e.g. A middle-aged man with a deep, raspy voice.",
                                     lines=3)
                    d_temp,d_top,d_rep,d_sub,d_srt,d_punc = _adv_accordion()
                    db  = gr.Button("Generate Audio", variant="primary", size="lg")
                with gr.Column():
                    da  = gr.Audio(label="Generated Speech")
                    ds  = gr.Textbox(label="SRT Preview", lines=6, interactive=False)
                    with gr.Group(elem_classes=["panel-gap"]):
                        dst = gr.Textbox(label="Status", interactive=False, lines=2)
                    dzp = gr.DownloadButton("Download ZIP (WAV + SRT)", visible=False)

            def on_design(text, desc, temp, top_p, rep, sub, gen_srt, punc):
                start=time.time(); yield None,"",gr.update(visible=False),"Initializing..."
                ts=time.time(); ap=None; last=""
                for s in voice_design(text,desc,gen_srt=False,convert_punc=punc,
                                      temperature=temp,top_p=top_p,repetition_penalty=rep,
                                      subtalker_temperature=sub,status_callback=lambda m:print(f"UI:{m}")):
                    if isinstance(s,str): last=s; yield None,"",gr.update(visible=False),s
                    else: ap,_=s
                td=time.time()-ts
                if not ap:
                    if not last.startswith("Error"): yield None,"",gr.update(visible=False),"Generation failed."
                    return
                yield ap,"",gr.update(visible=False),"Audio ready. Aligning subtitles..."
                srt=""; as_=time.time()
                for s in _run_srt(text,ap,start,gen_srt):
                    if isinstance(s,str) and not s.startswith("1\n"): yield ap,"",gr.update(visible=False),s
                    else: srt=s
                ad=time.time()-as_; tot=time.time()-start
                pm=f"Done! Total:{tot:.1f}s | Gen:{td:.1f}s | Asr:{ad:.1f}s | Words:{count_words(text)}"
                yield ap,srt,gr.update(value=package_zip(text,ap,srt),visible=True),pm

            db.click(on_design,
                     inputs=[dt,dd,d_temp,d_top,d_rep,d_sub,d_srt,d_punc],
                     outputs=[da,ds,dzp,dst])

        # ── ROLE MAKER TAB ─────────────────────────────────────────────────────
        with gr.Tab("Role Maker"):
            gr.Markdown("""
            ### Compile a Voice → `.qwen3tts`
            Upload reference audio + transcript, name the role, hit **Compile**.
            Load the resulting file via **Load Role** in any Voice Cloning role panel.
            """)
            with gr.Row():
                with gr.Column():
                    mk_audio = gr.Audio(label="Reference Audio", type="filepath")
                    mk_text  = gr.Textbox(label="Transcript",
                                          placeholder="Type transcript or use Trans Ref...", lines=4)
                    with gr.Row():
                        mk_trans = gr.Button("Trans Ref", variant="primary", size="sm")
                        mk_zip   = gr.UploadButton("Ref Zip", file_types=[".zip"], variant="primary", size="sm")
                        mk_txt   = gr.UploadButton("Ref Txt", file_types=[".txt"], variant="primary", size="sm")
                        mk_clear = gr.Button("Clear", variant="secondary", size="sm")
                    mk_name    = gr.Textbox(label="Role Name (used as filename)", placeholder="e.g. Natasha")
                    mk_compile = gr.Button("COMPILE TO .QWEN3TTS", variant="primary", size="lg")

                with gr.Column():
                    mk_dl     = gr.File(label="Download Compiled Voice (.qwen3tts)")
                    mk_status = gr.Textbox(label="Status", value="Ready", interactive=False, lines=2)

            mk_trans.click(on_transcribe, inputs=[mk_audio], outputs=[mk_text])
            mk_zip.upload(process_ref_zip, inputs=[mk_zip], outputs=[mk_audio, mk_text])
            mk_txt.upload(process_ref_txt, inputs=[mk_txt], outputs=[mk_text])
            mk_clear.click(lambda: (None,"",""), outputs=[mk_audio, mk_text, mk_name])

            def on_compile(audio, text, name):
                msgs = []
                def _cb(m): msgs.append(m)
                yield gr.update(), f"Compiling '{name}'..."
                fp, final = compile_role(audio, text, name, status_callback=_cb)
                log = "\n".join(msgs[-4:]) + "\n" + final
                if fp and os.path.exists(fp): yield gr.update(value=fp, visible=True), log.strip()
                else: yield gr.update(visible=False), log.strip()

            mk_compile.click(on_compile,
                             inputs=[mk_audio, mk_text, mk_name],
                             outputs=[mk_dl, mk_status])

    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True)
