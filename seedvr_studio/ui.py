from __future__ import annotations

import gradio as gr

from .backend import MODEL_FILES, backend_status, tensorrt_status
from .cancellation import cancel_current_render
from .hardware import HARDWARE_PRESETS
from .jobs import describe_video, empty_inspection_player_html, render_full, render_preview
from .paths import ensure_workspace

CSS = """
:root { --studio-bg: #090b10; --panel: #121620; --line: #252b39; --accent: #65d6ad; }
body, .gradio-container { background: var(--studio-bg) !important; }
.gradio-container { max-width: 1680px !important; color: #edf2f7; }
#studio-header { border-bottom: 1px solid var(--line); padding: 10px 4px 18px; margin-bottom: 12px; }
#studio-header h1 { font-size: 25px; letter-spacing: -0.02em; margin-bottom: 2px; }
.studio-panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px; }
.primary-action button { background: var(--accent) !important; color: #06110d !important; font-weight: 750 !important; }
#compare-slider { min-height: 420px; }
#status-card { border-left: 3px solid var(--accent); padding-left: 12px; }
.seedvr-inspector { display:flex; flex-direction:column; overflow:hidden; background:#101010; color:#f4f4f5; border:1px solid #343434; border-radius:10px; outline:none; }
.seedvr-inspector button { border:0; border-radius:4px; background:#252525; color:#e5e7eb; cursor:pointer; padding:8px 11px; font:inherit; }
.seedvr-inspector button:hover { background:#3a3a3a; }
.seedvr-inspector button:disabled { cursor:default; opacity:.48; }
.seedvr-inspector-toolbar { display:flex; align-items:center; justify-content:center; gap:12px; min-height:52px; padding:5px 10px; background:#202020; border-bottom:1px solid #353535; }
.seedvr-mode-group { display:flex; overflow:hidden; border:1px solid #3b3b3b; border-radius:5px; }
.seedvr-mode-group button { border-radius:0; border-right:1px solid #3b3b3b; }
.seedvr-mode-group button:last-child { border-right:0; }
.seedvr-mode-group button.active { background:#555; color:#fff; box-shadow:inset 0 -2px 0 var(--accent); }
.seedvr-zoom-reset { min-width:58px; }
.seedvr-inspector-stage { position:relative; flex:none; aspect-ratio:16/9; min-height:340px; overflow:hidden; background:#050505; touch-action:none; user-select:none; }
.seedvr-inspector-media { position:absolute; inset:0; overflow:hidden; }
.seedvr-pane { position:absolute; inset:0; overflow:hidden; background:#000; }
.seedvr-pane video { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; background:#000; will-change:transform; }
.seedvr-pane-label { position:absolute; z-index:3; top:10px; padding:5px 8px; border-radius:4px; background:#000a; font-size:12px; font-weight:700; pointer-events:none; }
.seedvr-before-pane .seedvr-pane-label { left:10px; }
.seedvr-after-pane .seedvr-pane-label { right:10px; }
.mode-compare .seedvr-after-pane { clip-path:inset(0 0 0 50%); }
.mode-original .seedvr-after-pane, .mode-original .seedvr-wipe-divider { display:none; }
.mode-restored .seedvr-before-pane, .mode-restored .seedvr-wipe-divider { display:none; }
.mode-restored .seedvr-after-pane { clip-path:none !important; }
.mode-side .seedvr-inspector-media { display:grid; grid-template-columns:1fr 1fr; gap:2px; }
.mode-side .seedvr-pane { position:relative; inset:auto; min-width:0; }
.mode-side .seedvr-after-pane { clip-path:none !important; }
.mode-side .seedvr-wipe-divider { display:none; }
.seedvr-wipe-divider { position:absolute; z-index:5; top:0; bottom:0; left:50%; width:2px; transform:translateX(-1px); background:#fff; box-shadow:0 0 8px #000; cursor:ew-resize; }
.seedvr-wipe-divider span { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); display:grid; place-items:center; width:32px; height:32px; border-radius:50%; background:#fff; color:#111; font-weight:800; }
.seedvr-zoom-help { position:absolute; z-index:8; right:10px; bottom:8px; padding:4px 7px; border-radius:4px; background:#0009; color:#d1d5db; font-size:11px; pointer-events:none; opacity:.8; }
.seedvr-empty-message { position:absolute; inset:0; display:grid; place-content:center; gap:8px; text-align:center; color:#a1a1aa; }
.seedvr-empty-message strong { color:#e4e4e7; font-size:18px; }
.seedvr-scrubber-row { display:flex; align-items:center; gap:10px; padding:8px 12px 4px; background:#222; border-top:1px solid #3b3b3b; }
.seedvr-time { flex:1; accent-color:var(--accent); }
.seedvr-clock { min-width:150px; text-align:right; font-variant-numeric:tabular-nums; font-size:12px; }
.seedvr-transport { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:12px; min-height:54px; padding:4px 12px 9px; background:#222; }
.seedvr-transport label { display:flex; align-items:center; gap:6px; font-size:12px; }
.seedvr-frame-number { width:72px; border:1px solid #454545; border-radius:4px; background:#151515; color:#fff; padding:5px 6px; }
.seedvr-transport-buttons { display:flex; gap:4px; justify-content:center; }
.seedvr-transport-buttons button { min-width:42px; font-size:17px; }
.seedvr-wipe-control { justify-self:end; }
.seedvr-wipe-range { width:130px; accent-color:var(--accent); }
.mode-original .seedvr-wipe-control, .mode-restored .seedvr-wipe-control, .mode-side .seedvr-wipe-control { visibility:hidden; }
.seedvr-inspector-downloads { display:flex; justify-content:flex-end; gap:18px; padding:7px 12px; background:#181818; border-top:1px solid #303030; font-size:12px; }
.seedvr-inspector-downloads a { color:var(--accent); }
.seedvr-inspector:fullscreen { width:100vw; height:100vh; border:0; border-radius:0; }
.seedvr-inspector:fullscreen .seedvr-inspector-stage { flex:1; min-height:0; aspect-ratio:auto; }
@media (max-width: 800px) { .seedvr-inspector-toolbar { flex-wrap:wrap; } .seedvr-inspector-toolbar button span { display:none; } .seedvr-transport { grid-template-columns:1fr; } .seedvr-transport > * { justify-self:center !important; } .seedvr-clock { min-width:120px; } }
footer { display: none !important; }
"""

VIDEO_PLAYER_JS = r"""
const mount = () => element.querySelectorAll('.seedvr-inspector').forEach((root) => {
    if (root.dataset.mounted === '1') return;
    root.dataset.mounted = '1';
    const before = root.querySelector('.seedvr-before');
    const after = root.querySelector('.seedvr-after');
    if (!before || !after) return;
    const afterPane = root.querySelector('.seedvr-after-pane');
    const divider = root.querySelector('.seedvr-wipe-divider');
    const stage = root.querySelector('.seedvr-inspector-stage');
    const play = root.querySelector('.seedvr-play');
    const time = root.querySelector('.seedvr-time');
    const clock = root.querySelector('.seedvr-clock');
    const frameNumber = root.querySelector('.seedvr-frame-number');
    const frameTotal = root.querySelector('.seedvr-frame-total');
    const wipe = root.querySelector('.seedvr-wipe-range');
    const wipeValue = root.querySelector('.seedvr-wipe-control span');
    const zoomReset = root.querySelector('.seedvr-zoom-reset');
    const fps = Math.max(1, Number(root.dataset.fps) || 30);
    let zoom = 1, panX = 0, panY = 0, originX = 50, originY = 50, dragging = '', startX = 0, startY = 0, baseX = 0, baseY = 0;
    const duration = () => Math.min(Number.isFinite(before.duration) ? before.duration : Infinity, Number.isFinite(after.duration) ? after.duration : Infinity);
    const format = (v) => { v = Math.max(0, Number(v) || 0); const m = Math.floor(v / 60); const s = Math.floor(v % 60); const ms = Math.floor((v % 1) * 1000); return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`; };
    const applyTransform = () => { [before, after].forEach((video) => { video.style.transformOrigin = `${originX}% ${originY}%`; video.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`; }); zoomReset.textContent = `${Math.round(zoom * 100)}%`; stage.style.cursor = zoom > 1 ? 'grab' : 'default'; };
    const resetZoom = () => { zoom = 1; panX = panY = 0; originX = originY = 50; applyTransform(); };
    const update = () => { const d = duration(); if (!Number.isFinite(d)) return; time.max = String(d); time.value = String(before.currentTime || 0); clock.textContent = `${format(before.currentTime)} / ${format(d)}`; const total = Math.max(0, Math.floor(d * fps)); frameNumber.max = String(total); frameNumber.value = String(Math.min(total, Math.round((before.currentTime || 0) * fps))); frameTotal.textContent = `/ ${total}`; };
    const seek = (value) => { const d = duration(); const t = Math.max(0, Math.min(Number.isFinite(d) ? d : Number(value), Number(value) || 0)); before.currentTime = t; after.currentTime = t; update(); };
    const pause = () => { before.pause(); after.pause(); play.textContent = '▶'; };
    const togglePlay = async () => { if (before.paused) { try { after.currentTime = before.currentTime; await Promise.all([before.play(), after.play()]); play.textContent = 'Ⅱ'; } catch (_) { pause(); } } else pause(); };
    const stepFrame = (direction) => { pause(); seek((before.currentTime || 0) + direction / fps); };
    const setMode = (mode) => { ['original','restored','compare','side'].forEach((name) => root.classList.toggle(`mode-${name}`, name === mode)); root.querySelectorAll('[data-mode]').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode)); resetZoom(); };
    const setWipe = (value) => { const p = Math.max(0, Math.min(100, Number(value) || 50)); wipe.value = String(p); afterPane.style.clipPath = `inset(0 0 0 ${p}%)`; divider.style.left = `${p}%`; wipeValue.textContent = `${Math.round(p)}%`; };
    const wipeFromPointer = (event) => { const rect = stage.getBoundingClientRect(); setWipe(((event.clientX - rect.left) / rect.width) * 100); };
    root.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
    root.querySelector('.seedvr-fullscreen').addEventListener('click', () => { (root.requestFullscreen || root.webkitRequestFullscreen)?.call(root); });
    zoomReset.addEventListener('click', resetZoom);
    wipe.addEventListener('input', () => setWipe(wipe.value));
    time.addEventListener('input', () => seek(time.value));
    frameNumber.addEventListener('change', () => seek((Number(frameNumber.value) || 0) / fps));
    play.addEventListener('click', togglePlay);
    root.querySelector('.seedvr-prev-frame').addEventListener('click', () => stepFrame(-1));
    root.querySelector('.seedvr-next-frame').addEventListener('click', () => stepFrame(1));
    [before, after].forEach((video) => video.addEventListener('loadedmetadata', update));
    before.addEventListener('timeupdate', () => { if (!before.paused && Math.abs(after.currentTime - before.currentTime) > 0.04) after.currentTime = before.currentTime; update(); });
    before.addEventListener('ended', pause);
    stage.addEventListener('wheel', (event) => { event.preventDefault(); const pane = event.target.closest('.seedvr-pane'); const rect = (pane || stage).getBoundingClientRect(); originX = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100)); originY = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100)); zoom = Math.max(1, Math.min(8, zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2))); if (zoom === 1) panX = panY = 0; applyTransform(); }, {passive:false});
    stage.addEventListener('dblclick', resetZoom);
    stage.addEventListener('pointerdown', (event) => { root.focus(); dragging = event.target.closest('.seedvr-wipe-divider') ? 'wipe' : (zoom > 1 ? 'pan' : ''); if (!dragging) return; stage.setPointerCapture?.(event.pointerId); startX = event.clientX; startY = event.clientY; baseX = panX; baseY = panY; if (dragging === 'wipe') wipeFromPointer(event); });
    stage.addEventListener('pointermove', (event) => { if (dragging === 'wipe') wipeFromPointer(event); else if (dragging === 'pan') { panX = baseX + event.clientX - startX; panY = baseY + event.clientY - startY; applyTransform(); } });
    stage.addEventListener('pointerup', () => { dragging = ''; stage.style.cursor = zoom > 1 ? 'grab' : 'default'; });
    root.addEventListener('keydown', (event) => { if (event.target.matches('input')) return; if (event.code === 'Space') { event.preventDefault(); togglePlay(); } else if (event.key === 'ArrowLeft') stepFrame(-1); else if (event.key === 'ArrowRight') stepFrame(1); });
    setWipe(50); applyTransform(); update();
  });
mount();
watch('value', () => requestAnimationFrame(mount));
"""


def apply_hardware_preset(name, model, resolution, max_resolution, batch_size, attention, blocks, vae_tiling):
    preset = HARDWARE_PRESETS.get(name)
    if preset is None:
        return model, resolution, max_resolution, batch_size, attention, blocks, vae_tiling
    return (preset["model"], preset["resolution"], preset["max_resolution"],
            preset["batch_size"], preset["attention"], preset["blocks"], preset["vae_tiling"])


def build_app() -> gr.Blocks:
    ensure_workspace()
    installed, engine_message = backend_status()
    trt_installed, trt_message = tensorrt_status()
    with gr.Blocks(title="SeedVR Studio") as app:
        gr.Markdown("# SeedVR Studio\nLocal SeedVR2 video restoration, short previews, and visual comparison.", elem_id="studio-header")
        with gr.Row():
            with gr.Column(scale=1, min_width=340, elem_classes="studio-panel"):
                video = gr.Video(label="Source video", sources=["upload"], include_audio=True)
                media_info = gr.Markdown("Drop a video to begin.")
                backend = gr.Dropdown(["SeedVR2 + TensorRT", "SeedVR2 (Legacy)"], value="SeedVR2 + TensorRT", label="Render engine", interactive=True, info="TensorRT currently supports temporal batches 5 and 21.")
                gr.Markdown(f"**SeedVR2:** {engine_message}\n\n**TensorRT:** {trt_message}")
                hardware = gr.Dropdown(list(HARDWARE_PRESETS), value="Custom / manual", label="Hardware / VRAM preset", info="Applies safe starting settings; you can fine-tune them afterward.")
                with gr.Accordion("Model and quality", open=True):
                    model = gr.Dropdown(list(MODEL_FILES), value=list(MODEL_FILES)[0], label="Model")
                    resolution = gr.Dropdown([480, 720, 1080, 1440, 2160], value=1080, label="Target short edge")
                    max_resolution = gr.Dropdown([1920, 2560, 3840, 7680], value=3840, label="Maximum long edge")
                    batch_size = gr.Dropdown([1, 5, 9, 13, 17, 21, 33, 45, 65, 81], value=21, label="Temporal batch (4n+1)")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    color = gr.Dropdown(["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"], value="none", label="Color correction")
                with gr.Accordion("Performance", open=False):
                    attention = gr.Dropdown(["sageattn_2", "sageattn_3", "flash_attn_3", "flash_attn_2", "sdpa"], value="sageattn_2", label="Attention backend", info="Automatically falls through to another installed accelerated backend, then SDPA.")
                    blocks = gr.Slider(0, 36, value=0, step=1, label="Blocks to swap", info="Keep at 0 on a 32 GB RTX 5090 unless a model runs out of VRAM.")
                    vae_tiling = gr.Checkbox(False, label="VAE tiling", info="Saves VRAM but costs speed.")
                    stop_before_vae = gr.State(False)
                with gr.Accordion("Post-processing", open=False):
                    sharpen_enabled = gr.Checkbox(False, label="Extra sharpen", info="GPU unsharp mask applied after TensorRT VAE decode.")
                    sharpen_strength = gr.Slider(0.0, 10.0, value=0.25, step=0.01, label="Sharpen amount")
                    grain_enabled = gr.Checkbox(False, label="Film grain", info="Deterministic per-frame grain; stable across temporal batches.")
                    grain_intensity = gr.Slider(0.0, 0.1, value=0.02, step=0.001, label="Grain intensity")
                    grain_saturation = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Grain color mix", info="0 = monochrome, 1 = colored grain.")
                with gr.Accordion("Skin finishing", open=False):
                    skin_finishing_enabled = gr.Checkbox(False, label="Enable skin finishing", info="Off by default. TensorRT post-processing only.")
                    skin_evenness = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Even skin tone")
                    skin_smoothing = gr.Slider(0.0, 1.0, value=0.20, step=0.05, label="Skin smoothing")
                    skin_redness = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Redness reduction")
                    skin_shine = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Shine reduction")
                    blemish_mode = gr.Dropdown(["off", "subtle", "strong"], value="off", label="Blemish cleanup")
                    preserve_marks = gr.Checkbox(True, label="Preserve freckles and moles")
                    microtexture_enabled = gr.Checkbox(False, label="Face-aware microtexture", info="Enhances existing skin detail with a temporally stabilized skin mask.")
                    microtexture_strength = gr.Slider(0.0, 3.0, value=0.6, step=0.05, label="Microtexture strength")
                gr.Markdown("### Preview range")
                preview_start = gr.Slider(0, 60, value=3, step=0.1, label="Start time (seconds)")
                preview_seconds = gr.Slider(1, 10, value=3, step=0.5, label="Preview length")
                with gr.Row():
                    preview_button = gr.Button("Render preview", variant="primary", elem_classes="primary-action")
                    full_button = gr.Button("Render full video")
                stop_button = gr.Button("Stop current render", variant="stop")
            with gr.Column(scale=3):
                status = gr.Markdown("### Ready", elem_id="status-card")
                gr.Markdown("A progress bar with the current SeedVR2 phase appears while rendering.")
                comparison_player = gr.HTML(value=empty_inspection_player_html(), label="Video inspection viewer", elem_id="video-inspector", js_on_load=VIDEO_PLAYER_JS)
        video.change(describe_video, inputs=video, outputs=[media_info, preview_start, preview_start], show_progress="hidden")
        hardware.change(apply_hardware_preset,
                        inputs=[hardware, model, resolution, max_resolution, batch_size, attention, blocks, vae_tiling],
                        outputs=[model, resolution, max_resolution, batch_size, attention, blocks, vae_tiling],
                        show_progress="hidden")
        common = [video, backend, resolution, max_resolution, batch_size, seed, model, color, attention, blocks, vae_tiling, stop_before_vae, sharpen_enabled, sharpen_strength, grain_enabled, grain_intensity, grain_saturation, microtexture_enabled, microtexture_strength, skin_finishing_enabled, skin_evenness, skin_smoothing, skin_redness, skin_shine, blemish_mode, preserve_marks]
        preview_button.click(render_preview, inputs=[video, backend, preview_start, preview_seconds, *common[2:]], outputs=[status, comparison_player], concurrency_limit=1, show_progress="full")
        full_button.click(render_full, inputs=common, outputs=[status, comparison_player], concurrency_limit=1, show_progress="full")
        stop_button.click(cancel_current_render, outputs=status, queue=False, show_progress="hidden")
    return app
