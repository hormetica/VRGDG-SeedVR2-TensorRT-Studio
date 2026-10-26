const $ = (selector) => document.querySelector(selector);
const before = $('#before');
const after = $('#after');
const inspector = $('#inspector');
const stage = $('#stage');
const afterPane = document.querySelector('.after-pane');
const divider = $('#divider');
let fps = 30;
let syncFrame = 0;
let zoom = 1, panX = 0, panY = 0, originX = 50, originY = 50, drag = '', dragX = 0, dragY = 0, baseX = 0, baseY = 0;
let currentOutputPath = '', currentOutputReprocessable = false;
let selectedSourceFile = null;
let renderBusy = false, recommendingSettings = false;
const SETTINGS_STORAGE_KEY = 'seedvr-studio-render-settings-v2';
const LEGACY_SETTINGS_STORAGE_KEY = 'seedvr-studio-render-settings-v1';
const SAVED_SETTING_IDS = [
  'backend', 'preset', 'output-preset', 'crop-policy', 'chunked-render', 'chunk-seconds',
  'model', 'batch-size', 'seed', 'color', 'attention', 'blocks', 'vae-tiling',
  'gpu-finishing', 'warm-worker', 'scene-cut-detection', 'export-preset',
  'sharpen', 'sharpen-strength', 'microtexture', 'microtexture-strength',
  'skin-finishing', 'skin-evenness', 'skin-smoothing', 'skin-redness', 'skin-shine', 'blemish-mode', 'preserve-marks',
  'grain', 'grain-intensity', 'grain-saturation', 'seam-enabled',
  'preview-start', 'preview-seconds'
];
const loadSavedSettings = () => {
  try {
    const current = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || 'null');
    if (current?.values) return current;
    const legacy = JSON.parse(localStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY) || 'null');
    if (!legacy?.values) return null;
    // The first settings implementation could capture Chromium's restored SDPA
    // value before our intended default was applied. Migrate everything else,
    // but repair attention once; future explicit saves remain authoritative.
    legacy.values.attention = 'sageattn_2';
    legacy.version = 2;
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(legacy));
    return legacy;
  } catch (_) { return null; }
};
const refreshSettingsUi = () => {
  const tensorRT = $('#backend').value.includes('TensorRT');
  const features = window.__seedvrFeatures || {};
  $('#recommend-settings').disabled = renderBusy || recommendingSettings || features.hardware_recommendations !== true;
  $('#recommend-settings').title = features.hardware_recommendations === true ? '' : 'Restart SeedVR Studio to enable hardware recommendations.';
  const capabilities = {'gpu-finishing': features.gpu_finishing, 'warm-worker': features.warm_worker,
    'scene-cut-detection': features.scene_cut_detection, 'export-preset': window.__seedvrExportAvailable};
  ['gpu-finishing', 'warm-worker', 'scene-cut-detection', 'export-preset'].forEach((id) => {
    const element = document.getElementById(id);
    element.disabled = !tensorRT || capabilities[id] !== true;
    element.title = capabilities[id] === true ? '' : 'Restart SeedVR Studio to enable this backend feature.';
  });
  $('#chunk-length-wrap').hidden = !$('#chunked-render').checked;
  $('#seam-warning').hidden = $('#seam-enabled').checked;
  $('#blocks-value').textContent = $('#blocks').value;
  const skinOff = !$('#skin-finishing').checked;
  $('#skin-options').classList.toggle('inactive', skinOff);
  $('#skin-options').querySelectorAll('input,select').forEach((control) => { control.disabled = skinOff; });
  ['sharpen-strength', 'microtexture-strength', 'skin-evenness', 'skin-smoothing', 'skin-redness', 'skin-shine', 'grain-intensity', 'grain-saturation'].forEach((id) => document.getElementById(id)?.dispatchEvent(new Event('input')));
};
const restoreSettings = () => {
  const saved = loadSavedSettings();
  const values = {attention:'sageattn_2', color:'none', ...(saved?.values || {})};
  Object.entries(values).forEach(([id, value]) => {
    const element = document.getElementById(id);
    if (!element) return;
    if (element.type === 'checkbox') { element.checked = Boolean(value); return; }
    if (element.tagName === 'SELECT' && !Array.from(element.options).some((option) => option.value === String(value))) return;
    element.value = String(value);
  });
  // Browser form restoration must never silently replace the unsaved default.
  if (!saved) { $('#backend').value = 'SeedVR2 + TensorRT'; $('#attention').value = 'sageattn_2'; $('#color').value = 'none'; }
  refreshSettingsUi();
  $('#settings-save-status').textContent = saved ? 'Saved settings loaded' : 'Using app defaults';
};
const saveCurrentSettings = () => {
  const values = {};
  SAVED_SETTING_IDS.forEach((id) => { const element = document.getElementById(id); if (element) values[id] = element.type === 'checkbox' ? element.checked : element.value; });
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({version:1, savedAt:new Date().toISOString(), values}));
    $('#settings-save-status').textContent = 'Saved — these settings will load next time';
  } catch (_) { $('#settings-save-status').textContent = 'Could not save settings in this browser'; }
};
// Establish the real defaults immediately, before Chromium restores old form state.
$('#attention').value = 'sageattn_2';
$('#color').value = 'none';

const format = (value) => { const v = Math.max(0, Number(value) || 0); const m = Math.floor(v / 60); const s = Math.floor(v % 60); const ms = Math.floor((v % 1) * 1000); return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`; };
const loadedVideos = () => [before, after].filter((video) => video.hasAttribute('src'));
const playbackVideo = () => before.hasAttribute('src') ? before : after;
const refreshPlaybackButton = () => { const playing = !playbackVideo().paused; $('#play').textContent = playing ? 'Ⅱ' : '▶'; $('#play').setAttribute('aria-label', playing ? 'Pause' : 'Play'); };
const duration = () => { const values = loadedVideos().map((video) => video.duration).filter(Number.isFinite); return values.length ? Math.min(...values) : 0; };
const update = () => { const d = duration(); const currentTime = playbackVideo().currentTime || 0; $('#seek').max = d; $('#seek').value = currentTime; $('#clock').textContent = `${format(currentTime)} / ${format(d)}`; const total = Math.max(0, Math.round(d * fps) - 1); $('#frame').max = total; $('#frame').value = Math.min(total, Math.round(currentTime * fps)); $('#frame-total').textContent = `/ ${total}`; ['seek', 'frame', 'prev', 'play', 'next'].forEach((id) => { $(`#${id}`).disabled = d <= 0; }); };
const seek = (value) => { const t = Math.max(0, Math.min(duration(), Number(value) || 0)); loadedVideos().forEach((video) => { video.currentTime = t; }); update(); };
const stopPlaybackSync = () => { if (syncFrame) cancelAnimationFrame(syncFrame); syncFrame = 0; after.playbackRate = 1; };
const syncTime = (source, target) => { if (!target.hasAttribute('src') || !Number.isFinite(source.duration)) return; if (Math.abs(target.currentTime - source.currentTime) > 0.08) target.currentTime = Math.min(source.currentTime, target.duration || source.currentTime); };
const pause = () => { before.pause(); after.pause(); stopPlaybackSync(); refreshPlaybackButton(); };
const play = async () => { const videos = loadedVideos(); if (!videos.length) return; if (playbackVideo().paused) { try { if (videos.length === 2) after.currentTime = before.currentTime; await Promise.all(videos.map((video) => video.play())); refreshPlaybackButton(); } catch (_) { pause(); } } else pause(); };
const setWipe = (value) => { const p = Math.max(0, Math.min(100, Number(value) || 50)); $('#wipe').value = p; afterPane.style.setProperty('clip-path', `inset(0 0 0 ${p}%)`, inspector.classList.contains('mode-compare') ? 'important' : ''); divider.style.left = `${p}%`; $('#wipe-value').textContent = `${Math.round(p)}%`; };
const applyTransform = () => { [before, after].forEach((video) => { video.style.transformOrigin = `${originX}% ${originY}%`; video.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`; }); $('#zoom-reset').textContent = `${Math.round(zoom * 100)}%`; };
const resetZoom = () => { zoom = 1; panX = panY = 0; originX = originY = 50; applyTransform(); };
const setMode = (mode) => { ['original','restored','compare','side'].forEach((name) => inspector.classList.toggle(`mode-${name}`, name === mode)); document.querySelectorAll('[data-mode]').forEach((button) => { const active = button.dataset.mode === mode; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active)); }); afterPane.style.setProperty('clip-path', mode === 'compare' ? `inset(0 0 0 ${$('#wipe').value}%)` : 'none', 'important'); resetZoom(); updateEmptyState(); };
const updateEmptyState = () => {
  const hasBefore = before.hasAttribute('src'), hasAfter = after.hasAttribute('src');
  const mode = ['original','restored','compare','side'].find((name) => inspector.classList.contains(`mode-${name}`)) || 'compare';
  const ready = mode === 'original' ? hasBefore : mode === 'restored' ? hasAfter : hasBefore && hasAfter;
  $('#empty-state').hidden = ready; inspector.classList.toggle('is-empty', !ready);
  $('#empty-browse').hidden = hasBefore || mode === 'restored';
  $('#wipe').disabled = mode !== 'compare' || !ready;
  $('#zoom-reset').disabled = !ready;
  if (ready) return;
  if (mode === 'restored' || hasBefore) {
    $('#empty-title').textContent = 'Your restored video will appear here';
    $('#empty-subtitle').textContent = 'Render a short preview to see the difference.';
  } else {
    $('#empty-title').textContent = 'Add a video to get started';
    $('#empty-subtitle').textContent = 'Choose a local file, then try a short preview.';
  }
};
const loadVideo = (video, url) => { if (url) video.src = url; else video.removeAttribute('src'); video.load(); updateEmptyState(); update(); };
const setFile = (video, file, nameTarget) => { if (!file) return; pause(); if (video === before) beginNewProject(file); loadVideo(video, URL.createObjectURL(file)); setMode('original'); $(nameTarget).textContent = file.name; $(nameTarget).title = file.name; $('#media-info').textContent = `${file.name} · local file`; };

const sourceInput = $('#source-file');
const sourceDropzone = document.querySelector('label[for="source-file"]');
$('#empty-browse').addEventListener('click', () => sourceInput.click());
sourceDropzone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); sourceInput.click(); } });
const isVideoFile = (file) => Boolean(file && (file.type.startsWith('video/') || /\.(mp4|mov|mkv|avi|webm|m4v|mpeg|mpg)$/i.test(file.name)));
const acceptSourceFile = (file) => {
  if (!isVideoFile(file)) { $('#media-info').textContent = 'Please choose a supported video file.'; return; }
  selectedSourceFile = file;
  setFile(before, file, '#source-name');
};
sourceInput.addEventListener('change', (event) => acceptSourceFile(event.target.files[0]));
[sourceDropzone, stage].forEach((target) => {
  ['dragenter','dragover'].forEach((name) => target.addEventListener(name, (event) => {
    event.preventDefault(); event.stopPropagation(); target.classList.add('drag-over');
  }));
  ['dragleave','drop'].forEach((name) => target.addEventListener(name, (event) => {
    event.preventDefault(); event.stopPropagation(); target.classList.remove('drag-over');
  }));
  target.addEventListener('drop', (event) => acceptSourceFile(event.dataTransfer?.files?.[0]));
});
['dragover','drop'].forEach((name) => document.addEventListener(name, (event) => {
  if (Array.from(event.dataTransfer?.types || []).includes('Files')) event.preventDefault();
}));
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
$('#seek').addEventListener('input', (event) => seek(event.target.value));
$('#frame').addEventListener('change', (event) => seek((Number(event.target.value) || 0) / fps));
$('#prev').addEventListener('click', () => { pause(); seek(before.currentTime - 1 / fps); });
$('#next').addEventListener('click', () => { pause(); seek(before.currentTime + 1 / fps); });
$('#play').addEventListener('click', play);
$('#wipe').addEventListener('input', (event) => setWipe(event.target.value));
$('#zoom-reset').addEventListener('click', resetZoom);
$('#fullscreen').addEventListener('click', () => (inspector.requestFullscreen || inspector.webkitRequestFullscreen)?.call(inspector));
$('#exit-studio').addEventListener('click', async () => { if (!window.confirm('Exit SeedVR Studio? Any active render will stop and GPU/RAM will be released.')) return; $('#render-status').textContent = 'Shutting down…'; try { await fetch('/api/shutdown', {method:'POST', keepalive:true}); } catch (_) {} window.close(); });
before.addEventListener('timeupdate', () => { syncTime(before, after); update(); });
after.addEventListener('timeupdate', () => { if (!before.hasAttribute('src')) update(); });
before.addEventListener('ended', pause); before.addEventListener('loadedmetadata', update); after.addEventListener('loadedmetadata', update);
after.addEventListener('ended', pause);
[before, after].forEach((video) => { ['play', 'pause'].forEach((name) => video.addEventListener(name, refreshPlaybackButton)); });
stage.addEventListener('wheel', (event) => { event.preventDefault(); const rect = stage.getBoundingClientRect(); originX = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100)); originY = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100)); zoom = Math.max(1, Math.min(8, zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2))); if (zoom === 1) panX = panY = 0; applyTransform(); }, { passive:false });
stage.addEventListener('dblclick', resetZoom);
stage.addEventListener('pointerdown', (event) => { drag = event.target.closest('#divider') ? 'wipe' : (zoom > 1 ? 'pan' : ''); if (!drag) return; stage.setPointerCapture?.(event.pointerId); dragX = event.clientX; dragY = event.clientY; baseX = panX; baseY = panY; });
stage.addEventListener('pointermove', (event) => { if (drag === 'wipe') setWipe(((event.clientX - stage.getBoundingClientRect().left) / stage.getBoundingClientRect().width) * 100); if (drag === 'pan') { panX = baseX + event.clientX - dragX; panY = baseY + event.clientY - dragY; applyTransform(); } });
stage.addEventListener('pointerup', () => { drag = ''; });
inspector.addEventListener('keydown', (event) => { if (event.target.closest('input, button, select')) return; if (event.code === 'Space') { event.preventDefault(); play(); } if (event.key === 'ArrowLeft') { event.preventDefault(); $('#prev').click(); } if (event.key === 'ArrowRight') { event.preventDefault(); $('#next').click(); } });

let selectedWorkspaceOutput = null;
const outputPicker = $('#output-picker'); const outputPickerToggle = $('#output-picker-toggle'); const outputPickerMenu = $('#output-picker-menu'); const outputPickerLabel = $('#output-picker-label'); const outputPickerThumb = $('#output-picker-thumb');
const closeOutputPicker = () => { outputPickerMenu.hidden = true; outputPickerToggle.setAttribute('aria-expanded', 'false'); };
const selectWorkspaceOutput = (output) => { selectedWorkspaceOutput = output; outputPickerLabel.textContent = output.name; outputPickerThumb.src = '/api/output-thumbnail?path=' + encodeURIComponent(output.path); outputPickerToggle.classList.add('has-selection'); $('#load-output').disabled = false; closeOutputPicker(); };
outputPickerToggle.addEventListener('click', () => { outputPickerMenu.hidden = !outputPickerMenu.hidden; outputPickerToggle.setAttribute('aria-expanded', String(!outputPickerMenu.hidden)); });
document.addEventListener('click', (event) => { if (!outputPicker.contains(event.target)) closeOutputPicker(); });
outputPicker.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') { closeOutputPicker(); outputPickerToggle.focus(); }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    const items = Array.from(outputPickerMenu.querySelectorAll('button'));
    if (!items.length) return;
    event.preventDefault(); outputPickerMenu.hidden = false; outputPickerToggle.setAttribute('aria-expanded', 'true');
    const index = items.indexOf(document.activeElement);
    const next = event.key === 'ArrowDown' ? (index + 1) % items.length : (index < 0 ? items.length - 1 : (index - 1 + items.length) % items.length);
    items[next].focus();
  }
});
async function loadOutputs() {
  try {
    const response = await fetch('/api/outputs'); if (!response.ok) throw new Error('Could not load outputs'); const outputs = await response.json(); outputPickerMenu.innerHTML = ''; selectedWorkspaceOutput = null; $('#load-output').disabled = true; outputPickerToggle.disabled = !outputs.length; outputPickerLabel.textContent = outputs.length ? 'Select a previous result…' : 'No results yet'; outputPickerThumb.removeAttribute('src'); outputPickerToggle.classList.remove('has-selection');
    outputs.forEach((output) => { const item = document.createElement('button'); item.type = 'button'; item.className = 'output-picker-item'; item.setAttribute('role', 'option'); const thumb = document.createElement('img'); thumb.className = 'output-item-thumb'; thumb.alt = ''; thumb.setAttribute('aria-hidden', 'true'); thumb.src = '/api/output-thumbnail?path=' + encodeURIComponent(output.path); const copy = document.createElement('span'); copy.className = 'output-item-copy'; const name = document.createElement('strong'); name.textContent = output.name; const kind = document.createElement('small'); kind.textContent = output.reprocessable ? 'TensorRT result' : 'Workspace result'; copy.append(name, kind); item.append(thumb, copy); item.addEventListener('click', () => selectWorkspaceOutput(output)); outputPickerMenu.append(item); });
  } catch (_) { outputPickerLabel.textContent = 'Could not load previous results'; }
}$('#load-output').addEventListener('click', () => { const output = selectedWorkspaceOutput; if (!output) return; pause(); currentOutputPath = output.path || output.url; currentOutputReprocessable = Boolean(output.reprocessable); fps = Number(output.fps) || 30; $('#reprocess-post').hidden = !currentOutputReprocessable; $('#open-output-folder').hidden = false; loadVideo(after, output.url); setMode(before.hasAttribute('src') ? 'compare' : 'restored'); $('#media-info').textContent = currentOutputReprocessable ? 'Previous result loaded · post-only reprocess available' : 'Previous workspace result loaded'; });
$('#open-output-folder').addEventListener('click', async () => { if (!currentOutputPath) return; const data = new FormData(); data.append('output_path', currentOutputPath); try { const response = await fetch('/api/open-folder', {method:'POST', body:data}); if (!response.ok) throw new Error(await response.text()); } catch (error) { $('#media-info').textContent = `Could not open folder: ${error.message}`; } });
const chunkToggle = $('#chunked-render');
const chunkLengthWrap = $('#chunk-length-wrap');
chunkToggle.addEventListener('change', () => { chunkLengthWrap.hidden = !chunkToggle.checked; });
$('#chunk-help').addEventListener('click', () => $('#chunk-help-dialog').showModal());
const seamToggle = $('#seam-enabled');
seamToggle.addEventListener('change', () => { $('#seam-warning').hidden = seamToggle.checked; });
$('#skin-finishing').addEventListener('change', refreshSettingsUi);
$('#save-settings').addEventListener('click', saveCurrentSettings);
window.addEventListener('pageshow', () => setTimeout(restoreSettings, 0));
setWipe(50); applyTransform(); updateEmptyState(); update(); loadOutputs();

let activeJob = null, lastFailedJob = null, elapsedTimer = null, elapsedStarted = 0, projectToken = 0, lastProfileJob = null;
$('#performance-report').addEventListener('click', () => { if (lastProfileJob) window.open(`/api/jobs/${lastProfileJob}/profile`, '_blank', 'noopener'); });
$('#load-output').addEventListener('click', () => { lastProfileJob = null; $('#performance-report').hidden = true; });
const beginNewProject = (file) => { projectToken += 1; const previousJob = activeJob; activeJob = null; lastFailedJob = null; lastProfileJob = null; $('#performance-report').hidden = true; currentOutputPath = ''; currentOutputReprocessable = false; stopElapsed(0); setRenderBusy(false); loadVideo(after, ''); $('#reprocess-post').hidden = true; $('#open-output-folder').hidden = true; $('#media-info').textContent = `${file.name} · new project`; $('#render-status').textContent = previousJob ? 'New project ready · previous render continues in background' : 'Ready'; };
const formatElapsed = (seconds) => { const total = Math.max(0, Math.floor(Number(seconds) || 0)); return `${String(Math.floor(total / 60)).padStart(2,'0')}:${String(total % 60).padStart(2,'0')}`; };
const updateEta = (progress) => { const p = Number(progress) || 0; const elapsed = (performance.now() - elapsedStarted) / 1000; if (p < 0.05 || elapsed < 5) { $('#eta-time').textContent = 'Estimating…'; return; } $('#eta-time').textContent = `~${formatElapsed(elapsed * (1 - Math.min(p, 0.99)) / p)}`; };
const startElapsed = () => { clearInterval(elapsedTimer); elapsedStarted = performance.now(); $('#elapsed-time').textContent = '00:00'; $('#eta-time').textContent = 'Estimating…'; elapsedTimer = setInterval(() => { const elapsed = (performance.now() - elapsedStarted) / 1000; $('#elapsed-time').textContent = formatElapsed(elapsed); updateEta(window.__seedvrProgress || 0); }, 250); };
const stopElapsed = (seconds) => { clearInterval(elapsedTimer); elapsedTimer = null; $('#elapsed-time').textContent = formatElapsed(seconds); $('#eta-time').textContent = '—'; };
const formValue = (id) => document.getElementById(id).value;
const formChecked = (id) => document.getElementById(id).checked ? 'true' : 'false';
const renderForm = (type, sourceFps = 0) => {
  const file = selectedSourceFile || $('#source-file').files[0];
  if (!file) throw new Error('Choose an original video first.');
  const data = new FormData();
  data.append('file', file); data.append('job_type', type); data.append('backend_name', formValue('backend'));
  data.append('preview_start', formValue('preview-start')); data.append('preview_seconds', formValue('preview-seconds')); data.append('output_preset', formValue('output-preset')); data.append('crop_policy', formValue('crop-policy'));
  data.append('chunked_render', type === 'full' && $('#chunked-render').checked ? 'true' : 'false'); data.append('chunk_seconds', formValue('chunk-seconds'));
  data.append('batch_size', formValue('batch-size')); data.append('seed', formValue('seed'));
  data.append('model_label', formValue('model')); data.append('color_correction', formValue('color')); data.append('attention_mode', formValue('attention')); data.append('blocks_to_swap', formValue('blocks'));
  data.append('vae_tiling', formChecked('vae-tiling')); data.append('stop_before_vae', 'false'); data.append('sharpen_enabled', formChecked('sharpen')); data.append('sharpen_strength', formValue('sharpen-strength')); data.append('microtexture_enabled', $('#skin-finishing').checked ? formChecked('microtexture') : 'false'); data.append('microtexture_strength', formValue('microtexture-strength')); data.append('grain_enabled', formChecked('grain')); data.append('grain_intensity', formValue('grain-intensity')); data.append('grain_saturation', formValue('grain-saturation'));
  data.append('skin_finishing_enabled', formChecked('skin-finishing')); data.append('skin_evenness', formValue('skin-evenness')); data.append('skin_smoothing', formValue('skin-smoothing')); data.append('skin_redness', formValue('skin-redness')); data.append('skin_shine', formValue('skin-shine')); data.append('blemish_mode', formValue('blemish-mode')); data.append('preserve_marks', formChecked('preserve-marks'));
  data.append('seam_mode', $('#seam-enabled').checked ? 'match' : 'off'); data.append('seam_frames', '2'); data.append('decoder_mode', 'optimized_fast');
  data.append('source_fps', String(sourceFps || 0));
  const tensorRT = formValue('backend').includes('TensorRT');
  data.append('gpu_finishing', tensorRT && !$('#gpu-finishing').disabled ? formChecked('gpu-finishing') : 'false');
  data.append('warm_worker', tensorRT && !$('#warm-worker').disabled ? formChecked('warm-worker') : 'false');
  data.append('scene_cut_detection', tensorRT && !$('#scene-cut-detection').disabled ? formChecked('scene-cut-detection') : 'false');
  data.append('export_preset', tensorRT && !$('#export-preset').disabled ? formValue('export-preset') : 'software_h264');
  return data;
};
const requestSourceFps = () => {
  while (true) {
    const value = window.prompt('Frame rate could not be detected. Please enter the source frame rate (for example 23.976, 29.97, 48, or 60):');
    if (value === null) return null;
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 240) return parsed;
    window.alert('Enter a frame rate between 1 and 240. Decimals are allowed.');
  }
};

const setRenderBusy = (busy) => { renderBusy = busy; $('#recommend-settings').disabled = busy || recommendingSettings || window.__seedvrFeatures?.hardware_recommendations !== true; ['render-preview','render-full','reprocess-post'].forEach((id) => { $(`#${id}`).disabled = busy || (id !== 'reprocess-post' && !selectedSourceFile && !sourceInput.files[0]); }); $('#stop-render').hidden = !busy; $('#render-progress').hidden = !busy; if (busy) { $('#render-progress').value = 0; $('#performance-report').hidden = true; $('#resume-render').hidden = true; lastFailedJob = null; } };
setRenderBusy(false);
const pollJob = async (id, token = projectToken) => {
  const job = await (await fetch(`/api/jobs/${id}`)).json();
  if (token !== projectToken) return;
  window.__seedvrProgress = Number(job.progress) || 0;
  $('#render-progress').value = window.__seedvrProgress;
  updateEta(window.__seedvrProgress);
  $('#render-status').textContent = `${job.message || job.status}${job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : ''}`;
  if (job.status === 'complete') { pause(); stopElapsed(job.elapsed_seconds); setRenderBusy(false); activeJob = null; lastProfileJob = job.id; $('#performance-report').hidden = false; if (Number(job.fps) > 0) fps = Number(job.fps); if (job.original_url) loadVideo(before, job.original_url); if (job.restored_url) loadVideo(after, job.restored_url); setMode(before.hasAttribute('src') ? 'compare' : 'restored'); currentOutputPath = job.output_relative || ''; currentOutputReprocessable = Boolean(job.reprocessable); $('#reprocess-post').hidden = !currentOutputReprocessable; $('#open-output-folder').hidden = !currentOutputPath; $('#media-info').textContent = job.job_type === 'reprocess' ? 'Post-only reprocess complete' : `${job.job_type} complete`; loadOutputs(); return; }
  if (job.status === 'error' || job.status === 'cancelled') {
    stopElapsed(job.elapsed_seconds); setRenderBusy(false); activeJob = null;
    lastProfileJob = job.id; $('#performance-report').hidden = false;
    if (job.status === 'error' && job.failure_code === 'fps_required') {
      const enteredFps = requestSourceFps();
      if (enteredFps !== null) startRender(job.job_type, enteredFps);
      else $('#render-status').textContent = 'Render cancelled · source frame rate is required';
      return;
    }
    lastFailedJob = job.resumable ? job.id : null; $('#resume-render').hidden = !lastFailedJob;
    $('#render-status').textContent = `${job.failure_reason || job.error || job.message || job.status}${job.log_file ? ' · log saved' : ''}`;
    return;
  }
  setTimeout(() => pollJob(id, token).catch((error) => { if (token !== projectToken) return; setRenderBusy(false); $('#render-status').textContent = error.message; }), 800);
};
const startRender = async (type, sourceFps = 0) => { try { setRenderBusy(true); $('#render-status').textContent = `Uploading ${type}…`; const response = await fetch('/api/jobs', { method:'POST', body:renderForm(type, sourceFps) }); if (!response.ok) throw new Error(await response.text()); activeJob = (await response.json()).id; startElapsed(); pollJob(activeJob, projectToken); } catch (error) { stopElapsed(0); setRenderBusy(false); $('#render-status').textContent = error.message; } };
const startReprocess = async () => {
  try {
    if (!currentOutputPath || !currentOutputReprocessable) throw new Error('Load a TensorRT result with saved decoded batches first.');
    const data = new FormData();
    data.append('output_path', currentOutputPath); data.append('seed', formValue('seed'));
    [['sharpen', 'sharpen_enabled'], ['skin-finishing', 'skin_finishing_enabled'], ['preserve-marks', 'preserve_marks'], ['grain', 'grain_enabled']].forEach(([id, key]) => data.append(key, formChecked(id)));
    ['sharpen-strength', 'microtexture-strength', 'skin-evenness', 'skin-smoothing', 'skin-redness', 'skin-shine', 'blemish-mode', 'grain-intensity', 'grain-saturation'].forEach((id) => data.append(id.replaceAll('-', '_'), formValue(id)));
    data.append('microtexture_enabled', $('#skin-finishing').checked ? formChecked('microtexture') : 'false');
    data.append('seam_mode', $('#seam-enabled').checked ? 'match' : 'off'); data.append('seam_frames', '2');
    data.append('scene_cut_detection', !$('#scene-cut-detection').disabled ? formChecked('scene-cut-detection') : 'false');
    data.append('export_preset', !$('#export-preset').disabled ? formValue('export-preset') : 'software_h264');
    setRenderBusy(true); $('#render-status').textContent = 'Starting post-only reprocess…';
    const response = await fetch('/api/reprocess', {method:'POST', body:data});
    if (!response.ok) throw new Error(await response.text());
    activeJob = (await response.json()).id; startElapsed(); pollJob(activeJob, projectToken);
  } catch (error) { stopElapsed(0); setRenderBusy(false); $('#render-status').textContent = error.message; }
};
$('#render-preview').addEventListener('click', () => startRender('preview')); $('#render-full').addEventListener('click', () => startRender('full'));
$('#reprocess-post').addEventListener('click', startReprocess);
$('#resume-render').addEventListener('click', async () => { if (!lastFailedJob) return; try { $('#resume-render').hidden = true; setRenderBusy(true); const response = await fetch(`/api/jobs/${lastFailedJob}/resume`, {method:'POST'}); if (!response.ok) throw new Error(await response.text()); activeJob = lastFailedJob; startElapsed(); pollJob(activeJob, projectToken); } catch (error) { setRenderBusy(false); $('#render-status').textContent = error.message; } });
$('#stop-render').addEventListener('click', async () => { if (!activeJob) return; await fetch(`/api/jobs/${activeJob}/cancel`, {method:'POST'}); });
$('#blocks').addEventListener('input', (event) => { $('#blocks-value').textContent = event.target.value; });
const bindSliderValue = (inputId, outputId, digits) => { const input = $(`#${inputId}`); const output = $(`#${outputId}`); const refresh = () => { output.textContent = Number(input.value).toFixed(digits); }; input.addEventListener('input', refresh); refresh(); };
bindSliderValue('sharpen-strength', 'sharpen-strength-value', 2);
bindSliderValue('microtexture-strength', 'microtexture-strength-value', 2);
bindSliderValue('skin-evenness', 'skin-evenness-value', 2);
bindSliderValue('skin-smoothing', 'skin-smoothing-value', 2);
bindSliderValue('skin-redness', 'skin-redness-value', 2);
bindSliderValue('skin-shine', 'skin-shine-value', 2);
bindSliderValue('grain-intensity', 'grain-intensity-value', 3);
bindSliderValue('grain-saturation', 'grain-saturation-value', 2);
const fillSelect = (id, values) => { const select = $(`#${id}`); select.innerHTML = ''; values.forEach((value) => { const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option); }); };
const defaultBatchSizes = {'SeedVR2 + TensorRT': [5, 21], 'SeedVR2 (Legacy)': [1, 5, 9, 13, 17, 21, 33, 45]};
const updateBatchOptions = (backend, preferred = null) => { const values = (window.__seedvrBatchSizes?.[backend] || defaultBatchSizes[backend] || defaultBatchSizes['SeedVR2 (Legacy)']).map(String); const select = $('#batch-size'); const current = preferred == null ? select.value : String(preferred); select.innerHTML = ''; values.forEach((value) => { const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option); }); select.value = values.includes(current) ? current : (values.includes('21') ? '21' : values[0]); $('#batch-note').textContent = backend === 'SeedVR2 + TensorRT' ? 'TensorRT supports temporal batches 5 and 21.' : 'SeedVR2 (Legacy) supports 4n+1 temporal batch sizes.'; };
const applyHardwareSettings = (settings) => {
  // Validate before changing anything, including when a newer UI meets an older server.
  ['model', 'attention'].forEach((id) => {
    if (!Array.from($(`#${id}`).options).some((option) => option.value === settings[id])) throw new Error('Recommended settings are unavailable in this UI. Restart Studio and try again.');
  });
  const batches = window.__seedvrBatchSizes?.[$('#backend').value] || defaultBatchSizes[$('#backend').value];
  if (!batches.map(Number).includes(Number(settings.batch_size))) throw new Error('Recommended temporal batch is unsupported by this render engine.');
  if (!Number.isInteger(settings.blocks) || settings.blocks < Number($('#blocks').min) || settings.blocks > Number($('#blocks').max) || typeof settings.vae_tiling !== 'boolean') throw new Error('Recommended memory settings are invalid. Restart Studio and try again.');
  $('#model').value = settings.model; $('#attention').value = settings.attention;
  updateBatchOptions($('#backend').value, settings.batch_size);
  $('#blocks').value = settings.blocks; $('#vae-tiling').checked = Boolean(settings.vae_tiling);
  refreshSettingsUi();
};
const hardwareDialog = $('#hardware-dialog');
let pendingHardwareRecommendation = null, hardwareRequestId = 0;
const hardwareContext = () => {
  const params = new URLSearchParams({backend_name: $('#backend').value, output_preset: $('#output-preset').value,
    source_width: String(before.videoWidth || 0), source_height: String(before.videoHeight || 0)});
  return {params, key: `${projectToken}:${params}`};
};
const describeHardware = (hardware) => {
  const gib = (bytes) => bytes == null ? 'unknown' : `${(Number(bytes) / 1024 ** 3).toFixed(1)} GiB`;
  const parts = [];
  if (hardware.gpu_name) parts.push(`${hardware.gpu_name} (${hardware.device})`, `VRAM: ${gib(hardware.total_vram_bytes)} total / ${gib(hardware.free_vram_bytes)} free`);
  if (hardware.total_ram_bytes != null) parts.push(`System RAM: ${gib(hardware.total_ram_bytes)} total / ${gib(hardware.available_ram_bytes)} free`);
  if (hardware.cpu_threads) parts.push(`${hardware.cpu_threads} CPU threads`);
  return parts.join(' · ');
};
$('#recommend-settings').addEventListener('click', async () => {
  const context = hardwareContext();
  const requestId = ++hardwareRequestId;
  pendingHardwareRecommendation = null; recommendingSettings = true; refreshSettingsUi();
  $('#hardware-message').textContent = 'Inspecting hardware…'; $('#hardware-summary').textContent = '';
  $('#hardware-settings').replaceChildren(); $('#hardware-notes').replaceChildren(); $('#hardware-apply').disabled = true;
  hardwareDialog.showModal();
  try {
    const response = await fetch(`/api/hardware/recommendation?${context.params}`, {cache:'no-store'});
    if (!response.ok) throw new Error(response.status === 404 ? 'Restart Studio to enable hardware recommendations.' : 'Could not inspect hardware. Please try again.');
    const recommendation = await response.json();
    if (requestId !== hardwareRequestId || !hardwareDialog.open) return;
    $('#hardware-message').textContent = recommendation.message;
    $('#hardware-summary').textContent = describeHardware(recommendation.hardware || {});
    (recommendation.notes || []).forEach((note) => { const item = document.createElement('li'); item.textContent = note; $('#hardware-notes').append(item); });
    if (recommendation.available && recommendation.settings) {
      const settings = recommendation.settings;
      [['Model', settings.model], ['Temporal batch', settings.batch_size], ['Attention', settings.attention],
        ['Blocks to swap', settings.blocks], ['VAE tiling', settings.vae_tiling ? 'On' : 'Off']].forEach(([label, value]) => {
        const term = document.createElement('dt'); term.textContent = label;
        const detail = document.createElement('dd'); detail.textContent = String(value); $('#hardware-settings').append(term, detail);
      });
      pendingHardwareRecommendation = {settings, context: context.key}; $('#hardware-apply').disabled = false;
    }
  } catch (error) {
    if (requestId === hardwareRequestId && hardwareDialog.open) $('#hardware-message').textContent = `${error.message} Your settings have not changed.`;
  } finally { recommendingSettings = false; refreshSettingsUi(); }
});
hardwareDialog.addEventListener('close', () => {
  // A queued close event from the previous review must not discard a new one.
  if (!hardwareDialog.open) { pendingHardwareRecommendation = null; hardwareRequestId += 1; }
});
$('#hardware-apply').addEventListener('click', () => {
  if (!pendingHardwareRecommendation) return;
  if (renderBusy || pendingHardwareRecommendation.context !== hardwareContext().key) {
    $('#hardware-message').textContent = 'The source or render settings changed. Close this dialog and request a fresh recommendation.';
    $('#hardware-apply').disabled = true; return;
  }
  try {
    applyHardwareSettings(pendingHardwareRecommendation.settings); $('#preset').value = 'Custom / manual';
    $('#hardware-status').textContent = 'Recommendation applied — test a short preview before a full render.';
    $('#settings-save-status').textContent = 'Recommendation applied — not saved yet'; hardwareDialog.close();
  } catch (error) { $('#hardware-message').textContent = error.message; }
});
$('#backend').addEventListener('change', () => { updateBatchOptions($('#backend').value); refreshSettingsUi(); });
const setSkinBackendAvailable = (available) => { $('#skin-backend-warning').hidden = available; $('#skin-finishing').disabled = !available; if (!available) $('#skin-finishing').checked = false; refreshSettingsUi(); };
const loadConfig = async () => {
  try {
    const response = await fetch('/api/config');
    if (!response.ok) throw new Error('Configuration unavailable');
    const config = await response.json();
    window.__seedvrBatchSizes = config.batch_sizes || defaultBatchSizes;
    window.__seedvrFeatures = config.features || {};
    window.__seedvrExportAvailable = Boolean(config.export_presets);
    fillSelect('model', config.models); fillSelect('preset', Object.keys(config.presets));
    fillSelect('output-preset', Object.keys(config.output_presets)); fillSelect('crop-policy', config.crop_policies);
    const saved = loadSavedSettings();
    const savedBackend = String(saved?.values?.backend || 'SeedVR2 + TensorRT');
    const backendAliases = {'TensorRT VAE (experimental)':'SeedVR2 + TensorRT', 'SeedVR2 AI':'SeedVR2 (Legacy)'};
    const requestedBackend = backendAliases[savedBackend] || savedBackend;
    $('#backend').value = Array.from($('#backend').options).some((option) => option.value === requestedBackend) ? requestedBackend : 'SeedVR2 + TensorRT';
    updateBatchOptions($('#backend').value); $('#output-preset').value = '1K / 1080p'; $('#crop-policy').value = 'Preserve original aspect ratio';
    if (config.seam_modes) {
      $('#seam-enabled').disabled = false; $('#seam-enabled').title = 'Color + noise match over 2 frames';
    } else { $('#seam-enabled').title = 'Available after the JS backend is restarted'; }
    const preset = $('#preset');
    preset.addEventListener('change', () => {
      const value = config.presets[preset.value];
      if (!value) return;
      applyHardwareSettings(value); $('#hardware-status').textContent = 'Manual hardware preset applied.';
    });
    restoreSettings(); updateBatchOptions($('#backend').value, $('#batch-size').value);
    setSkinBackendAvailable(config.features?.skin_finishing === true);
    $('#api-dot').classList.toggle('ok', config.backend.seedvr?.ready === true);
    $('#api-status').textContent = config.backend.seedvr?.ready === true ? 'Local · Connected' : 'Setup needed';
    $('.connection').title = config.backend.seedvr?.message || 'All processing stays on this device.';
  } catch (_) {
    window.__seedvrFeatures = {}; window.__seedvrExportAvailable = false;
    $('#backend').value = 'SeedVR2 + TensorRT'; updateBatchOptions($('#backend').value); restoreSettings();
    if (!$('#backend').value) $('#backend').value = 'SeedVR2 + TensorRT';
    updateBatchOptions($('#backend').value, $('#batch-size').value); setSkinBackendAvailable(false);
    $('#api-dot').classList.remove('ok'); $('#api-status').textContent = 'Service unavailable';
  }
};
loadConfig();



$('#engine-help').addEventListener('click', () => $('#engine-help-dialog').showModal());

const updateDialog = document.createElement('dialog');
updateDialog.id = 'update-dialog';
updateDialog.innerHTML = '<form method="dialog"><button class="dialog-close" aria-label="Close">&times;</button><h2>SeedVR Studio updates</h2><p id="update-message">Checking for updates...</p><p id="update-version"></p><div class="render-actions"><button class="secondary-button" value="close">Close</button><button class="primary-button" id="apply-update" type="button" hidden>Update and restart</button></div></form>';
document.body.append(updateDialog);
const updateButton = document.createElement('button');
updateButton.className = 'icon-button';
updateButton.id = 'check-update';
updateButton.title = 'Check for updates';
updateButton.setAttribute('aria-label', 'Check for updates');
updateButton.innerHTML = '&#8635;';
$('#exit-studio').before(updateButton);
const applyUpdateButton = $('#apply-update');
const updateMessage = $('#update-message');
const updateVersion = $('#update-version');

updateButton.addEventListener('click', async () => {
  updateButton.disabled = true;
  updateMessage.textContent = 'Checking for updates...';
  updateVersion.textContent = '';
  applyUpdateButton.hidden = true;
  updateDialog.showModal();
  try {
    const response = await fetch('/api/update/check', {cache:'no-store'});
    const status = await response.json();
    if (!response.ok) throw new Error(status.detail || 'Update check failed.');
    updateMessage.textContent = status.message;
    const versions = [];
    if (status.current_version) versions.push(`Installed: ${status.current_version}`);
    if (status.latest_version) versions.push(`Latest: ${status.latest_version}`);
    updateVersion.textContent = versions.join(' / ');
    applyUpdateButton.hidden = !(status.supported && status.update_available);
  } catch (error) {
    updateMessage.textContent = `Update check failed: ${error.message}`;
  } finally {
    updateButton.disabled = false;
  }
});

applyUpdateButton.addEventListener('click', async () => {
  if (!window.confirm('Update and restart SeedVR Studio now? Active renders must be stopped first.')) return;
  applyUpdateButton.disabled = true;
  updateMessage.textContent = 'Starting the safe updater...';
  try {
    const response = await fetch('/api/update/apply', {method:'POST'});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Update could not start.');
    updateMessage.textContent = payload.message;
    applyUpdateButton.hidden = true;
    setTimeout(() => window.close(), 900);
  } catch (error) {
    updateMessage.textContent = `Update failed: ${error.message}`;
    applyUpdateButton.disabled = false;
  }
});
