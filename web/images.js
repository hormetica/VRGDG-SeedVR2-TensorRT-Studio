// Images have their own state and viewer; video controls and saved settings stay untouched.
const $ = (id) => document.getElementById(id);
const STORE = 'seedvr-studio-images-v1';
const activeStates = new Set(['queued', 'running', 'cancelling']);
const settingIds = ['image-engine', 'image-model', 'image-scale', 'image-max-edge', 'image-seed', 'image-attention', 'image-blocks', 'image-tiling', 'image-output-root', 'image-create-root'];
let paths = [], review = null, batch = null, selected = '', page = 0, loading = false, usedReview = false, ready = false;
let previousReview = '', lastSourceFolder = '', polling = 0, mode = 'compare';
let zoom = 1, panX = 0, panY = 0, drag = null;
const pageSize = 50;
const busy = () => Boolean(batch && activeStates.has(batch.status));
const status = (text, error = false) => { $('image-status').textContent = text; $('image-status').classList.toggle('image-error', error); };

async function request(path, body) {
  const response = await fetch('/api/images' + path, body === undefined ? {cache: 'no-store'} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail || result));
  return result;
}
const settings = () => ({engine: $('image-engine').value, model_label: $('image-model').value,
  scale: Number($('image-scale').value), max_edge: Number($('image-max-edge').value), seed: Number($('image-seed').value),
  attention_mode: $('image-attention').value, blocks_to_swap: Number($('image-blocks').value), vae_tiling: $('image-tiling').checked});
function savePreferences() {
  const values = Object.fromEntries(settingIds.map((id) => [id, $(id).type === 'checkbox' ? $(id).checked : $(id).value]));
  try { localStorage.setItem(STORE, JSON.stringify({values, lastSourceFolder})); } catch (_) { /* Private/full storage must not block a render. */ }
}
function rows() {
  const items = batch?.kind === 'batch' ? batch.items : review?.items || batch?.items || [];
  return items.map((item) => batch?.kind === 'preview' ? batch.items.find((candidate) => candidate.id === item.id) || item : item);
}
function controls() {
  const locked = loading || busy();
  [...settingIds, 'image-add-files', 'image-add-folder', 'image-add-path', 'image-source-path', 'image-browse-output', 'image-clear', 'image-load-history'].forEach((id) => { $(id).disabled = locked || !ready; });
  $('image-review').disabled = locked || !ready || !paths.length;
  $('image-start').disabled = locked || !review || usedReview || !review.items.some((item) => item.status === 'pending');
  $('image-preview').disabled = locked || !review || !review.items.some((item) => item.id === selected && item.status === 'pending');
  $('image-cancel').hidden = !busy(); $('image-cancel').disabled = batch?.status === 'cancelling';
  $('image-retry').hidden = !batch || busy() || batch.kind === 'preview';
  $('image-retry').disabled = loading;
  $('image-open-output').hidden = !batch; $('image-open-output').disabled = loading;
  $('image-log').hidden = !batch; if (batch) $('image-log').href = `/api/images/batches/${batch.id}/log`;
}
function transform() {
  ['image-before', 'image-after'].forEach((id) => { $(id).style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`; });
  $('image-zoom-reset').textContent = `${Math.round(zoom * 100)}%`;
}
const resetZoom = () => { zoom = 1; panX = panY = 0; transform(); };
function imageUrl(id, url) {
  const element = $(id);
  if (element.dataset.url === (url || '')) return;
  element.dataset.url = url || '';
  if (url) element.src = url; else element.removeAttribute('src');
}
function showSelected() {
  const item = rows().find((entry) => entry.id === selected);
  imageUrl('image-before', item?.source_url);
  imageUrl('image-after', item?.output_url);
  const noImage = !item || (mode === 'restored' && !item.output_url);
  $('image-empty').hidden = !noImage;
  $('image-empty').textContent = item ? 'No restored image yet. Preview this image or start the batch.' : 'Add images or a folder, review, then preview a selected image.';
  // Compare without a restored image should still show the original, not an opaque blank pane.
  document.querySelector('.image-restored').hidden = !item?.output_url && mode !== 'restored';
  $('image-selected').textContent = item ? `${item.source}${item.error ? ' · ' + item.error : ''}${item.result?.fallback_reason ? ' · SeedVR fallback: ' + item.result.fallback_reason : ''}` : 'No image selected';
}
function renderRows() {
  const items = rows();
  if (!items.some((item) => item.id === selected)) { selected = items.find((item) => item.status !== 'error')?.id || items[0]?.id || ''; resetZoom(); }
  page = Math.max(0, Math.min(page, Math.ceil(items.length / pageSize) - 1));
  const fragment = document.createDocumentFragment();
  items.slice(page * pageSize, (page + 1) * pageSize).forEach((item) => {
    const tr = document.createElement('tr'); tr.classList.toggle('selected', selected === item.id); tr.tabIndex = 0;
    tr.setAttribute('aria-label', `${item.name}: ${item.status}`);
    const source = document.createElement('td'); const name = document.createElement('strong'); name.textContent = item.name;
    const location = document.createElement('small'); location.textContent = item.source; source.append(name, location);
    const output = document.createElement('td'); output.textContent = item.output_name || 'Not eligible';
    const size = document.createElement('small'); size.textContent = item.width ? `${item.width} × ${item.height} → ${item.target_width} × ${item.target_height}${item.alpha ? ' · alpha' : ''}` : ''; output.append(size);
    const state = document.createElement('td'); state.className = `item-${item.status}`;
    state.textContent = ({pending: 'Ready', running: 'Restoring…', complete: 'Complete', error: 'Error', cancelled: 'Cancelled'})[item.status] || item.status;
    const detail = document.createElement('small');
    detail.textContent = item.error || (item.result ? `${item.result.engine === 'tensorrt' ? 'TensorRT' : 'SeedVR'}${item.result.fallback_reason ? ' · fallback' : ''}` : '');
    detail.title = item.result?.fallback_reason || item.error || ''; state.append(detail);
    const action = document.createElement('td'); const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'image-remove'; remove.textContent = '×'; remove.title = `Remove ${item.name}`;
    remove.disabled = busy() || loading || !review || batch?.kind === 'batch';
    remove.addEventListener('click', (event) => { event.stopPropagation(); paths = paths.filter((path) => path !== item.source); reviewSelection(); });
    action.append(remove); tr.append(source, output, state, action);
    const select = () => { selected = item.id; resetZoom(); renderRows(); controls(); };
    tr.addEventListener('click', select); tr.addEventListener('keydown', (event) => { if (event.target === tr && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); select(); } });
    fragment.append(tr);
  });
  $('image-items').replaceChildren(fragment);
  const completed = items.filter((item) => item.status === 'complete').length;
  const failed = items.filter((item) => item.status === 'error').length;
  const skipped = review?.skipped?.length || batch?.skipped?.length || 0;
  $('image-count').textContent = `${items.length} image${items.length === 1 ? '' : 's'} · ${completed} completed · ${failed} errors${skipped ? ` · ${skipped} unsupported/subfolders skipped` : ''}`;
  $('image-page').textContent = items.length ? `${page * pageSize + 1}–${Math.min(items.length, (page + 1) * pageSize)} of ${items.length}` : '0 images';
  $('image-prev-page').disabled = page === 0; $('image-next-page').disabled = (page + 1) * pageSize >= items.length;
  showSelected();
}
function invalidate() {
  if (busy()) return;
  review = null; usedReview = false;
  savePreferences(); controls();
  if (paths.length) status('Settings or destination changed. Review selection again before starting.');
}
async function reviewSelection(folders = []) {
  if (busy()) return;
  if (!paths.length && !folders.length) {
    review = batch = null; selected = ''; previousReview = ''; usedReview = false;
    $('image-destination').textContent = ''; status('Choose local images or folders to begin.'); renderRows(); controls(); return;
  }
  const previousSource = rows().find((item) => item.id === selected)?.source;
  loading = true; controls(); status('Reading images and checking names, dimensions, and destination…');
  try {
    const result = await request('/review', {paths, folders, output_root: $('image-output-root').value.trim(), settings: settings(),
      create_output_root: $('image-create-root').checked, previous_review: previousReview});
    review = result; previousReview = result.id; batch = null; usedReview = false; polling += 1;
    paths = result.items.map((item) => item.source);
    selected = result.items.find((item) => item.source === previousSource)?.id || '';
    $('image-destination').textContent = `Batch output folder: ${result.output_dir}`;
    $('image-progress').value = 0;
    status(`Review ready · ${result.items.filter((item) => item.status === 'pending').length} readable images${result.duplicates ? ` · ${result.duplicates} duplicate selections removed` : ''}. Originals will not be modified.`);
    savePreferences(); renderRows();
  } catch (error) { review = null; status(error.message, true); }
  finally { loading = false; controls(); }
}
async function pick(kind) {
  loading = true; controls(); status('Opening the Windows picker…');
  try {
    const result = await request('/pick', {kind, initial_dir: kind === 'output_folder' ? $('image-output-root').value : lastSourceFolder});
    loading = false;
    if (!result.paths.length) { status('Selection cancelled. Existing selections were kept.'); return; }
    if (kind === 'output_folder') {
      $('image-output-root').value = result.paths[0]; invalidate(); if (paths.length) await reviewSelection();
    } else if (kind === 'source_folder') {
      lastSourceFolder = result.paths[0]; await reviewSelection(result.paths);
    } else {
      lastSourceFolder = result.paths[0].replace(/[\\/][^\\/]+$/, ''); paths.push(...result.paths); await reviewSelection();
    }
    savePreferences();
  } catch (error) { status(error.message, true); }
  finally { loading = false; controls(); }
}
async function loadHistory() {
  const history = await request('/batches');
  const select = $('image-history'), previous = select.value; select.replaceChildren();
  const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = history.length ? 'Select an image batch…' : 'No image batches yet'; select.append(placeholder);
  history.forEach((entry) => { const option = document.createElement('option'); option.value = entry.id;
    option.textContent = `${new Date(entry.created_at * 1000).toLocaleString()} · ${entry.kind} · ${entry.completed}/${entry.total} · ${entry.status}`;
    option.title = entry.output_dir; select.append(option); });
  select.value = batch?.id || previous;
  return history;
}
function showBatch(result) {
  batch = result; $('image-progress').value = result.progress || 0;
  const elapsed = result.started_at ? Math.floor(result.elapsed_seconds || (Date.now() / 1000 - result.started_at)) : 0;
  status(`${result.message}${elapsed ? ` · ${Math.floor(elapsed / 60)}m ${elapsed % 60}s` : ''}`, result.status === 'error');
  $('image-destination').textContent = `${result.kind === 'preview' ? 'Preview' : 'Batch'} output folder: ${result.output_dir}`;
  renderRows(); controls();
}
async function poll(id, token) {
  if (token !== polling) return;
  try {
    const result = await request(`/batches/${id}`);
    if (token !== polling) return;
    showBatch(result);
    if (activeStates.has(result.status)) { setTimeout(() => poll(id, token), 1000); }
    else { await loadHistory(); controls(); }
  } catch (error) {
    if (token !== polling) return;
    status(`Status connection lost: ${error.message}. Reconnecting…`, true);
    setTimeout(() => poll(id, token), 2500);
  }
}
async function start(preview) {
  if (!review) return;
  loading = true; controls();
  try {
    const result = await request('/batches', {review_id: review.id, preview_item: preview ? selected : ''});
    if (!preview) usedReview = true;
    showBatch(result); poll(result.id, ++polling);
  } catch (error) { status(error.message, true); }
  finally { loading = false; controls(); }
}

$('video-tab').addEventListener('click', () => switchWorkflow(false));
$('images-tab').addEventListener('click', () => switchWorkflow(true));
function switchWorkflow(images) {
  $('video-workspace').hidden = images; $('image-workspace').hidden = !images;
  ['video-tab', 'images-tab'].forEach((id) => { const active = (id === 'images-tab') === images; $(id).classList.toggle('active', active); $(id).setAttribute('aria-pressed', String(active)); });
  if (images) { document.getElementById('before')?.pause(); document.getElementById('after')?.pause(); }
}
$('image-add-files').addEventListener('click', () => pick('files'));
$('image-add-folder').addEventListener('click', () => pick('source_folder'));
$('image-browse-output').addEventListener('click', () => pick('output_folder'));
$('image-add-path').addEventListener('click', () => {
  const path = $('image-source-path').value.trim().replace(/^"(.*)"$/, '$1');
  if (!path) { status('Enter an absolute folder path first.', true); return; }
  lastSourceFolder = path; reviewSelection([path]);
});
$('image-source-path').addEventListener('keydown', (event) => { if (event.key === 'Enter') $('image-add-path').click(); });
settingIds.forEach((id) => $(id).addEventListener('change', invalidate));
function syncModelLimits() {
  const limit = $('image-model').value.startsWith('7B') ? 36 : 32;
  $('image-blocks').max = limit;
  if (Number($('image-blocks').value) > limit) $('image-blocks').value = limit;
}
$('image-model').addEventListener('change', syncModelLimits);
$('image-review').addEventListener('click', () => reviewSelection());
$('image-preview').addEventListener('click', () => start(true));
$('image-start').addEventListener('click', () => start(false));
$('image-clear').addEventListener('click', () => { paths = []; previousReview = ''; reviewSelection(); });
$('image-cancel').addEventListener('click', async () => { if (!batch) return; try { showBatch(await request(`/batches/${batch.id}/cancel`, {})); } catch (error) { status(error.message, true); } });
$('image-retry').addEventListener('click', async () => {
  if (!batch) return; loading = true; controls();
  try { const result = await request(`/batches/${batch.id}/retry`, {}); showBatch(result); poll(result.id, ++polling); }
  catch (error) { status(error.message, true); } finally { loading = false; controls(); }
});
$('image-open-output').addEventListener('click', async () => { if (batch) try { await request(`/batches/${batch.id}/open-folder`, {}); } catch (error) { status(error.message, true); } });
$('image-load-history').addEventListener('click', async () => {
  if (!$('image-history').value) return;
  try {
    const result = await request(`/batches/${$('image-history').value}`);
    review = null; paths = result.items.map((item) => item.source); usedReview = false; previousReview = ''; selected = ''; page = 0;
    showBatch(result); if (activeStates.has(result.status)) poll(result.id, ++polling); else polling += 1;
  } catch (error) { status(error.message, true); }
});
$('image-prev-page').addEventListener('click', () => { page -= 1; renderRows(); });
$('image-next-page').addEventListener('click', () => { page += 1; renderRows(); });
document.querySelectorAll('[data-image-mode]').forEach((button) => button.addEventListener('click', () => {
  mode = button.dataset.imageMode; $('image-stage').dataset.imageView = mode;
  document.querySelectorAll('[data-image-mode]').forEach((other) => other.classList.toggle('active', other === button));
  resetZoom(); showSelected();
}));
$('image-wipe').addEventListener('input', () => $('image-stage').style.setProperty('--image-wipe', `${$('image-wipe').value}%`));
$('image-zoom-reset').addEventListener('click', resetZoom);
$('image-stage').addEventListener('dblclick', resetZoom);
$('image-stage').addEventListener('wheel', (event) => { event.preventDefault(); zoom = Math.max(1, Math.min(8, zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2))); if (zoom === 1) panX = panY = 0; transform(); }, {passive: false});
$('image-stage').addEventListener('pointerdown', (event) => { if (zoom <= 1) return; drag = {x: event.clientX, y: event.clientY, panX, panY}; $('image-stage').setPointerCapture(event.pointerId); });
$('image-stage').addEventListener('pointermove', (event) => { if (!drag) return; panX = drag.panX + event.clientX - drag.x; panY = drag.panY + event.clientY - drag.y; transform(); });
$('image-stage').addEventListener('pointerup', () => { drag = null; });
$('image-stage').addEventListener('pointercancel', () => { drag = null; });
$('image-fullscreen').addEventListener('click', () => document.querySelector('.image-inspector').requestFullscreen?.());
$('image-exit').addEventListener('click', () => $('exit-studio').click());
['image-before', 'image-after'].forEach((id) => $(id).addEventListener('error', () => { $('image-selected').textContent = 'Image location is unavailable or changed since review. Check the source/output folder.'; }));

async function initialize() {
  try {
    const config = await request('/config');
    config.models.forEach((label) => { const option = document.createElement('option'); option.value = option.textContent = label; $('image-model').append(option); });
    $('image-output-root').value = config.default_output_root;
    try {
      const saved = JSON.parse(localStorage.getItem(STORE) || 'null'); lastSourceFolder = saved?.lastSourceFolder || '';
      Object.entries(saved?.values || {}).forEach(([id, value]) => {
        if (!settingIds.includes(id)) return;
        if ($(id).type === 'checkbox') $(id).checked = Boolean(value);
        else if ($(id).tagName !== 'SELECT' || Array.from($(id).options).some((option) => option.value === String(value))) $(id).value = value;
      });
    } catch (_) { /* Use defaults if saved preferences are invalid. */ }
    syncModelLimits();
    ready = true;
    const history = await loadHistory();
    const running = history.find((entry) => activeStates.has(entry.status));
    if (running) { showBatch(await request(`/batches/${running.id}`)); poll(running.id, ++polling); }
  } catch (error) { status(`Image workflow unavailable: ${error.message}. Restart Studio after updating.`, true); }
  controls();
}
initialize();
