const $ = (id) => document.getElementById(id);
const state = { config: null, mode: 'design', reference: null, results: [], busy: false, drafts: new Map() };
const modes = { design: 'Design a voice from a description, then use a take as your cloning reference.', clone: 'Give a reference voice new words. Upload a clip or reuse a take from your session.', speech: 'Explore a model’s preset voices and native synthesis controls.' };
const mime = { wav: 'audio/wav', mp3: 'audio/mpeg', flac: 'audio/flac', opus: 'audio/ogg', aac: 'audio/aac', pcm: 'application/octet-stream' };
let referenceURL = null;
let draftKey = null;
let audioContext;
let tooltipTimer;
const tooltip = create('div', 'studio-tooltip');
tooltip.id = 'studio-tooltip'; tooltip.role = 'tooltip'; tooltip.hidden = true;
document.body.append(tooltip);
let tooltipTrigger = null;

function positionTooltip() {
  if (!tooltipTrigger || tooltip.hidden) return;
  const rect = tooltipTrigger.getBoundingClientRect();
  if (rect.bottom < 0 || rect.top > innerHeight) { hideTooltip(); return; }
  const width = tooltip.offsetWidth, height = tooltip.offsetHeight;
  const limit = tooltipTrigger.closest('dialog') ? innerHeight : document.querySelector('.action-footer')?.getBoundingClientRect().top ?? innerHeight;
  tooltip.style.left = `${Math.max(12, Math.min(innerWidth - width - 12, rect.left + rect.width / 2 - width / 2))}px`;
  tooltip.style.top = `${Math.max(12, Math.min(limit - height - 12, rect.bottom + height + 12 > limit ? rect.top - height - 8 : rect.bottom + 8))}px`;
}
function hideTooltip() {
  clearTimeout(tooltipTimer);
  tooltip.hidden = true;
  tooltipTrigger?.removeAttribute('aria-describedby');
  tooltipTrigger = null;
}
function addHelp(host, content, label = 'More information') {
  const trigger = create('span', 'help-trigger', '?');
  trigger.tabIndex = 0; trigger.role = 'button'; trigger.setAttribute('aria-label', label);
  const show = () => {
    clearTimeout(tooltipTimer);
    const text = content().trim(); if (!text) return;
    tooltipTrigger?.removeAttribute('aria-describedby');
    tooltipTrigger = trigger; trigger.setAttribute('aria-describedby', tooltip.id);
    const layer = trigger.closest('dialog') || document.body;
    if (tooltip.parentElement !== layer) layer.append(tooltip);
    tooltip.textContent = text; tooltip.hidden = false;
    positionTooltip();
  };
  trigger.addEventListener('pointerenter', show);
  trigger.addEventListener('focus', show);
  trigger.addEventListener('pointerleave', () => { tooltipTimer = setTimeout(hideTooltip, 120); });
  trigger.addEventListener('blur', hideTooltip);
  trigger.addEventListener('click', (event) => { event.preventDefault(); event.stopPropagation(); show(); });
  trigger.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); show(); } });
  const control = host.matches('label') ? host.querySelector(':scope > input, :scope > select, :scope > textarea') : null;
  host.insertBefore(trigger, control);
}
function prepareHelp() {
  for (const row of document.querySelectorAll('.parameter')) {
    if (row.dataset.helpPrepared) continue;
    row.dataset.helpPrepared = 'true';
    addHelp(row.querySelector('.parameter-title'), () => [...row.querySelectorAll('.help')].filter((item) => !item.hidden).map((item) => item.textContent).join(' '), `Help: ${humanize(row.dataset.name)}`);
  }
  for (const source of document.querySelectorAll('.help')) {
    if (source.id === 'generation-status' || source.closest('.parameter') || source.dataset.helpPrepared) continue;
    source.dataset.helpPrepared = 'true';
    let host;
    if (source.id === 'mode-help') host = $('mode-hint-slot');
    else if (source.classList.contains('intro-help')) host = source.parentElement.querySelector('.eyebrow');
    else if (source.classList.contains('empty-help')) host = source.parentElement.querySelector('h2');
    else if (source.closest('label')) host = source.closest('label');
    else if (source.closest('details')) { const summary = source.closest('details').querySelector('summary'); host = summary.firstElementChild || summary; }
    else if (source.closest('dialog')) host = $('api-key').closest('label');
    if (host) addHelp(host, () => source.textContent, `Help: ${host.querySelector('input, select, textarea')?.id || source.id || 'details'}`);
  }
  for (const host of document.querySelectorAll('[data-help]')) {
    if (host.dataset.helpPrepared) continue;
    host.dataset.helpPrepared = 'true';
    addHelp(host, () => host.dataset.help, 'Help: reusing a take');
  }
}
tooltip.addEventListener('pointerenter', () => clearTimeout(tooltipTimer));
tooltip.addEventListener('pointerleave', hideTooltip);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !tooltip.hidden) { event.preventDefault(); event.stopPropagation(); hideTooltip(); }
});
document.addEventListener('pointerdown', (event) => { if (tooltipTrigger && !tooltipTrigger.contains(event.target) && !tooltip.contains(event.target)) hideTooltip(); });
document.addEventListener('scroll', (event) => {
  if (tooltip.contains(event.target)) return;
  if (tooltipTrigger?.matches(':hover') || document.activeElement === tooltipTrigger) positionTooltip();
  else hideTooltip();
}, { capture: true, passive: true });
window.addEventListener('resize', hideTooltip);

function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function profile() { return state.config?.profiles.find((p) => p.id === $('model').value); }
function nativeModel(p = profile(), mode = state.mode) {
  const id = ({ design: p?.design_model, clone: p?.clone_model, speech: p?.backend_model })[mode];
  return state.config?.native_models.find((m) => m.id === id);
}
function create(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; }
function humanize(value) { return value.replaceAll('_', ' ').replace(/^\w/, (c) => c.toUpperCase()); }
function modelName(p) { return p.display_name || humanize(p.id); }
function modeDefaults(p = profile()) {
  return { ...p.options, ...({ speech: p.preset_options, design: p.design_options, clone: p.clone_options, stream: p.streaming_options }[state.mode] || {}) };
}
function matches(options, conditions = {}) { return Object.entries(conditions).every(([key, values]) => values.includes(options[key])); }
function effectiveFormOptions() {
  const options = modeDefaults();
  for (const row of $('dynamic-options').children) {
    if (!row.querySelector('[data-override]').checked) continue;
    const value = row.querySelector('[data-value]').value;
    const type = row.dataset.type;
    try {
      options[row.dataset.name] = type === 'bool' ? value === 'true' : ['int', 'float', 'number'].includes(type) ? Number(value)
        : type.includes('list') || type === 'json' || type === 'object' ? JSON.parse(value) : value;
    } catch { /* Incomplete edits are validated by the request builder. */ }
  }
  try { Object.assign(options, JSON.parse($('extra-options').value || '{}')); } catch { /* The request builder reports invalid JSON. */ }
  return options;
}
function updateOptionAvailability() {
  const p = profile(); if (!p) return;
  const options = effectiveFormOptions();
  for (const row of $('dynamic-options').children) {
    const conditions = row.dataset.name === p.instruction_option ? p.instruction_conditions : p.option_dependencies?.[row.dataset.name];
    const available = matches(options, conditions);
    row.dataset.inactive = String(!available);
    const toggle = row.querySelector('[data-override]'); toggle.disabled = !available;
    row.querySelector('[data-value]').disabled = !available || !toggle.checked;
    let hint = row.querySelector('[data-dependency-hint]');
    if (!hint) { hint = create('span', 'help'); hint.dataset.dependencyHint = ''; row.append(hint); }
    hint.textContent = !available ? `Requires ${Object.entries(conditions).map(([key, values]) => `${humanize(key)}: ${values.join(' or ')}`).join('; ')}. Change the template to enable this control.` : '';
    hint.hidden = available;
  }
  const instruction = matches(options, p.instruction_conditions);
  $('design-section').hidden = state.mode !== 'design' || !instruction;
  $('instructions').required = state.mode === 'design' && instruction;
  $('clone-direction-field').hidden = !p.instruction_option || !instruction;
  $('mode-help').textContent = !instruction && state.mode === 'design'
    ? 'This template uses its own structured controls instead of the voice description. For built-in voices, choose a Voice id below or use the Presets tab.' : modes[state.mode];
  prepareHelp();
}
function b64Blob(base64, type) { const raw = atob(base64); const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0)); return new Blob([bytes], { type }); }
function blobB64(blob) { return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = () => reject(new Error('Could not read the recording.')); reader.readAsDataURL(blob); }); }
function headers() { const key = sessionStorage.getItem('voice-studio-key'); return key ? { Authorization: `Bearer ${key}` } : {}; }
async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...headers(), ...options.headers } });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    if (response.status === 401 && !$('connection-dialog').open) $('connection-dialog').showModal();
    throw new Error(error.error?.message || error.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function rememberDraft() {
  if (!draftKey) return;
  state.drafts.set(draftKey, { options: [...$('dynamic-options').children].map((row) => ({ name: row.dataset.name, checked: row.querySelector('[data-override]').checked, value: row.querySelector('[data-value]').value })), extra: $('extra-options').value });
}
function renderOptions() {
  const p = profile();
  const params = nativeModel()?.params || [];
  const dedicated = new Set(['seed', p.language_option]);
  if (state.mode === 'design' || state.mode === 'clone') dedicated.add(p.instruction_option);
  if (state.mode === 'clone') dedicated.add(p.reference_text_option);
  if (p.native_speed) dedicated.add('speed');
  const rows = params.filter((param) => !dedicated.has(param.name));
  $('dynamic-options').replaceChildren();
  draftKey = `${p.id}:${state.mode}`;
  const draft = state.drafts.get(draftKey);
  const defaults = modeDefaults(p);
  for (const param of rows) {
    const row = create('div', 'parameter'); row.dataset.name = param.name; row.dataset.type = param.type;
    const top = create('div', 'parameter-top');
    const title = create('label', 'parameter-title', humanize(param.name)); title.htmlFor = `param-${param.name}`;
    const toggleLabel = create('label', 'override-label');
    const toggle = document.createElement('input'); toggle.type = 'checkbox'; toggle.dataset.override = ''; toggle.setAttribute('aria-label', `Override ${humanize(param.name)}`);
    toggleLabel.append(toggle, ' Override'); top.append(title, toggleLabel); row.append(top);
    const field = create('div', 'field'); let control;
    if (param.type === 'enum' || param.type === 'bool') {
      control = document.createElement('select');
      const allowed = p.mode_conditions?.[state.mode]?.[param.name];
      const values = param.type === 'bool' ? [true, false] : (param.values || []).filter((value) => !allowed || allowed.includes(value));
      for (const value of values) { const option = document.createElement('option'); option.value = String(value); option.textContent = String(value); control.append(option); }
    } else if (['int', 'float', 'number'].includes(param.type)) {
      control = document.createElement('input'); control.type = 'number'; control.step = param.type === 'int' ? '1' : 'any';
      if (param.min !== undefined) control.min = param.min;
      if (param.max !== undefined) control.max = param.max;
    } else if (param.type.includes('list') || param.type === 'json' || param.type === 'object') {
      control = document.createElement('textarea'); control.rows = 2; control.placeholder = 'JSON array or object';
    } else { control = document.createElement('input'); control.type = 'text'; }
    control.id = `param-${param.name}`; control.dataset.value = ''; control.disabled = true;
    const effective = defaults[param.name] ?? param.default;
    if (effective !== undefined) control.value = typeof effective === 'object' ? JSON.stringify(effective) : String(effective);
    if (control.tagName === 'INPUT' && !control.value) control.placeholder = 'Native default';
    const previous = draft?.options.find((item) => item.name === param.name);
    if (previous) { toggle.checked = previous.checked; control.value = previous.value; control.disabled = !toggle.checked; }
    toggle.addEventListener('change', () => { updateOptionAvailability(); updatePreview(); });
    field.append(control); row.append(field);
    const description = [param.description, effective !== undefined ? `Default: ${typeof effective === 'object' ? JSON.stringify(effective) : effective}.` : '', param.required ? 'Required by this model.' : ''].filter(Boolean).join(' ');
    if (description) row.append(create('span', 'help', description));
    $('dynamic-options').append(row);
  }
  $('option-count').textContent = String(params.length);
  $('extra-options').value = draft?.extra || '{}';
  updateOptionAvailability();
}

function selectMode(mode) {
  const p = profile(); if (!p) return;
  rememberDraft();
  if (mode === 'design' && !p.design_model) mode = 'speech';
  if (mode === 'clone' && !p.clone_model) mode = p.design_model ? 'design' : 'speech';
  state.mode = mode;
  for (const button of document.querySelectorAll('[data-mode]')) {
    const disabled = (button.dataset.mode === 'design' && !p.design_model) || (button.dataset.mode === 'clone' && !p.clone_model);
    button.disabled = disabled;
    button.title = disabled ? `This profile does not support ${button.dataset.mode}.` : '';
    button.setAttribute('aria-selected', String(button.dataset.mode === mode));
    button.tabIndex = button.dataset.mode === mode ? 0 : -1;
  }
  $('design-section').hidden = mode !== 'design';
  $('clone-direction-field').hidden = !p.instruction_option;
  $('reference-transcript-field').hidden = !p.reference_text_option;
  $('language-field').hidden = !p.language_option;
  $('description-label').textContent = mode === 'clone' ? 'Additional voice direction · optional' : 'Voice direction';
  $('instructions').required = mode === 'design';
  $('direction-chips').hidden = mode !== 'design';
  $('reference-section').hidden = mode !== 'clone';
  $('voice-field').hidden = mode !== 'speech';
  $('context-section').hidden = mode !== 'speech' || !p.context;
  $('mode-help').textContent = modes[mode];
  $('native-label').textContent = nativeModel()?.id || 'NATIVE MODEL';
  $('model-info').textContent = p.description || [p.design_model ? 'Voice design' : null, p.clone_model ? 'Voice cloning' : null, 'Preset synthesis'].filter(Boolean).join(' · ');
  $('generate-label').textContent = mode === 'clone' ? 'Generate cloned speech' : mode === 'design' ? 'Design this voice' : 'Generate speech';
  $('text').maxLength = p.max_input_chars;
  $('seed').title = `An integer from 0 to ${p.seed_max || '18446744073709551615'}`;
  $('seed').placeholder = 'Native default';
  renderOptions(); updateCount(); updatePreview();
}
function selectModel() {
  const p = profile(); if (!p) return;
  const voiceIDs = new Set([...(p.voices || []), ...Object.keys(p.voice_aliases), ...Object.values(p.voice_aliases), ...Object.keys(p.voice_descriptions)]);
  if (p.default_voice) voiceIDs.add(p.default_voice);
  const voiceParam = nativeModel(p, 'speech')?.params?.find((param) => param.name === (p.voice_option || 'voice_id'));
  for (const voice of voiceParam?.values || []) voiceIDs.add(voice);
  const defaultOption = create('option', '', p.default_voice ? `Model default (${p.default_voice})` : 'Model default');
  defaultOption.value = '';
  $('voice').replaceChildren(defaultOption, ...[...voiceIDs].filter(Boolean).sort().map((voice) => { const option = create('option', '', voice); option.value = voice; return option; }));
  $('voice').value = '';
  $('language-list').replaceChildren(...[...new Set(['Auto', 'English', ...Object.keys(p.language_map), ...Object.values(p.language_map).filter(Boolean)])].map((lang) => { const option = document.createElement('option'); option.value = lang; return option; }));
  selectMode(state.mode);
}

function readOptions() {
  const options = {};
  for (const row of $('dynamic-options').children) {
    if (row.dataset.inactive === 'true' || !row.querySelector('[data-override]').checked) continue;
    const value = row.querySelector('[data-value]').value;
    if (!value.trim()) throw new Error(`Enter a value for ${humanize(row.dataset.name)}, or turn off its override.`);
    const type = row.dataset.type;
    if (type === 'bool') options[row.dataset.name] = value === 'true';
    else if (['int', 'float', 'number'].includes(type)) {
      const number = Number(value); if (!Number.isFinite(number) || (type === 'int' && !Number.isSafeInteger(number))) throw new Error(`${humanize(row.dataset.name)} must be a valid ${type === 'int' ? 'integer' : 'number'}.`);
      options[row.dataset.name] = number;
    } else if (type.includes('list') || type === 'json' || type === 'object') { try { options[row.dataset.name] = JSON.parse(value); } catch { throw new Error(`${humanize(row.dataset.name)} must contain valid JSON.`); } }
    else options[row.dataset.name] = value;
  }
  let extra;
  try { extra = JSON.parse($('extra-options').value || '{}'); } catch { throw new Error('Additional options must contain valid JSON.'); }
  if (!extra || Array.isArray(extra) || typeof extra !== 'object') throw new Error('Additional options must be a JSON object.');
  const combined = { ...options, ...extra };
  const effective = { ...modeDefaults(), ...combined };
  for (const [name, conditions] of Object.entries(profile().option_dependencies || {})) {
    if (name in combined && !matches(effective, conditions)) throw new Error(`${humanize(name)} requires ${Object.entries(conditions).map(([key, values]) => `${humanize(key)}: ${values.join(' or ')}`).join('; ')}. Change the template or remove this override.`);
  }
  return combined;
}
function buildRequest(preview = false) {
  const p = profile(); if (!p) throw new Error('Choose an available model.');
  const speech = { model: p.id, input: $('text').value, response_format: $('format').value, language: $('language').value || 'Auto', speed: Number($('speed').value), options: readOptions() };
  if ($('seed').value.trim()) {
    const seed = $('seed').value.trim();
    const maximum = BigInt(p.seed_max ?? '18446744073709551615');
    if (!/^\d+$/.test(seed) || BigInt(seed) > maximum) throw new Error(`Seed must be an integer from 0 to ${maximum} for this model.`);
    speech.seed = seed; // Preserve all 64 bits through JSON and Python validation.
  }
  const acceptsInstruction = matches({ ...modeDefaults(), ...speech.options }, p.instruction_conditions);
  if (state.mode === 'design' && acceptsInstruction) speech.instructions = $('instructions').value;
  if (state.mode === 'clone' && acceptsInstruction && $('clone-direction')?.value) speech.instructions = $('clone-direction').value;
  if (state.mode === 'speech' && $('voice').value) speech.voice = $('voice').value;
  if (state.mode === 'speech' && p.context) { speech.context_before = $('context-before').value; speech.context_after = $('context-after').value; }
  const request = { mode: state.mode, speech };
  if (state.mode === 'clone') {
    if (!state.reference && !preview) throw new Error('Upload a reference or choose “Use as reference” on a generated take.');
    if (state.reference) request.reference_audio = preview ? `[${state.reference.name} · base64 audio]` : state.reference.audio;
    request.reference_text = $('reference-text').value;
  }
  return request;
}
function updatePreview() { try { $('request-preview').textContent = JSON.stringify(buildRequest(true), null, 2); } catch (error) { $('request-preview').textContent = error.message; } }
function updateCount() { const count = [...$('text').value].length; $('char-count').textContent = `${count} / ${profile()?.max_input_chars || '—'}`; }

async function connect() {
  try {
    state.config = await api('/ui/config');
    $('model').replaceChildren(...state.config.profiles.map((p) => { const option = document.createElement('option'); option.value = p.id; option.textContent = modelName(p); return option; }));
    $('model').value = state.config.default_model;
    $('status-dot').classList.add('online'); $('connection').textContent = 'Local engine connected';
    $('generate').disabled = false;
    notice(); selectModel();
  } catch (error) { notice(error.message); $('connection').textContent = 'Connection unavailable'; $('status-dot').classList.remove('online'); $('generate').disabled = true; }
}

function setReference(audio, name, transcript = '', type = 'audio/wav') {
  if (referenceURL) URL.revokeObjectURL(referenceURL);
  state.reference = { audio, name };
  referenceURL = URL.createObjectURL(b64Blob(audio, type));
  $('reference-player').src = referenceURL;
  $('reference-preview').hidden = false;
  $('reference-name').textContent = 'Replace reference recording';
  $('reference-source').textContent = name;
  $('reference-text').value = transcript;
  updatePreview();
}
async function uploadReference(file) {
  if (!file) return;
  if (file.size > (state.config?.max_reference_bytes || 5242880)) { notice('Reference audio must be at most 5 MiB. Use a short, clean clip.'); return; }
  try { setReference(await blobB64(file), file.name, '', file.type || 'audio/wav'); notice(); } catch (error) { notice(error.message); }
}
function useReference(take) {
  const candidates = state.config.profiles.filter((p) => p.clone_model);
  const source = state.config.profiles.find((p) => p.id === take.model);
  const family = nativeModel(source, take.mode === 'design' ? 'design' : 'speech')?.family;
  const target = candidates.find((p) => p.id === take.model) || candidates.find((p) => p.id === $('model').value)
    || (family && candidates.find((p) => nativeModel(p, 'clone')?.family === family)) || candidates[0];
  if (!target) { notice('No configured model supports voice cloning.'); return; }
  $('model').value = target.id; selectModel(); selectMode('clone');
  setReference(take.reference_audio, `Take ${String(take.index).padStart(2, '0')} · ${take.model}`, take.transcript);
  $('text').value = 'It is good to see you again. Tell me, where will our story go from here?';
  updateCount(); updatePreview(); notice();
  $('reference-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
  $('text').focus({ preventScroll: true });
}

async function drawWaveform(canvas, blob) {
  const width = canvas.width = 640; const height = canvas.height = 110;
  const context = canvas.getContext('2d');
  try {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
    const values = decoded.getChannelData(0);
    context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    for (let x = 0; x < width; x += 5) {
      const start = Math.floor(x / width * values.length); const end = Math.floor((x + 4) / width * values.length);
      let peak = 0; for (let i = start; i < end; i += Math.max(1, Math.floor((end - start) / 80))) peak = Math.max(peak, Math.abs(values[i]));
      const amplitude = Math.max(2, Math.min(height - 10, peak * height));
      context.fillRect(x, (height - amplitude) / 2, 2, amplitude);
    }
  } catch { context.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim(); context.font = '16px sans-serif'; context.fillText('Use the player to listen to this take.', 30, 65); }
}
function addTake(take, request) {
  take.index = (state.results.at(-1)?.index || 0) + 1;
  const card = create('article', 'take');
  const takeProfile = state.config.profiles.find((p) => p.id === take.model);
  const top = create('div', 'take-top'); top.append(create('strong', '', `Take ${String(take.index).padStart(2, '0')} / ${takeProfile ? modelName(takeProfile) : humanize(take.model)}`), create('span', '', take.mode.toUpperCase())); card.append(top);
  const canvas = document.createElement('canvas'); canvas.setAttribute('aria-label', 'Audio waveform'); card.append(canvas);
  const blob = b64Blob(take.audio, 'audio/wav'); const url = URL.createObjectURL(blob);
  const downloadURL = URL.createObjectURL(b64Blob(take.download, mime[take.format])); take.urls = [url, downloadURL];
  const player = document.createElement('audio'); player.controls = true; player.preload = 'metadata'; player.src = url; card.append(player);
  card.append(create('p', 'take-text', take.transcript));
  const metrics = create('div', 'take-metrics');
  metrics.append(...[`${take.duration.toFixed(2)}s audio`, `${take.elapsed.toFixed(2)}s generation`, `${(1 / take.realtime_factor).toFixed(1)}× realtime`, `${(take.native_sample_rate / 1000).toFixed(1)} → ${(take.sample_rate / 1000).toFixed(1)} kHz`, take.context_status ? `Context: ${take.context_status}` : null].filter(Boolean).map((text) => create('span', '', text))); card.append(metrics);
  const details = create('details', 'take-parameters'); details.append(create('summary', '', 'Generation settings')); const clean = structuredClone(request); if (clean.reference_audio) clean.reference_audio = '[reference recording]'; details.append(create('pre', '', JSON.stringify(clean, null, 2))); card.append(details);
  const actions = create('div', 'take-actions');
  const reuse = create('button', '', 'Use as reference ↗'); reuse.type = 'button'; reuse.disabled = !state.config.profiles.some((p) => p.clone_model); reuse.addEventListener('click', () => useReference(take));
  const download = create('a', '', `Download .${take.format}`); download.href = downloadURL; download.download = `voice-${take.index}-${take.mode}.${take.format}`;
  actions.append(reuse, download); card.append(actions); $('results').prepend(card); take.card = card;
  state.results.push(take); if (state.results.length > 8) { const old = state.results.shift(); old.card.querySelector('audio').pause(); old.card.remove(); old.urls.forEach((item) => URL.revokeObjectURL(item)); }
  $('empty-state').hidden = true; $('clip-count').textContent = `${state.results.length} ${state.results.length === 1 ? 'TAKE' : 'TAKES'} / THIS SESSION`;
  drawWaveform(canvas, blob);
}

$('generation-form').addEventListener('submit', async (event) => {
  event.preventDefault(); if (state.busy) return;
  let request;
  try { request = buildRequest(); } catch (error) { notice(error.message); return; }
  notice(); state.busy = true; $('generate').disabled = true; $('generate').classList.add('busy');
  const started = performance.now();
  $('generation-status').textContent = 'Generating…';
  const timer = setInterval(() => { $('generation-status').textContent = `Generating · ${((performance.now() - started) / 1000).toFixed(1)}s`; }, 100);
  try {
    const take = await api('/ui/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
    addTake(take, request); $('generation-status').textContent = `Take ready · ${take.elapsed.toFixed(2)}s generation`;
  } catch (error) { notice(error.message); $('generation-status').textContent = 'Generation failed'; }
  finally { clearInterval(timer); state.busy = false; $('generate').disabled = false; $('generate').classList.remove('busy'); }
});
$('model').addEventListener('change', selectModel);
for (const button of document.querySelectorAll('[data-mode]')) button.addEventListener('click', () => selectMode(button.dataset.mode));
document.querySelector('.mode-tabs').addEventListener('keydown', (event) => { if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return; const tabs = [...document.querySelectorAll('[data-mode]')].filter((tab) => !tab.disabled); const index = tabs.indexOf(document.activeElement); const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length; event.preventDefault(); tabs[next].click(); tabs[next].focus(); });
for (const button of document.querySelectorAll('[data-prompt]')) button.addEventListener('click', () => { $('instructions').value = button.dataset.prompt; updatePreview(); });
$('generation-form').addEventListener('input', () => { $('speed-value').textContent = `${Number($('speed').value).toFixed(2)}×`; updateCount(); updateOptionAvailability(); updatePreview(); });
$('generation-form').addEventListener('change', () => { updateOptionAvailability(); updatePreview(); });
$('reference-file').addEventListener('change', (event) => uploadReference(event.target.files[0]));
const zone = $('upload-zone');
zone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('reference-file').click(); } });
zone.addEventListener('dragover', (event) => { event.preventDefault(); zone.classList.add('dragging'); });
zone.addEventListener('dragleave', () => zone.classList.remove('dragging'));
zone.addEventListener('drop', (event) => { event.preventDefault(); zone.classList.remove('dragging'); uploadReference(event.dataTransfer.files[0]); });
$('clear-reference').addEventListener('click', () => { if (referenceURL) URL.revokeObjectURL(referenceURL); referenceURL = null; state.reference = null; $('reference-file').value = ''; $('reference-preview').hidden = true; $('reference-player').removeAttribute('src'); $('reference-player').load(); $('reference-name').textContent = 'Drop a reference recording'; $('reference-text').value = ''; updatePreview(); });
$('reset-options').addEventListener('click', () => { state.drafts.delete(draftKey); renderOptions(); updatePreview(); });
$('copy-request').addEventListener('click', async () => { try { await navigator.clipboard.writeText(JSON.stringify(buildRequest(), null, 2)); $('copy-request').textContent = 'Copied'; setTimeout(() => { $('copy-request').textContent = 'Copy request JSON'; }, 1500); } catch (error) { notice(error.message); } });
$('connect-button').addEventListener('click', () => { $('api-key').value = sessionStorage.getItem('voice-studio-key') || ''; $('connection-dialog').showModal(); });
$('close-connection').addEventListener('click', () => $('connection-dialog').close());
$('connection-form').addEventListener('submit', (event) => { event.preventDefault(); sessionStorage.setItem('voice-studio-key', $('api-key').value); $('connection-dialog').close(); connect(); });
document.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); $('generation-form').requestSubmit(); } });
window.addEventListener('beforeunload', () => { if (referenceURL) URL.revokeObjectURL(referenceURL); for (const take of state.results) take.urls.forEach((url) => URL.revokeObjectURL(url)); });
for (let i = 0; i < 37; i++) { const bar = document.createElement('i'); bar.style.height = `${7 + Math.sin(i / 36 * Math.PI) ** 2 * (24 + Math.abs(Math.sin(i * 1.7)) * 57)}px`; $('art-bars').append(bar); }
$('generate').disabled = true;
prepareHelp();
connect();
