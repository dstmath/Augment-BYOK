"use strict";

const nodePath = require("path");

const { debug, warn } = require("../infra/log");
const { ensureConfigManager, state } = require("../config/state");
const { decideRoute } = require("../core/router");
const { normalizeEndpoint, normalizeString, normalizeRawToken, safeTransform, emptyAsyncGenerator } = require("../infra/util");
const { ensureModelRegistryFeatureFlags } = require("../core/model-registry");
const { openAiCompleteText, openAiStreamTextDeltas, openAiChatStreamChunks } = require("../providers/openai");
const { anthropicCompleteText, anthropicStreamTextDeltas, anthropicChatStreamChunks } = require("../providers/anthropic");
const { joinBaseUrl, safeFetch, readTextLimit } = require("../providers/http");
const { getOfficialConnection } = require("../config/official");
const { normalizeAugmentChatRequest, buildSystemPrompt, convertOpenAiTools, convertAnthropicTools, buildToolMetaByName, buildOpenAiMessages, buildAnthropicMessages } = require("../core/augment-chat");
const augmentChatShared = require("../core/augment-chat.shared");
const { maybeSummarizeAndCompactAugmentChatRequest } = require("../core/augment-history-summary-auto");
const { REQUEST_NODE_TEXT, REQUEST_NODE_TOOL_RESULT, STOP_REASON_END_TURN, makeBackChatChunk } = require("../core/augment-protocol");
const { makeEndpointErrorText, guardObjectStream } = require("../core/stream-guard");
const {
  buildMessagesForEndpoint,
  makeBackTextResult,
  makeBackChatResult,
  makeBackCompletionResult,
  makeBackNextEditGenerationChunk,
  makeBackNextEditLocationResult,
  buildByokModelsFromConfig,
  makeBackGetModelsResult,
  makeModelInfo
} = require("../core/protocol");

const OFFICIAL_CODEBASE_RETRIEVAL_MAX_OUTPUT_LENGTH = 20000;
const OFFICIAL_CODEBASE_RETRIEVAL_TIMEOUT_MS = 12000;
const OFFICIAL_CONTEXT_CANVAS_TIMEOUT_MS = 4000;
const CONTEXT_CANVAS_CACHE_TTL_MS = 5 * 60 * 1000;
const CONTEXT_CANVAS_CACHE = new Map();

function resolveProviderApiKey(provider, label) {
  if (!provider || typeof provider !== "object") throw new Error(`${label} provider 无效`);
  const key = normalizeRawToken(provider.apiKey);
  if (key) return key;
  throw new Error(`${label} 未配置 api_key`);
}

function providerLabel(provider) {
  const id = normalizeString(provider?.id);
  const type = normalizeString(provider?.type);
  return `Provider(${id || type || "unknown"})`;
}

function providerRequestContext(provider) {
  if (!provider || typeof provider !== "object") throw new Error("BYOK provider 未选择");
  const type = normalizeString(provider.type);
  const baseUrl = normalizeString(provider.baseUrl);
  const apiKey = resolveProviderApiKey(provider, providerLabel(provider));
  const extraHeaders = provider.headers && typeof provider.headers === "object" ? provider.headers : {};
  const requestDefaults = provider.requestDefaults && typeof provider.requestDefaults === "object" ? provider.requestDefaults : {};
  return { type, baseUrl, apiKey, extraHeaders, requestDefaults };
}

function asOpenAiMessages(system, messages) {
  const sys = typeof system === "string" ? system : "";
  const ms = Array.isArray(messages) ? messages : [];
  return [{ role: "system", content: sys }, ...ms].filter((m) => m && typeof m.content === "string" && m.content);
}

function asAnthropicMessages(system, messages) {
  const sys = normalizeString(system);
  const ms = Array.isArray(messages) ? messages : [];
  const out = ms
    .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content)
    .map((m) => ({ role: m.role, content: m.content }));
  return { system: sys, messages: out };
}

function isTelemetryDisabled(cfg, ep) {
  const list = Array.isArray(cfg?.telemetry?.disabledEndpoints) ? cfg.telemetry.disabledEndpoints : [];
  return list.includes(ep);
}

function normalizeLineNumber(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return null;
  if (n <= 0) return 0;
  return Math.floor(n);
}

function normalizeNewlines(s) {
  return typeof s === "string" ? s.replace(/\r\n/g, "\n") : "";
}

function countNewlines(s) {
  const text = typeof s === "string" ? s : "";
  let n = 0;
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n += 1;
  return n;
}

function clampLineNumber(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return 1;
  if (v <= 1) return 1;
  return Math.floor(v);
}

function commonPrefixLen(a, b) {
  const s1 = typeof a === "string" ? a : "";
  const s2 = typeof b === "string" ? b : "";
  const n = Math.min(s1.length, s2.length);
  let i = 0;
  for (; i < n; i++) if (s1.charCodeAt(i) !== s2.charCodeAt(i)) break;
  return i;
}

function commonSuffixLen(a, b) {
  const s1 = typeof a === "string" ? a : "";
  const s2 = typeof b === "string" ? b : "";
  const n = Math.min(s1.length, s2.length);
  let i = 0;
  for (; i < n; i++) if (s1.charCodeAt(s1.length - 1 - i) !== s2.charCodeAt(s2.length - 1 - i)) break;
  return i;
}

function bestMatchIndex(haystack, needle, { prefixHint, suffixHint, maxCandidates = 200 } = {}) {
  const h = typeof haystack === "string" ? haystack : "";
  const n = typeof needle === "string" ? needle : "";
  if (!h || !n) return -1;
  const pre = typeof prefixHint === "string" ? prefixHint : "";
  const suf = typeof suffixHint === "string" ? suffixHint : "";
  let bestIdx = -1;
  let bestScore = -1;
  let i = 0;
  for (let pos = h.indexOf(n); pos !== -1; pos = h.indexOf(n, pos + 1)) {
    i += 1;
    if (i > maxCandidates) break;
    const before = pre ? h.slice(Math.max(0, pos - pre.length), pos) : "";
    const after = suf ? h.slice(pos + n.length, pos + n.length + suf.length) : "";
    const score = commonSuffixLen(before, pre) * 2 + commonPrefixLen(after, suf);
    if (score > bestScore || (score === bestScore && pos > bestIdx)) { bestScore = score; bestIdx = pos; }
  }
  return bestIdx;
}

function bestInsertionIndex(haystack, { prefixHint, suffixHint, maxCandidates = 200 } = {}) {
  const h = typeof haystack === "string" ? haystack : "";
  const pre = typeof prefixHint === "string" ? prefixHint : "";
  const suf = typeof suffixHint === "string" ? suffixHint : "";
  if (!h) return 0;
  if (!pre && !suf) return 0;

  if (pre) {
    let bestIdx = -1;
    let bestScore = -1;
    let i = 0;
    for (let pos = h.indexOf(pre); pos !== -1; pos = h.indexOf(pre, pos + 1)) {
      i += 1;
      if (i > maxCandidates) break;
      const ins = pos + pre.length;
      const after = suf ? h.slice(ins, ins + suf.length) : "";
      const score = pre.length * 2 + commonPrefixLen(after, suf);
      if (score > bestScore || (score === bestScore && ins > bestIdx)) { bestScore = score; bestIdx = ins; }
    }
    if (bestIdx >= 0) return bestIdx;
  }

  if (suf) {
    const pos = h.indexOf(suf);
    if (pos >= 0) return pos;
  }
  return 0;
}

function trimTrailingNewlines(s) {
  const t = normalizeNewlines(s);
  return t.replace(/\n+$/g, "");
}

function resolveTextField(obj, keys) {
  const b = obj && typeof obj === "object" ? obj : {};
  for (const k of Array.isArray(keys) ? keys : []) {
    if (typeof b[k] === "string") return b[k];
  }
  return "";
}

async function readWorkspaceFileTextByPath(p) {
  const raw = normalizeString(p);
  if (!raw) return "";
  const vscode = state.vscode;
  const ws = vscode && vscode.workspace ? vscode.workspace : null;
  const Uri = vscode && vscode.Uri ? vscode.Uri : null;
  if (!ws || !ws.fs || typeof ws.fs.readFile !== "function" || !Uri) return "";

  const tryRead = async (uri) => {
    try {
      const bytes = await ws.fs.readFile(uri);
      return Buffer.from(bytes).toString("utf8");
    } catch {
      return "";
    }
  };

  if (raw.includes("://")) {
    try { return await tryRead(Uri.parse(raw)); } catch {}
  }

  try {
    if (nodePath.isAbsolute(raw)) return await tryRead(Uri.file(raw));
  } catch {}

  const rel = raw.replace(/\\/g, "/").replace(/^\/+/, "");
  const folders = Array.isArray(ws.workspaceFolders) ? ws.workspaceFolders : [];
  for (const f of folders) {
    const base = f && f.uri ? f.uri : null;
    if (!base) continue;
    const u = Uri.joinPath(base, rel);
    const txt = await tryRead(u);
    if (txt) return txt;
  }
  return "";
}

async function buildInstructionReplacementMeta(body) {
  const b = body && typeof body === "object" ? body : {};
  const selectedTextRaw = resolveTextField(b, ["selected_text", "selectedText"]);
  const prefixRaw = resolveTextField(b, ["prefix"]);
  const suffixRaw = resolveTextField(b, ["suffix"]);
  const targetPath = normalizeString(resolveTextField(b, ["target_file_path", "targetFilePath"]));
  const path = normalizeString(resolveTextField(b, ["path", "pathName"]));
  const filePath = targetPath || path;

  const targetFileContentRaw = resolveTextField(b, ["target_file_content", "targetFileContent"]);
  const fileTextRaw = targetFileContentRaw ? targetFileContentRaw : await readWorkspaceFileTextByPath(filePath);
  const fileText = normalizeNewlines(fileTextRaw);
  const selectedText = normalizeNewlines(selectedTextRaw);
  const prefix = normalizeNewlines(prefixRaw);
  const suffix = normalizeNewlines(suffixRaw);

  const prefixHint = prefix ? prefix.slice(Math.max(0, prefix.length - 400)) : "";
  const suffixHint = suffix ? suffix.slice(0, 400) : "";

  if (fileText && selectedText) {
    const idx = bestMatchIndex(fileText, selectedText, { prefixHint, suffixHint });
    if (idx >= 0) {
      const startLine = 1 + countNewlines(fileText.slice(0, idx));
      const trimmed = trimTrailingNewlines(selectedText);
      const endLine = startLine + countNewlines(trimmed);
      return { replacement_start_line: clampLineNumber(startLine), replacement_end_line: clampLineNumber(endLine), replacement_old_text: selectedText };
    }
  }

  const insertIdx = fileText ? bestInsertionIndex(fileText, { prefixHint, suffixHint }) : 0;
  const insertLine = fileText ? 1 + countNewlines(fileText.slice(0, insertIdx)) : 1;
  const lines = fileText ? fileText.split("\n") : [];
  const lineBefore = insertLine > 1 && lines[insertLine - 2] != null ? String(lines[insertLine - 2]).trimEnd() : "";
  const oldText = selectedText ? selectedText : `PURE INSERTION AFTER LINE:${lineBefore}`;
  return { replacement_start_line: clampLineNumber(insertLine), replacement_end_line: clampLineNumber(insertLine), replacement_old_text: oldText };
}

function pickNextEditLocationCandidates(body) {
  const b = body && typeof body === "object" ? body : {};
  const max =
    Number.isFinite(Number(b.num_results)) && Number(b.num_results) > 0 ? Math.min(6, Math.floor(Number(b.num_results))) : 1;

  const out = [];
  const diags = Array.isArray(b.diagnostics) ? b.diagnostics : [];
  for (const d of diags) {
    const path = normalizeString(d?.path || d?.file_path || d?.filePath || d?.item?.path);
    if (!path) continue;
    const r = d?.range || d?.item?.range || d?.location?.range;
    const start = normalizeLineNumber(r?.start?.line ?? r?.start_line ?? r?.startLine ?? r?.start);
    if (start === null) continue;
    const stop = normalizeLineNumber(r?.end?.line ?? r?.stop?.line ?? r?.end_line ?? r?.stopLine ?? r?.stop ?? start) ?? start;
    out.push({ item: { path, range: { start, stop: Math.max(start, stop) } }, score: 1, debug_info: { source: "diagnostic" } });
    if (out.length >= max) break;
  }

  if (!out.length) {
    const path = normalizeString(b.path);
    if (path) out.push({ item: { path, range: { start: 0, stop: 0 } }, score: 1, debug_info: { source: "fallback" } });
  }

  return out;
}

async function byokCompleteText({ provider, model, system, messages, timeoutMs, abortSignal }) {
  const { type, baseUrl, apiKey, extraHeaders, requestDefaults } = providerRequestContext(provider);

  if (type === "openai_compatible") {
    return await openAiCompleteText({
      baseUrl,
      apiKey,
      model,
      messages: asOpenAiMessages(system, messages),
      timeoutMs,
      abortSignal,
      extraHeaders,
      requestDefaults
    });
  }
  if (type === "anthropic") {
    const { system: sys, messages: msgs } = asAnthropicMessages(system, messages);
    return await anthropicCompleteText({ baseUrl, apiKey, model, system: sys, messages: msgs, timeoutMs, abortSignal, extraHeaders, requestDefaults });
  }
  throw new Error(`未知 provider.type: ${type}`);
}

async function* byokStreamText({ provider, model, system, messages, timeoutMs, abortSignal }) {
  const { type, baseUrl, apiKey, extraHeaders, requestDefaults } = providerRequestContext(provider);

  if (type === "openai_compatible") {
    yield* openAiStreamTextDeltas({
      baseUrl,
      apiKey,
      model,
      messages: asOpenAiMessages(system, messages),
      timeoutMs,
      abortSignal,
      extraHeaders,
      requestDefaults
    });
    return;
  }
  if (type === "anthropic") {
    const { system: sys, messages: msgs } = asAnthropicMessages(system, messages);
    yield* anthropicStreamTextDeltas({ baseUrl, apiKey, model, system: sys, messages: msgs, timeoutMs, abortSignal, extraHeaders, requestDefaults });
    return;
  }
  throw new Error(`未知 provider.type: ${type}`);
}

async function* byokChatStream({ cfg, provider, model, requestedModel, body, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken }) {
  const { type, baseUrl, apiKey, extraHeaders, requestDefaults } = providerRequestContext(provider);
  const req = normalizeAugmentChatRequest(body);
  const msg = normalizeString(req.message);
  const hasNodes = Array.isArray(req.nodes) && req.nodes.length;
  const hasHistory = Array.isArray(req.chat_history) && req.chat_history.length;
  const hasReqNodes = (Array.isArray(req.structured_request_nodes) && req.structured_request_nodes.length) || (Array.isArray(req.request_nodes) && req.request_nodes.length);
  if (!msg && !hasNodes && !hasHistory && !hasReqNodes) {
    yield makeBackChatChunk({ text: "", stop_reason: STOP_REASON_END_TURN });
    return;
  }
  try {
    await maybeSummarizeAndCompactAugmentChatRequest({ cfg, req, requestedModel, fallbackProvider: provider, fallbackModel: model, timeoutMs, abortSignal });
  } catch (err) {
    warn(`historySummary failed (ignored): ${err instanceof Error ? err.message : String(err)}`);
  }
  await maybeInjectOfficialCodebaseRetrieval({ req, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken });
  await maybeInjectOfficialContextCanvas({ req, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken });
  await maybeInjectOfficialExternalSources({ req, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken });
  const toolMetaByName = buildToolMetaByName(req.tool_definitions);
  const fdf = req && typeof req === "object" && req.feature_detection_flags && typeof req.feature_detection_flags === "object" ? req.feature_detection_flags : {};
  const supportToolUseStart = fdf.support_tool_use_start === true || fdf.supportToolUseStart === true;
  if (type === "openai_compatible") {
    yield* openAiChatStreamChunks({ baseUrl, apiKey, model, messages: buildOpenAiMessages(req), tools: convertOpenAiTools(req.tool_definitions), timeoutMs, abortSignal, extraHeaders, requestDefaults, toolMetaByName, supportToolUseStart });
    return;
  }
  if (type === "anthropic") {
    yield* anthropicChatStreamChunks({ baseUrl, apiKey, model, system: buildSystemPrompt(req), messages: buildAnthropicMessages(req), tools: convertAnthropicTools(req.tool_definitions), timeoutMs, abortSignal, extraHeaders, requestDefaults, toolMetaByName, supportToolUseStart });
    return;
  }
  throw new Error(`未知 provider.type: ${type}`);
}

async function fetchOfficialGetModels({ completionURL, apiToken, timeoutMs, abortSignal }) {
  const url = joinBaseUrl(normalizeString(completionURL), "get-models");
  if (!url) throw new Error("completionURL 无效（无法请求官方 get-models）");
  const headers = { "content-type": "application/json" };
  if (apiToken) headers.authorization = `Bearer ${apiToken}`;
  const resp = await safeFetch(url, { method: "POST", headers, body: "{}" }, { timeoutMs, abortSignal, label: "augment/get-models" });
  if (!resp.ok) throw new Error(`get-models ${resp.status}: ${await readTextLimit(resp, 300)}`.trim());
  const json = await resp.json().catch(() => null);
  if (!json || typeof json !== "object") throw new Error("get-models 响应不是 JSON 对象");
  return json;
}

function normalizeExternalSourceIdsFromImplicitResult(raw) {
  const out = [];
  if (Array.isArray(raw)) out.push(...raw);
  else if (raw && typeof raw === "object") {
    const r = raw;
    const candidates =
      (Array.isArray(r.external_source_ids) && r.external_source_ids) ||
      (Array.isArray(r.externalSourceIds) && r.externalSourceIds) ||
      (Array.isArray(r.source_ids) && r.source_ids) ||
      (Array.isArray(r.sourceIds) && r.sourceIds) ||
      (Array.isArray(r.implicit_external_source_ids) && r.implicit_external_source_ids) ||
      (Array.isArray(r.implicitExternalSourceIds) && r.implicitExternalSourceIds) ||
      (Array.isArray(r.external_sources) && r.external_sources) ||
      (Array.isArray(r.externalSources) && r.externalSources) ||
      (Array.isArray(r.sources) && r.sources) ||
      (Array.isArray(r.implicit_external_sources) && r.implicit_external_sources) ||
      (Array.isArray(r.implicitExternalSources) && r.implicitExternalSources) ||
      [];
    out.push(...candidates);
  }
  const ids = [];
  for (const it of out) {
    if (typeof it === "string") ids.push(it);
    else if (it && typeof it === "object") {
      const obj = it;
      const cand = obj.id ?? obj.source_id ?? obj.sourceId ?? obj.external_source_id ?? obj.externalSourceId ?? obj.externalSourceID ?? "";
      if (typeof cand === "string") ids.push(cand);
    }
  }
  return normalizeStringList(ids, { maxItems: 200 });
}

async function fetchOfficialImplicitExternalSources({ completionURL, apiToken, message, timeoutMs, abortSignal }) {
  const url = joinBaseUrl(normalizeString(completionURL), "get-implicit-external-sources");
  if (!url) throw new Error("completionURL 无效（无法请求官方 get-implicit-external-sources）");
  const headers = { "content-type": "application/json" };
  if (apiToken) headers.authorization = `Bearer ${apiToken}`;
  const payload = { message: String(message || "") };
  const resp = await safeFetch(
    url,
    { method: "POST", headers, body: JSON.stringify(payload) },
    { timeoutMs, abortSignal, label: "augment/get-implicit-external-sources" }
  );
  if (!resp.ok) throw new Error(`get-implicit-external-sources ${resp.status}: ${await readTextLimit(resp, 300)}`.trim());
  return await resp.json().catch(() => null);
}

async function fetchOfficialSearchExternalSources({ completionURL, apiToken, query, sourceTypes, timeoutMs, abortSignal }) {
  const url = joinBaseUrl(normalizeString(completionURL), "search-external-sources");
  if (!url) throw new Error("completionURL 无效（无法请求官方 search-external-sources）");
  const headers = { "content-type": "application/json" };
  if (apiToken) headers.authorization = `Bearer ${apiToken}`;
  const payload = { query: String(query || ""), source_types: Array.isArray(sourceTypes) ? sourceTypes : [] };
  const resp = await safeFetch(
    url,
    { method: "POST", headers, body: JSON.stringify(payload) },
    { timeoutMs, abortSignal, label: "augment/search-external-sources" }
  );
  if (!resp.ok) throw new Error(`search-external-sources ${resp.status}: ${await readTextLimit(resp, 300)}`.trim());
  return await resp.json().catch(() => null);
}

async function fetchOfficialContextCanvasList({ completionURL, apiToken, pageSize, pageToken, timeoutMs, abortSignal }) {
  const url = joinBaseUrl(normalizeString(completionURL), "context-canvas/list");
  if (!url) throw new Error("completionURL 无效（无法请求官方 context-canvas/list）");
  const headers = { "content-type": "application/json" };
  if (apiToken) headers.authorization = `Bearer ${apiToken}`;
  const page_size = Number.isFinite(Number(pageSize)) && Number(pageSize) > 0 ? Math.floor(Number(pageSize)) : 100;
  const payload = { page_size, page_token: String(pageToken || "") };
  const resp = await safeFetch(
    url,
    { method: "POST", headers, body: JSON.stringify(payload) },
    { timeoutMs, abortSignal, label: "augment/context-canvas/list" }
  );
  if (!resp.ok) throw new Error(`context-canvas/list ${resp.status}: ${await readTextLimit(resp, 300)}`.trim());
  return await resp.json().catch(() => null);
}

function normalizeOfficialContextCanvasListResponse(raw) {
  const r = raw && typeof raw === "object" ? raw : null;
  const list = [];
  if (Array.isArray(raw)) list.push(...raw);
  else if (r) {
    const canvases = Array.isArray(r.canvases) ? r.canvases : [];
    list.push(...canvases);
  }

  const out = [];
  for (const it of list) {
    if (!it || typeof it !== "object") continue;
    const c = it;
    const id = normalizeString(c.canvas_id ?? c.canvasId ?? c.canvasID ?? c.id ?? "");
    const name = normalizeString(c.name ?? c.title ?? "");
    const description = normalizeString(c.description ?? c.summary ?? "");
    if (!id && !name && !description) continue;
    out.push({ id, name, description });
  }

  const nextPageToken =
    r && typeof r === "object"
      ? normalizeString(r.next_page_token ?? r.nextPageToken ?? r.next_pageToken ?? r.page_token ?? r.pageToken ?? "")
      : "";
  return { canvases: out, nextPageToken };
}

function truncateText(s, maxChars) {
  const text = typeof s === "string" ? s : String(s ?? "");
  const max = Number.isFinite(Number(maxChars)) && Number(maxChars) > 0 ? Math.floor(Number(maxChars)) : 2000;
  if (!text.trim()) return "";
  return text.length > max ? text.slice(0, max).trimEnd() + "…" : text.trim();
}

function formatContextCanvasForPrompt(canvas, { canvasId } = {}) {
  const c = canvas && typeof canvas === "object" ? canvas : null;
  if (!c) return "";
  const id = normalizeString(canvasId ?? c.id);
  const name = truncateText(normalizeString(c.name), 200);
  const description = truncateText(normalizeString(c.description), 4000);
  const lines = ["[CONTEXT_CANVAS]"];
  if (id) lines.push(`canvas_id=${id}`);
  if (name) lines.push(`name=${name}`);
  if (description) lines.push(`description=${description}`);
  if (lines.length === 1) return "";
  lines.push("[/CONTEXT_CANVAS]");
  return lines.join("\n").trim();
}

function normalizeOfficialExternalSourcesSearchResults(raw) {
  const src = raw && typeof raw === "object" ? raw : null;
  const list = [];
  if (Array.isArray(raw)) list.push(...raw);
  else if (src) {
    const candidates =
      (Array.isArray(src.sources) && src.sources) ||
      (Array.isArray(src.external_sources) && src.external_sources) ||
      (Array.isArray(src.externalSources) && src.externalSources) ||
      (Array.isArray(src.items) && src.items) ||
      (Array.isArray(src.results) && src.results) ||
      [];
    list.push(...candidates);
  }

  const out = [];
  for (const it of list) {
    if (typeof it === "string") {
      const snippet = truncateText(it, 2000);
      if (snippet) out.push({ id: "", title: "", url: "", sourceType: "", snippet });
      continue;
    }
    if (!it || typeof it !== "object") continue;
    const r = it;
    const id = normalizeString(r.id ?? r.source_id ?? r.sourceId ?? r.external_source_id ?? r.externalSourceId ?? r.externalSourceID ?? "");
    const title = normalizeString(r.title ?? r.name ?? r.display_name ?? r.displayName ?? r.source_title ?? r.sourceTitle ?? "");
    const url = normalizeString(r.url ?? r.href ?? r.link ?? r.source_url ?? r.sourceUrl ?? "");
    const sourceType = normalizeString(r.source_type ?? r.sourceType ?? r.type ?? r.kind ?? "");
    const snippet = truncateText(r.snippet ?? r.summary ?? r.excerpt ?? r.text ?? r.content ?? r.body ?? "", 4000);
    if (!id && !title && !url && !snippet) continue;
    out.push({ id, title, url, sourceType, snippet });
  }
  return out;
}

function formatExternalSourcesForPrompt(results, { selectedExternalSourceIds } = {}) {
  const items = Array.isArray(results) ? results : [];
  const selected = Array.isArray(selectedExternalSourceIds) ? selectedExternalSourceIds : [];
  const lines = ["[EXTERNAL_SOURCES]"];
  if (selected.length) lines.push(`selected_external_source_ids=${selected.join(",")}`);
  for (const r of items) {
    if (!r || typeof r !== "object") continue;
    const title = normalizeString(r.title);
    const url = normalizeString(r.url);
    const id = normalizeString(r.id);
    const sourceType = normalizeString(r.sourceType);
    const snippet = truncateText(r.snippet, 4000);
    const headerParts = [];
    if (title) headerParts.push(title);
    if (sourceType) headerParts.push(`type=${sourceType}`);
    if (url) headerParts.push(url);
    else if (id) headerParts.push(`id=${id}`);
    if (!headerParts.length && !snippet) continue;
    lines.push(`- ${headerParts.join(" | ") || "(source)"}`);
    if (snippet) lines.push(snippet);
  }
  if (lines.length === 1) return "";
  lines.push("[/EXTERNAL_SOURCES]");
  return lines.join("\n").trim();
}

function normalizeOfficialBlobs(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const b = raw;
  const checkpointIdRaw = Object.prototype.hasOwnProperty.call(b, "checkpoint_id") ? b.checkpoint_id : b.checkpointId ?? b.checkpointID ?? null;
  const addedBlobsRaw = Object.prototype.hasOwnProperty.call(b, "added_blobs") ? b.added_blobs : b.addedBlobs;
  const deletedBlobsRaw = Object.prototype.hasOwnProperty.call(b, "deleted_blobs") ? b.deleted_blobs : b.deletedBlobs;
  const checkpoint_id = normalizeString(checkpointIdRaw) || null;
  const added_blobs = Array.isArray(addedBlobsRaw) ? addedBlobsRaw : [];
  const deleted_blobs = Array.isArray(deletedBlobsRaw) ? deletedBlobsRaw : [];
  return { checkpoint_id, added_blobs, deleted_blobs };
}

function normalizeStringList(raw, { maxItems } = {}) {
  const lim = Number.isFinite(Number(maxItems)) && Number(maxItems) > 0 ? Math.floor(Number(maxItems)) : 100;
  const out = [];
  const seen = new Set();
  const list = Array.isArray(raw) ? raw : [];
  for (const v of list) {
    const s = normalizeString(String(v ?? ""));
    if (!s || seen.has(s)) continue;
    seen.add(s);
    out.push(s);
    if (out.length >= lim) break;
  }
  return out;
}

async function fetchOfficialCodebaseRetrieval({ completionURL, apiToken, informationRequest, blobs, maxOutputLength, timeoutMs, abortSignal }) {
  const url = joinBaseUrl(normalizeString(completionURL), "agents/codebase-retrieval");
  if (!url) throw new Error("completionURL 无效（无法请求官方 agents/codebase-retrieval）");
  const headers = { "content-type": "application/json" };
  if (apiToken) headers.authorization = `Bearer ${apiToken}`;
  const max_output_length = Number.isFinite(Number(maxOutputLength)) && Number(maxOutputLength) > 0 ? Math.floor(Number(maxOutputLength)) : 20000;
  const basePayload = {
    information_request: String(informationRequest || ""),
    blobs: normalizeOfficialBlobs(blobs) || { checkpoint_id: null, added_blobs: [], deleted_blobs: [] },
    dialog: [],
    max_output_length
  };
  const payload = { ...basePayload, disable_codebase_retrieval: false, enable_commit_retrieval: false };

  const postOnce = async (p) => {
    const resp = await safeFetch(
      url,
      { method: "POST", headers, body: JSON.stringify(p) },
      { timeoutMs, abortSignal, label: "augment/agents/codebase-retrieval" }
    );
    if (resp.ok) {
      const json = await resp.json().catch(() => null);
      return { ok: true, json };
    }
    const text = String(await readTextLimit(resp, 300) || "").trim();
    return { ok: false, status: resp.status, text };
  };

  let result = await postOnce(payload);
  if (!result.ok && (result.status === 400 || result.status === 422)) {
    const retry = await postOnce(basePayload);
    if (retry.ok) result = retry;
  }
  if (!result.ok) throw new Error(`agents/codebase-retrieval ${result.status}: ${result.text}`.trim());

  const json = result.json;
  if (!json || typeof json !== "object") throw new Error("agents/codebase-retrieval 响应不是 JSON 对象");
  const formatted = normalizeString(json.formatted_retrieval ?? json.formattedRetrieval);
  return formatted;
}

function buildCodebaseRetrievalInformationRequest(req) {
  const parts = [];
  const main = normalizeString(req?.message);
  if (main) parts.push(main.trim());
  for (const p of augmentChatShared.buildUserExtraTextParts(req, { hasNodes: false })) {
    const s = normalizeString(p);
    if (s) parts.push(s.trim());
  }
  if (normalizeString(req?.path)) parts.push(`path: ${String(req.path).trim()}`);
  if (normalizeString(req?.lang)) parts.push(`lang: ${String(req.lang).trim()}`);
  return parts.join("\n\n").trim();
}

function makeTextRequestNode({ id, text }) {
  return { id: Number(id) || 0, type: REQUEST_NODE_TEXT, content: "", text_node: { content: String(text || "") } };
}

function countNonToolRequestNodes(req) {
  const nodes = [...(Array.isArray(req?.nodes) ? req.nodes : []), ...(Array.isArray(req?.structured_request_nodes) ? req.structured_request_nodes : []), ...(Array.isArray(req?.request_nodes) ? req.request_nodes : [])];
  let n = 0;
  for (const node of nodes) if (augmentChatShared.normalizeNodeType(node) !== REQUEST_NODE_TOOL_RESULT) n += 1;
  return n;
}

function maybeInjectUserExtraTextParts({ req, target, startId }) {
  if (!req || typeof req !== "object") return false;
  if (!Array.isArray(target)) return false;
  if (countNonToolRequestNodes(req) > 0) return false;
  let id = Number.isFinite(Number(startId)) ? Number(startId) : -30;
  for (const p of augmentChatShared.buildUserExtraTextParts(req, { hasNodes: false })) {
    const s = normalizeString(p);
    if (!s) continue;
    target.push(makeTextRequestNode({ id, text: s.trim() }));
    id -= 1;
  }
  return true;
}

function pickInjectionTargetArray(req) {
  if (Array.isArray(req?.request_nodes) && req.request_nodes.length) return req.request_nodes;
  if (Array.isArray(req?.structured_request_nodes) && req.structured_request_nodes.length) return req.structured_request_nodes;
  if (Array.isArray(req?.nodes) && req.nodes.length) return req.nodes;
  if (Array.isArray(req?.nodes)) return req.nodes;
  return null;
}

function cacheKeyForCanvas(completionURL) {
  const key = normalizeString(completionURL);
  return key ? key : "";
}

function getCanvasCacheEntry(completionURL) {
  const key = cacheKeyForCanvas(completionURL);
  if (!key) return null;
  const e = CONTEXT_CANVAS_CACHE.get(key);
  if (!e) return null;
  if (Number(e.expiresAtMs || 0) <= Date.now()) {
    CONTEXT_CANVAS_CACHE.delete(key);
    return null;
  }
  return e;
}

function ensureCanvasCacheEntry(completionURL) {
  const key = cacheKeyForCanvas(completionURL);
  if (!key) return null;
  const existing = getCanvasCacheEntry(key);
  if (existing) return existing;
  const created = { expiresAtMs: Date.now() + CONTEXT_CANVAS_CACHE_TTL_MS, byId: new Map() };
  CONTEXT_CANVAS_CACHE.set(key, created);
  return created;
}

function upsertCanvasCache(completionURL, canvases) {
  const entry = ensureCanvasCacheEntry(completionURL);
  if (!entry) return;
  for (const c of Array.isArray(canvases) ? canvases : []) {
    if (!c || typeof c !== "object") continue;
    const id = normalizeString(c.id);
    if (!id) continue;
    entry.byId.set(id, c);
  }
  entry.expiresAtMs = Date.now() + CONTEXT_CANVAS_CACHE_TTL_MS;
}

function getCanvasFromCache(completionURL, canvasId) {
  const entry = getCanvasCacheEntry(completionURL);
  if (!entry || !entry.byId) return null;
  const id = normalizeString(canvasId);
  if (!id) return null;
  return entry.byId.get(id) || null;
}

async function maybeInjectOfficialCodebaseRetrieval({ req, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken }) {
  if (!req || typeof req !== "object") return false;
  if (req.disable_retrieval === true) return false;

  const info = buildCodebaseRetrievalInformationRequest(req);
  if (!normalizeString(info)) return false;

  const off = getOfficialConnection();
  const completionURL = normalizeString(upstreamCompletionURL) || off.completionURL;
  const apiToken = normalizeRawToken(upstreamApiToken) || off.apiToken;
  if (!completionURL || !apiToken) {
    debug("officialRetrieval skipped: missing completionURL/apiToken");
    return false;
  }

  const hardTimeout = Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : 120000;
  const t = Math.max(2000, Math.min(OFFICIAL_CODEBASE_RETRIEVAL_TIMEOUT_MS, Math.floor(hardTimeout * 0.5)));

  const baseBlobs = normalizeOfficialBlobs(req.blobs) || { checkpoint_id: null, added_blobs: [], deleted_blobs: [] };
  const userGuidedBlobs = Array.isArray(req.user_guided_blobs) ? req.user_guided_blobs : [];
  const userGuidedBlobNames = userGuidedBlobs.map((b) => normalizeString(String(b ?? ""))).filter(Boolean);

  const hasCheckpoint = Boolean(normalizeString(baseBlobs.checkpoint_id));
  const hasAdded = Array.isArray(baseBlobs.added_blobs) && baseBlobs.added_blobs.length > 0;
  const hasDeleted = Array.isArray(baseBlobs.deleted_blobs) && baseBlobs.deleted_blobs.length > 0;
  const hasUserGuided = userGuidedBlobNames.length > 0;
  if (!hasCheckpoint && !hasAdded && !hasDeleted && !hasUserGuided) return false;

  try {
    const added_blobs = [...new Set([...(Array.isArray(baseBlobs.added_blobs) ? baseBlobs.added_blobs : []), ...userGuidedBlobNames])].slice(0, 500);
    const formatted = await fetchOfficialCodebaseRetrieval({
      completionURL,
      apiToken,
      informationRequest: info,
      blobs: { ...baseBlobs, added_blobs },
      maxOutputLength: OFFICIAL_CODEBASE_RETRIEVAL_MAX_OUTPUT_LENGTH,
      timeoutMs: t,
      abortSignal
    });
    if (!normalizeString(formatted)) return false;

    const retrievalNode = makeTextRequestNode({ id: -20, text: formatted.trim() });
    const target = pickInjectionTargetArray(req);
    if (!target) return false;
    maybeInjectUserExtraTextParts({ req, target, startId: -30 });
    target.push(retrievalNode);
    debug(`officialRetrieval injected: chars=${formatted.length} target_len=${target.length}`);
    return true;
  } catch (err) {
    warn(`officialRetrieval failed (ignored): ${err instanceof Error ? err.message : String(err)}`);
    return false;
  }
}

async function maybeInjectOfficialContextCanvas({ req, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken }) {
  if (!req || typeof req !== "object") return false;
  if (req.disable_retrieval === true) return false;

  const canvasId = normalizeString(req.canvas_id);
  if (!canvasId) return false;

  const off = getOfficialConnection();
  const completionURL = normalizeString(upstreamCompletionURL) || off.completionURL;
  const apiToken = normalizeRawToken(upstreamApiToken) || off.apiToken;
  if (!completionURL || !apiToken) {
    debug("officialContextCanvas skipped: missing completionURL/apiToken");
    return false;
  }

  try {
    let canvas = getCanvasFromCache(completionURL, canvasId);
    if (!canvas) {
      const hardTimeout = Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : 120000;
      const t = Math.max(800, Math.min(OFFICIAL_CONTEXT_CANVAS_TIMEOUT_MS, Math.floor(hardTimeout * 0.15)));
      const deadline = Date.now() + t;
      let pageToken = "";
      let pages = 0;
      while (pages < 3 && Date.now() < deadline - 200) {
        const remaining = Math.max(300, deadline - Date.now());
        const raw = await fetchOfficialContextCanvasList({ completionURL, apiToken, pageSize: 100, pageToken, timeoutMs: remaining, abortSignal });
        const { canvases, nextPageToken } = normalizeOfficialContextCanvasListResponse(raw);
        if (canvases.length) upsertCanvasCache(completionURL, canvases);
        canvas = canvases.find((c) => c && typeof c === "object" && normalizeString(c.id) === canvasId) || getCanvasFromCache(completionURL, canvasId);
        if (canvas) break;
        const next = normalizeString(nextPageToken);
        if (!next) break;
        pageToken = next;
        pages += 1;
      }
    }
    if (!canvas) return false;

    const text = formatContextCanvasForPrompt(canvas, { canvasId });
    if (!normalizeString(text)) return false;

    const target = pickInjectionTargetArray(req);
    if (!target) return false;
    maybeInjectUserExtraTextParts({ req, target, startId: -30 });

    const node = makeTextRequestNode({ id: -22, text });
    const idx = target.findIndex((n) => Number(n?.id) === -20);
    if (idx >= 0) target.splice(idx, 0, node);
    else target.push(node);
    debug(`officialContextCanvas injected: chars=${text.length} target_len=${target.length}`);
    return true;
  } catch (err) {
    warn(`officialContextCanvas failed (ignored): ${err instanceof Error ? err.message : String(err)}`);
    return false;
  }
}

async function maybeInjectOfficialExternalSources({ req, timeoutMs, abortSignal, upstreamCompletionURL, upstreamApiToken }) {
  if (!req || typeof req !== "object") return false;
  if (req.disable_retrieval === true) return false;

  const msg = normalizeString(req?.message);
  if (!msg) return false;

  const explicitExternalSourceIds = normalizeStringList(req.external_source_ids, { maxItems: 200 });
  const shouldAuto = req.disable_auto_external_sources !== true;
  if (!explicitExternalSourceIds.length && !shouldAuto) return false;

  const off = getOfficialConnection();
  const completionURL = normalizeString(upstreamCompletionURL) || off.completionURL;
  const apiToken = normalizeRawToken(upstreamApiToken) || off.apiToken;
  if (!completionURL || !apiToken) {
    debug("officialExternalSources skipped: missing completionURL/apiToken");
    return false;
  }

  const hardTimeout = Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : 120000;
  const t = Math.max(1500, Math.min(8000, Math.floor(hardTimeout * 0.25)));
  const implicitTimeout = Math.max(1000, Math.min(3500, Math.floor(t * 0.4)));

  let wantedIds = explicitExternalSourceIds;
  if (!wantedIds.length && shouldAuto) {
    try {
      const implicit = await fetchOfficialImplicitExternalSources({ completionURL, apiToken, message: msg, timeoutMs: implicitTimeout, abortSignal });
      const implicitIds = normalizeExternalSourceIdsFromImplicitResult(implicit);
      if (implicitIds.length) wantedIds = implicitIds;
    } catch (err) {
      debug(`officialExternalSources implicit failed (ignored): ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  if (!wantedIds.length && shouldAuto) return false;

  try {
    const searchTimeout = explicitExternalSourceIds.length ? t : Math.max(1500, t - implicitTimeout);
    const raw = await fetchOfficialSearchExternalSources({ completionURL, apiToken, query: msg, sourceTypes: [], timeoutMs: searchTimeout, abortSignal });
    const results = normalizeOfficialExternalSourcesSearchResults(raw);
    if (!results.length) return false;

    const wantedSet = wantedIds.length ? new Set(wantedIds) : null;
    const filtered = wantedSet ? results.filter((r) => r && typeof r === "object" && normalizeString(r.id) && wantedSet.has(String(r.id))) : [];
    const chosen = (filtered.length ? filtered : results).slice(0, 6);
    const text = formatExternalSourcesForPrompt(chosen, { selectedExternalSourceIds: wantedIds });
    if (!normalizeString(text)) return false;

    const target = pickInjectionTargetArray(req);
    if (!target) return false;
    maybeInjectUserExtraTextParts({ req, target, startId: -30 });

    const node = makeTextRequestNode({ id: -21, text });
    const idx = target.findIndex((n) => Number(n?.id) === -20);
    if (idx >= 0) target.splice(idx, 0, node);
    else target.push(node);
    debug(`officialExternalSources injected: chars=${text.length} target_len=${target.length}`);
    return true;
  } catch (err) {
    warn(`officialExternalSources failed (ignored): ${err instanceof Error ? err.message : String(err)}`);
    return false;
  }
}

function mergeModels(upstreamJson, byokModelNames, opts) {
  const base = upstreamJson && typeof upstreamJson === "object" ? upstreamJson : {};
  const models = Array.isArray(base.models) ? base.models.slice() : [];
  const existing = new Set(models.map((m) => (m && typeof m.name === "string" ? m.name : "")).filter(Boolean));
  for (const name of byokModelNames) {
    if (!name || existing.has(name)) continue;
    models.push(makeModelInfo(name));
    existing.add(name);
  }
  const baseDefaultModel = typeof base.default_model === "string" && base.default_model ? base.default_model : (models[0]?.name || "unknown");
  const baseFlags = base.feature_flags && typeof base.feature_flags === "object" && !Array.isArray(base.feature_flags) ? base.feature_flags : {};
  const preferredDefaultModel = normalizeString(opts?.defaultModel);
  const defaultModel = preferredDefaultModel || baseDefaultModel;
  const flags = ensureModelRegistryFeatureFlags(baseFlags, { byokModelIds: byokModelNames, defaultModel, agentChatModel: defaultModel });
  return { ...base, default_model: defaultModel, models, feature_flags: flags };
}

async function maybeHandleCallApi({ endpoint, body, transform, timeoutMs, abortSignal, upstreamApiToken, upstreamCompletionURL }) {
  const ep = normalizeEndpoint(endpoint);
  if (!ep) return undefined;

  const cfgMgr = ensureConfigManager();
  const cfg = cfgMgr.get();
  if (!state.runtimeEnabled) return undefined;

  if (isTelemetryDisabled(cfg, ep)) {
    try {
      return safeTransform(transform, {}, `telemetry:${ep}`);
    } catch (err) {
      warn(`telemetry stub transform failed, fallback official: ${ep}`);
      return undefined;
    }
  }

  const route = decideRoute({ cfg, endpoint: ep, body, runtimeEnabled: state.runtimeEnabled });
  if (route.mode === "official") return undefined;
  if (route.mode === "disabled") {
    try {
      return safeTransform(transform, {}, `disabled:${ep}`);
    } catch {
      return {};
    }
  }
  if (route.mode !== "byok") return undefined;

  const t = Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : cfg.timeouts.upstreamMs;

  if (ep === "/get-models") {
    const byokModels = buildByokModelsFromConfig(cfg);
    const byokDefaultModel = byokModels.length ? byokModels[0] : "";
    const activeProviderId = normalizeString(cfg?.routing?.defaultProviderId) || normalizeString(cfg?.providers?.[0]?.id);
    const activeProvider = Array.isArray(cfg?.providers) ? cfg.providers.find((p) => p && normalizeString(p.id) === activeProviderId) : null;
    const activeProviderDefaultModel = normalizeString(activeProvider?.defaultModel) || normalizeString(activeProvider?.models?.[0]);
    const preferredByok = activeProviderId && activeProviderDefaultModel ? `byok:${activeProviderId}:${activeProviderDefaultModel}` : "";
    const preferredDefaultModel = byokModels.includes(preferredByok) ? preferredByok : byokDefaultModel;
    try {
      const off = getOfficialConnection();
      const completionURL = normalizeString(upstreamCompletionURL) || off.completionURL;
      const apiToken = normalizeRawToken(upstreamApiToken) || off.apiToken;
      const upstream = await fetchOfficialGetModels({ completionURL, apiToken, timeoutMs: Math.min(12000, t), abortSignal });
      const merged = mergeModels(upstream, byokModels, { defaultModel: preferredDefaultModel });
      return safeTransform(transform, merged, ep);
    } catch (err) {
      warn(`get-models fallback to local: ${err instanceof Error ? err.message : String(err)}`);
      const local = makeBackGetModelsResult({ defaultModel: preferredDefaultModel || "unknown", models: byokModels.map(makeModelInfo) });
      return safeTransform(transform, local, ep);
    }
  }

  if (ep === "/completion" || ep === "/chat-input-completion") {
    const { system, messages } = buildMessagesForEndpoint(ep, body);
    const text = await byokCompleteText({ provider: route.provider, model: route.model, system, messages, timeoutMs: t, abortSignal });
    return safeTransform(transform, makeBackCompletionResult(text), ep);
  }

  if (ep === "/edit") {
    const { system, messages } = buildMessagesForEndpoint(ep, body);
    const text = await byokCompleteText({ provider: route.provider, model: route.model, system, messages, timeoutMs: t, abortSignal });
    return safeTransform(transform, makeBackTextResult(text), ep);
  }

  if (ep === "/chat") {
    const { type, baseUrl, apiKey, extraHeaders, requestDefaults } = providerRequestContext(route.provider);
    const req = normalizeAugmentChatRequest(body);
    const msg = normalizeString(req.message);
    const hasNodes = Array.isArray(req.nodes) && req.nodes.length;
    const hasHistory = Array.isArray(req.chat_history) && req.chat_history.length;
    const hasReqNodes = (Array.isArray(req.structured_request_nodes) && req.structured_request_nodes.length) || (Array.isArray(req.request_nodes) && req.request_nodes.length);
    if (!msg && !hasNodes && !hasHistory && !hasReqNodes) return safeTransform(transform, makeBackChatResult("", { nodes: [] }), ep);
    try {
      await maybeSummarizeAndCompactAugmentChatRequest({ cfg, req, requestedModel: route.requestedModel, fallbackProvider: route.provider, fallbackModel: route.model, timeoutMs: t, abortSignal });
    } catch (err) {
      warn(`historySummary failed (ignored): ${err instanceof Error ? err.message : String(err)}`);
    }
    await maybeInjectOfficialCodebaseRetrieval({ req, timeoutMs: t, abortSignal, upstreamCompletionURL, upstreamApiToken });
    await maybeInjectOfficialContextCanvas({ req, timeoutMs: t, abortSignal, upstreamCompletionURL, upstreamApiToken });
    await maybeInjectOfficialExternalSources({ req, timeoutMs: t, abortSignal, upstreamCompletionURL, upstreamApiToken });
    if (type === "openai_compatible") {
      const text = await openAiCompleteText({ baseUrl, apiKey, model: route.model, messages: buildOpenAiMessages(req), timeoutMs: t, abortSignal, extraHeaders, requestDefaults });
      return safeTransform(transform, makeBackChatResult(text, { nodes: [] }), ep);
    }
    if (type === "anthropic") {
      const text = await anthropicCompleteText({ baseUrl, apiKey, model: route.model, system: buildSystemPrompt(req), messages: buildAnthropicMessages(req), timeoutMs: t, abortSignal, extraHeaders, requestDefaults });
      return safeTransform(transform, makeBackChatResult(text, { nodes: [] }), ep);
    }
    throw new Error(`未知 provider.type: ${type}`);
  }

  if (ep === "/next_edit_loc") {
    const candidate_locations = pickNextEditLocationCandidates(body);
    return safeTransform(transform, makeBackNextEditLocationResult(candidate_locations), ep);
  }

  return undefined;
}

async function maybeHandleCallApiStream({ endpoint, body, transform, timeoutMs, abortSignal, upstreamApiToken, upstreamCompletionURL }) {
  const ep = normalizeEndpoint(endpoint);
  if (!ep) return undefined;

  const cfgMgr = ensureConfigManager();
  const cfg = cfgMgr.get();
  if (!state.runtimeEnabled) return undefined;

  const route = decideRoute({ cfg, endpoint: ep, body, runtimeEnabled: state.runtimeEnabled });
  if (route.mode === "official") return undefined;
  if (route.mode === "disabled") return emptyAsyncGenerator();
  if (route.mode !== "byok") return undefined;

  const t = Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0 ? Number(timeoutMs) : cfg.timeouts.upstreamMs;

  if (isTelemetryDisabled(cfg, ep)) return emptyAsyncGenerator();

  if (ep === "/chat-stream") {
    const src = byokChatStream({ cfg, provider: route.provider, model: route.model, requestedModel: route.requestedModel, body, timeoutMs: t, abortSignal, upstreamApiToken, upstreamCompletionURL });
    return guardObjectStream({
      ep,
      src,
      transform,
      makeErrorChunk: (err) => makeBackChatChunk({ text: makeEndpointErrorText(ep, err), stop_reason: STOP_REASON_END_TURN })
    });
  }

  if (ep === "/prompt-enhancer" || ep === "/generate-conversation-title") {
    const { system, messages } = buildMessagesForEndpoint(ep, body);
    const src = byokStreamText({ provider: route.provider, model: route.model, system, messages, timeoutMs: t, abortSignal });
    return guardObjectStream({
      ep,
      transform,
      src: (async function* () { for await (const delta of src) yield makeBackChatResult(delta, { nodes: [] }); })(),
      makeErrorChunk: (err) => makeBackChatResult(makeEndpointErrorText(ep, err), { nodes: [] })
    });
  }

  if (ep === "/instruction-stream" || ep === "/smart-paste-stream") {
    const { system, messages } = buildMessagesForEndpoint(ep, body);
    const meta = await buildInstructionReplacementMeta(body);
    const src = byokStreamText({ provider: route.provider, model: route.model, system, messages, timeoutMs: t, abortSignal });
    return guardObjectStream({
      ep,
      transform,
      src: (async function* () {
        yield { text: "", ...meta };
        for await (const delta of src) {
          const t = typeof delta === "string" ? delta : String(delta ?? "");
          if (!t) continue;
          yield { text: t, replacement_text: t };
        }
      })(),
      makeErrorChunk: (err) => ({ text: makeEndpointErrorText(ep, err), ...meta })
    });
  }

  if (ep === "/generate-commit-message-stream") {
    const { system, messages } = buildMessagesForEndpoint(ep, body);
    const src = byokStreamText({ provider: route.provider, model: route.model, system, messages, timeoutMs: t, abortSignal });
    return guardObjectStream({
      ep,
      transform,
      src: (async function* () { for await (const delta of src) yield makeBackChatResult(delta, { nodes: [] }); })(),
      makeErrorChunk: (err) => makeBackChatResult(makeEndpointErrorText(ep, err), { nodes: [] })
    });
  }

  if (ep === "/next-edit-stream") {
    const b = body && typeof body === "object" ? body : {};
    const selectionBegin = Number.isFinite(Number(b.selection_begin_char)) ? Number(b.selection_begin_char) : 0;
    const selectionEnd = Number.isFinite(Number(b.selection_end_char)) ? Number(b.selection_end_char) : selectionBegin;
    const existingCode = typeof b.selected_text === "string" ? b.selected_text : "";

    const { system, messages } = buildMessagesForEndpoint(ep, body);
    const suggestedCode = await byokCompleteText({ provider: route.provider, model: route.model, system, messages, timeoutMs: t, abortSignal });

    const raw = makeBackNextEditGenerationChunk({
      path: normalizeString(b.path),
      blobName: normalizeString(b.blob_name),
      charStart: selectionBegin,
      charEnd: selectionEnd,
      existingCode,
      suggestedCode
    });
    return (async function* () { yield safeTransform(transform, raw, ep); })();
  }

  return undefined;
}

module.exports = { maybeHandleCallApi, maybeHandleCallApiStream };
