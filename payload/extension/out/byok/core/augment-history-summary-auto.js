"use strict";

const { debug } = require("../infra/log");
const { normalizeString } = require("../infra/util");
const { openAiCompleteText } = require("../providers/openai");
const { anthropicCompleteText } = require("../providers/anthropic");
const { buildSystemPrompt, buildOpenAiMessages, buildAnthropicMessages } = require("./augment-chat");
const { renderHistorySummaryNodeValue } = require("./augment-history-summary");
const shared = require("./augment-chat.shared");
const { buildAbridgedHistoryText, exchangeRequestNodes, exchangeResponseNodes } = require("./augment-history-summary-auto.abridged");
const { REQUEST_NODE_TEXT, REQUEST_NODE_TOOL_RESULT, REQUEST_NODE_HISTORY_SUMMARY } = require("./augment-protocol");
const { asRecord, asArray, asString, pick, normalizeNodeType } = shared;

const HISTORY_SUMMARY_CACHE = new Map();

function nowMs() {
  return Date.now();
}

function approxTokenCountFromByteLen(len) {
  const BYTES_PER_TOKEN = 4;
  return Math.ceil(Number(len) / BYTES_PER_TOKEN);
}

function nodeIsToolResult(n) {
  if (normalizeNodeType(n) !== REQUEST_NODE_TOOL_RESULT) return false;
  const tr = pick(n, ["tool_result_node", "toolResultNode"]);
  return tr && typeof tr === "object" && !Array.isArray(tr);
}

function hasHistorySummaryNode(nodes) {
  return asArray(nodes).some((n) => normalizeNodeType(n) === REQUEST_NODE_HISTORY_SUMMARY && pick(n, ["history_summary_node", "historySummaryNode"]) != null);
}

function historyContainsSummary(history) {
  return asArray(history).some((h) => {
    const it = asRecord(h);
    return hasHistorySummaryNode(it.request_nodes) || hasHistorySummaryNode(it.structured_request_nodes) || hasHistorySummaryNode(it.nodes);
  });
}

function exchangeHasToolResults(h) {
  return exchangeRequestNodes(h).some(nodeIsToolResult);
}

function estimateNodeSizeChars(node) {
  const n = asRecord(node);
  let out = 16;
  out += asString(pick(n, ["content"])).length;
  out += asString(pick(pick(n, ["text_node", "textNode"]), ["content"])).length;
  const tr = asRecord(pick(n, ["tool_result_node", "toolResultNode"]));
  if (normalizeNodeType(n) === REQUEST_NODE_TOOL_RESULT) {
    out += asString(pick(tr, ["tool_use_id", "toolUseId"])).length;
    out += asString(pick(tr, ["content"])).length;
    for (const c of asArray(pick(tr, ["content_nodes", "contentNodes"]))) {
      const cr = asRecord(c);
      out += 8;
      out += asString(pick(cr, ["text_content", "textContent"])).length;
      const img = asRecord(pick(cr, ["image_content", "imageContent"]));
      out += asString(pick(img, ["image_data", "imageData"])).length;
    }
  }
  const img = asRecord(pick(n, ["image_node", "imageNode"]));
  out += asString(pick(img, ["image_data", "imageData"])).length;
  for (const v of [pick(n, ["image_id_node", "imageIdNode"]), pick(n, ["ide_state_node", "ideStateNode"]), pick(n, ["edit_events_node", "editEventsNode"]), pick(n, ["checkpoint_ref_node", "checkpointRefNode"]), pick(n, ["change_personality_node", "changePersonalityNode"]), pick(n, ["file_node", "fileNode"]), pick(n, ["file_id_node", "fileIdNode"]), pick(n, ["history_summary_node", "historySummaryNode"])]) {
    if (v == null) continue;
    try { out += JSON.stringify(v).length; } catch {}
  }
  const tu = asRecord(pick(n, ["tool_use", "toolUse"]));
  out += asString(pick(tu, ["tool_use_id", "toolUseId"])).length;
  out += asString(pick(tu, ["tool_name", "toolName"])).length;
  out += asString(pick(tu, ["input_json", "inputJson"])).length;
  out += asString(pick(tu, ["mcp_server_name", "mcpServerName"])).length;
  out += asString(pick(tu, ["mcp_tool_name", "mcpToolName"])).length;
  const th = asRecord(pick(n, ["thinking", "thinking_node", "thinkingNode"]));
  out += asString(pick(th, ["summary"])).length;
  return out;
}

function estimateExchangeSizeChars(h) {
  const it = asRecord(h);
  const reqNodes = exchangeRequestNodes(it);
  const respNodes = exchangeResponseNodes(it);
  let n = 0;
  n += reqNodes.length ? reqNodes.map(estimateNodeSizeChars).reduce((a, b) => a + b, 0) : asString(it.request_message).length;
  n += respNodes.length ? respNodes.map(estimateNodeSizeChars).reduce((a, b) => a + b, 0) : asString(it.response_text).length;
  return n;
}

function estimateHistorySizeChars(history) {
  return asArray(history).map(estimateExchangeSizeChars).reduce((a, b) => a + b, 0);
}

function splitHistoryForSummary(history, tailSizeCharsToExclude, triggerOnHistorySizeChars, minTailExchanges) {
  const hs = asArray(history);
  if (!hs.length) return { head: [], tail: [] };
  const headRev = [];
  const tailRev = [];
  let seenChars = 0;
  let headChars = 0;
  let tailChars = 0;
  for (let i = hs.length - 1; i >= 0; i--) {
    const ex = hs[i];
    const sz = estimateExchangeSizeChars(ex);
    if (seenChars + sz < tailSizeCharsToExclude || tailRev.length < minTailExchanges) {
      tailRev.push(ex);
      tailChars += sz;
    } else {
      headRev.push(ex);
      headChars += sz;
    }
    seenChars += sz;
  }
  const totalChars = headChars + tailChars;
  if (totalChars < triggerOnHistorySizeChars) {
    const all = tailRev.concat(headRev).reverse();
    return { head: [], tail: all };
  }
  headRev.reverse();
  tailRev.reverse();
  return { head: headRev, tail: tailRev };
}

function adjustTailToAvoidToolResultOrphans(original, tailStart) {
  const hs = asArray(original);
  let start = Number.isFinite(Number(tailStart)) ? Math.floor(Number(tailStart)) : 0;
  while (start < hs.length) {
    if (!exchangeHasToolResults(hs[start])) break;
    if (start <= 0) break;
    start -= 1;
  }
  return start;
}

function resolveContextWindowTokens(hs, requestedModel) {
  const model = normalizeString(requestedModel);
  if (!model) return null;
  const overrides = asRecord(hs?.contextWindowTokensOverrides);
  const keys = Object.keys(overrides).sort((a, b) => String(b).length - String(a).length);
  for (const k of keys) {
    const key = normalizeString(k);
    const v = Number(overrides[k]);
    if (!key || !Number.isFinite(v) || v <= 0) continue;
    if (model.includes(key)) return Math.floor(v);
  }
  const d = Number(hs?.contextWindowTokensDefault);
  return Number.isFinite(d) && d > 0 ? Math.floor(d) : null;
}

function resolveHistorySummaryConfig(cfg) {
  const c = cfg && typeof cfg === "object" ? cfg : {};
  const hs = (c.historySummary && typeof c.historySummary === "object" && !Array.isArray(c.historySummary) ? c.historySummary : null) || (c.history_summary && typeof c.history_summary === "object" && !Array.isArray(c.history_summary) ? c.history_summary : null);
  if (!hs) return null;
  if (hs.enabled !== true) return null;
  const triggerOnHistorySizeChars = Number(hs.triggerOnHistorySizeChars);
  const maxTokens = Number(hs.maxTokens);
  const minTailExchanges = Number(hs.minTailExchanges);
  const tChars = Number.isFinite(triggerOnHistorySizeChars) && triggerOnHistorySizeChars > 0 ? Math.floor(triggerOnHistorySizeChars) : 0;
  if (!tChars) return null;
  return {
    ...hs,
    triggerOnHistorySizeChars: tChars,
    historyTailSizeCharsToExclude: Math.max(0, Math.floor(Number(hs.historyTailSizeCharsToExclude) || 0)),
    minTailExchanges: Number.isFinite(minTailExchanges) && minTailExchanges > 0 ? Math.floor(minTailExchanges) : 2,
    maxTokens: Number.isFinite(maxTokens) && maxTokens > 0 ? Math.floor(maxTokens) : 1024,
    timeoutSeconds: Math.max(1, Math.floor(Number(hs.timeoutSeconds) || 60)),
    cacheTtlMs: Math.max(0, Math.floor(Number(hs.cacheTtlMs) || 0)),
    maxSummarizationInputChars: Math.max(0, Math.floor(Number(hs.maxSummarizationInputChars) || 0)),
    triggerStrategy: normalizeString(hs.triggerStrategy) || "auto",
    triggerOnContextRatio: Number(hs.triggerOnContextRatio) || 0.7,
    targetContextRatio: Number(hs.targetContextRatio) || 0.55,
    prompt: typeof hs.prompt === "string" ? hs.prompt : "",
    abridgedHistoryParams: asRecord(hs.abridgedHistoryParams)
  };
}

function pickProviderById(cfg, providerId) {
  const pid = normalizeString(providerId);
  const list = Array.isArray(cfg?.providers) ? cfg.providers : [];
  if (!pid) return null;
  return list.find((p) => p && normalizeString(p.id) === pid) || null;
}

function normalizeProviderRequestDefaults(provider, maxTokens) {
  const base = provider && typeof provider === "object" && provider.requestDefaults && typeof provider.requestDefaults === "object" && !Array.isArray(provider.requestDefaults) ? provider.requestDefaults : {};
  const out = { ...base };
  if (Number.isFinite(Number(maxTokens)) && Number(maxTokens) > 0) out.max_tokens = Math.floor(Number(maxTokens));
  if (out.thinking) delete out.thinking;
  if (out.tools) delete out.tools;
  if (out.tool_choice) delete out.tool_choice;
  if (out.toolChoice) delete out.toolChoice;
  return out;
}

async function runSummaryModelOnce({ provider, model, prompt, chatHistory, maxTokens, timeoutMs, abortSignal }) {
  const p = provider && typeof provider === "object" ? provider : null;
  const type = normalizeString(p?.type);
  const baseUrl = normalizeString(p?.baseUrl);
  const apiKey = normalizeString(p?.apiKey);
  const extraHeaders = p?.headers && typeof p.headers === "object" && !Array.isArray(p.headers) ? p.headers : {};
  const requestDefaults = normalizeProviderRequestDefaults(p, maxTokens);
  if (!type || !baseUrl || !apiKey || !normalizeString(model)) throw new Error("historySummary provider/model 未配置");
  if (!normalizeString(prompt) || !Array.isArray(chatHistory) || !chatHistory.length) throw new Error("historySummary prompt/chatHistory 为空");

  const augmentReq = {
    message: prompt,
    conversation_id: "",
    chat_history: chatHistory,
    tool_definitions: [],
    nodes: [],
    structured_request_nodes: [],
    request_nodes: [],
    agent_memories: "",
    mode: "",
    prefix: "",
    suffix: "",
    lang: "",
    path: "",
    user_guidelines: "",
    workspace_guidelines: "",
    rules: null,
    feature_detection_flags: {}
  };

  if (type === "openai_compatible") {
    return await openAiCompleteText({ baseUrl, apiKey, model, messages: buildOpenAiMessages(augmentReq), timeoutMs, abortSignal, extraHeaders, requestDefaults });
  }
  if (type === "anthropic") {
    return await anthropicCompleteText({ baseUrl, apiKey, model, system: buildSystemPrompt(augmentReq), messages: buildAnthropicMessages(augmentReq), timeoutMs, abortSignal, extraHeaders, requestDefaults });
  }
  throw new Error(`historySummary 未知 provider.type: ${type}`);
}

function cacheGetFresh(conversationId, boundaryRequestId, now, ttlMs) {
  const cid = normalizeString(conversationId);
  const bid = normalizeString(boundaryRequestId);
  if (!cid || !bid) return null;
  const e = HISTORY_SUMMARY_CACHE.get(cid);
  if (!e) return null;
  if (ttlMs > 0 && now - Number(e.updatedAtMs || 0) > ttlMs) return null;
  if (normalizeString(e.summarizedUntilRequestId) !== bid) return null;
  return { summaryText: asString(e.summaryText), summarizationRequestId: asString(e.summarizationRequestId) };
}

function cacheGetFreshState(conversationId, now, ttlMs) {
  const cid = normalizeString(conversationId);
  if (!cid) return null;
  const e = HISTORY_SUMMARY_CACHE.get(cid);
  if (!e) return null;
  if (ttlMs > 0 && now - Number(e.updatedAtMs || 0) > ttlMs) return null;
  return { ...e };
}

function cachePut(conversationId, boundaryRequestId, summaryText, summarizationRequestId, now) {
  const cid = normalizeString(conversationId);
  const bid = normalizeString(boundaryRequestId);
  if (!cid || !bid) return;
  HISTORY_SUMMARY_CACHE.set(cid, { summaryText: asString(summaryText), summarizedUntilRequestId: bid, summarizationRequestId: asString(summarizationRequestId), updatedAtMs: nowMs() });
}

function computeTriggerDecision({ hs, requestedModel, totalWithExtra, convId }) {
  const triggerOnHistorySizeChars = Number(hs.triggerOnHistorySizeChars);
  const baseDecision = { thresholdChars: triggerOnHistorySizeChars, tailExcludeChars: hs.historyTailSizeCharsToExclude };
  const strategy = normalizeString(hs.triggerStrategy).toLowerCase();
  if (strategy === "chars") return totalWithExtra >= triggerOnHistorySizeChars ? baseDecision : null;

  const cwTokens = resolveContextWindowTokens(hs, requestedModel);
  if ((strategy === "ratio" || strategy === "auto") && cwTokens) {
    const approxTotalTokens = approxTokenCountFromByteLen(totalWithExtra);
    const ratio = cwTokens ? approxTotalTokens / cwTokens : 0;
    const triggerRatio = Number(hs.triggerOnContextRatio) || 0.7;
    if (ratio < triggerRatio) return null;
    const targetRatio = Number(hs.targetContextRatio) || 0.55;
    const thresholdTokens = Math.ceil(cwTokens * triggerRatio);
    const thresholdChars = thresholdTokens * 4;
    const targetTokens = Math.floor(cwTokens * targetRatio);
    const targetCharsBudget = targetTokens * 4;
    const summaryOverhead = (Number(hs.abridgedHistoryParams?.totalCharsLimit) || 0) + (Number(hs.maxTokens) || 0) * 4 + 4096;
    const tailExcludeChars = Math.max(0, targetCharsBudget - summaryOverhead);
    debug(`historySummary trigger ratio: conv=${convId} model=${normalizeString(requestedModel)} tokens≈${approxTotalTokens}/${cwTokens} ratio≈${ratio.toFixed(3)}`);
    return { thresholdChars, tailExcludeChars };
  }

  return totalWithExtra >= triggerOnHistorySizeChars ? baseDecision : null;
}

function computeTailSelection({ history, hs, decision }) {
  const split = splitHistoryForSummary(history, decision.tailExcludeChars, decision.thresholdChars, hs.minTailExchanges);
  if (!split.head.length || !split.tail.length) return null;
  const splitBoundaryRequestId = normalizeString(split.tail[0]?.request_id);
  if (!splitBoundaryRequestId) return null;
  let tailStart = history.findIndex((h) => normalizeString(h?.request_id) === splitBoundaryRequestId);
  if (tailStart < 0) tailStart = Math.max(0, history.length - split.tail.length);
  tailStart = adjustTailToAvoidToolResultOrphans(history, tailStart);
  const boundaryRequestId = normalizeString(history[tailStart]?.request_id);
  if (!boundaryRequestId) return null;
  const droppedHead = history.slice(0, tailStart);
  const tail = history.slice(tailStart);
  if (!droppedHead.length || !tail.length) return null;
  return { tailStart, boundaryRequestId, droppedHead, tail };
}

async function resolveSummaryText({ hs, cfg, convId, boundaryRequestId, history, tailStart, droppedHead, fallbackProvider, fallbackModel, timeoutMs, abortSignal }) {
  const now = nowMs();
  const cached = cacheGetFresh(convId, boundaryRequestId, now, hs.cacheTtlMs);
  if (cached) return { summaryText: cached.summaryText, summarizationRequestId: cached.summarizationRequestId, now };

  const provider = pickProviderById(cfg, hs.providerId) || fallbackProvider;
  const model = normalizeString(hs.model) || normalizeString(fallbackModel);
  let prompt = asString(hs.prompt);
  let inputHistory = droppedHead.slice();

  let usedRolling = false;
  if (hs.rollingSummary === true) {
    const prev = cacheGetFreshState(convId, now, hs.cacheTtlMs);
    if (prev && normalizeString(prev.summarizedUntilRequestId) && normalizeString(prev.summarizedUntilRequestId) !== boundaryRequestId) {
      const prevBoundaryPos = history.findIndex((h) => normalizeString(h?.request_id) === normalizeString(prev.summarizedUntilRequestId));
      if (prevBoundaryPos >= 0 && prevBoundaryPos < tailStart) {
        const delta = history.slice(prevBoundaryPos, tailStart);
        if (delta.length) {
          const prevExchange = { request_id: "byok_history_summary_prev", request_message: `[PREVIOUS_SUMMARY]\n${asString(prev.summaryText).trim()}\n[/PREVIOUS_SUMMARY]`, response_text: "", request_nodes: [], structured_request_nodes: [], nodes: [], response_nodes: [], structured_output_nodes: [] };
          inputHistory = [prevExchange, ...delta];
          usedRolling = true;
          prompt = `${normalizeString(hs.prompt)}\n\nYou will be given an existing summary and additional new conversation turns. Update the summary to include the new information. Output only the updated summary.`;
        }
      }
    }
  }

  const maxIn = Number(hs.maxSummarizationInputChars) || 0;
  if (maxIn > 0) {
    const shrink = () => estimateHistorySizeChars(inputHistory) > maxIn;
    if (usedRolling) while (inputHistory.length > 1 && shrink()) inputHistory.splice(1, 1);
    else while (inputHistory.length && shrink()) inputHistory.shift();
  }
  if (!inputHistory.length) return null;

  const timeout = Math.min(Math.max(1000, Number(timeoutMs) || 120000), hs.timeoutSeconds * 1000);
  const summaryText = normalizeString(await runSummaryModelOnce({ provider, model, prompt, chatHistory: inputHistory, maxTokens: hs.maxTokens, timeoutMs: timeout, abortSignal }));
  if (!summaryText) return null;
  const summarizationRequestId = `byok_history_summary_${now}`;
  cachePut(convId, boundaryRequestId, summaryText, summarizationRequestId, now);
  return { summaryText, summarizationRequestId, now };
}

function buildHistoryEnd(tail) {
  return asArray(tail).map((h) => {
    const it = asRecord(h);
    return { request_id: it.request_id, request_message: it.request_message, response_text: it.response_text, request_nodes: exchangeRequestNodes(it), response_nodes: exchangeResponseNodes(it) };
  });
}

function applySummaryToRequest({ hs, req, tail, summaryText, summarizationRequestId, abridged }) {
  const template = asString(hs.summaryNodeRequestMessageTemplate);
  const historyEnd = buildHistoryEnd(tail);
  const summaryNode = { summary_text: summaryText, summarization_request_id: summarizationRequestId, history_beginning_dropped_num_exchanges: abridged.droppedBeginning, history_middle_abridged_text: abridged.text, history_end: historyEnd, message_template: template };
  const rendered = renderHistorySummaryNodeValue(summaryNode, []);
  if (!normalizeString(rendered)) return null;
  const summaryItem = { request_id: "byok_history_summary", request_message: "", response_text: "", request_nodes: [{ id: -10, type: REQUEST_NODE_TEXT, content: "", text_node: { content: rendered } }], structured_request_nodes: [], nodes: [], response_nodes: [], structured_output_nodes: [] };
  const newHistory = [summaryItem];
  req.chat_history = newHistory;
  return newHistory;
}

async function maybeSummarizeAndCompactAugmentChatRequest({ cfg, req, requestedModel, fallbackProvider, fallbackModel, timeoutMs, abortSignal }) {
  const hs = resolveHistorySummaryConfig(cfg);
  if (!hs) return false;
  const convId = normalizeString(req?.conversation_id);
  if (!convId) return false;
  const history = asArray(req?.chat_history);
  if (!history.length) return false;
  if (historyContainsSummary(history)) return false;

  const totalChars = estimateHistorySizeChars(history);
  const totalWithExtra = totalChars + asString(req?.message).length;
  const decision = computeTriggerDecision({ hs, requestedModel, totalWithExtra, convId });
  if (!decision) return false;

  const sel = computeTailSelection({ history, hs, decision });
  if (!sel) return false;

  const abridged = buildAbridgedHistoryText(history, hs.abridgedHistoryParams, sel.boundaryRequestId);
  const summary = await resolveSummaryText({ hs, cfg, convId, boundaryRequestId: sel.boundaryRequestId, history, tailStart: sel.tailStart, droppedHead: sel.droppedHead, fallbackProvider, fallbackModel, timeoutMs, abortSignal });
  if (!summary) return false;

  const newHistory = applySummaryToRequest({ hs, req, tail: sel.tail, summaryText: summary.summaryText, summarizationRequestId: summary.summarizationRequestId, abridged });
  if (!newHistory) return false;

  const afterChars = estimateHistorySizeChars(newHistory);
  debug(`historySummary applied: conv=${convId} before≈${totalChars} after≈${afterChars} tailStart=${sel.tailStart}`);
  return true;
}

module.exports = { maybeSummarizeAndCompactAugmentChatRequest };
