"use strict";

const {
  REQUEST_NODE_TEXT,
  REQUEST_NODE_TOOL_RESULT,
  REQUEST_NODE_HISTORY_SUMMARY,
  RESPONSE_NODE_RAW_RESPONSE,
  RESPONSE_NODE_THINKING,
  RESPONSE_NODE_TOOL_USE
} = require("./augment-protocol");

function asRecord(v) {
  return v && typeof v === "object" && !Array.isArray(v) ? v : {};
}

function asArray(v) {
  return Array.isArray(v) ? v : [];
}

function asString(v) {
  if (typeof v === "string") return v;
  if (v == null) return "";
  return String(v);
}

function pick(obj, keys) {
  const o = asRecord(obj);
  for (const k of Array.isArray(keys) ? keys : []) if (Object.prototype.hasOwnProperty.call(o, k)) return o[k];
  return undefined;
}

function normalizeNodeType(node) {
  const v = pick(node, ["type", "node_type", "nodeType"]);
  const n = Number(v);
  return Number.isFinite(n) ? n : -1;
}

function normalizeJoinedLines(lines) {
  let out = "";
  for (const raw of lines) {
    const line = asString(raw).replace(/\n+$/g, "");
    if (!line.trim()) continue;
    if (out) out += "\n";
    out += line;
  }
  return out;
}

function extractUserMessageFromRequestNodes(nodes, fallback) {
  const joined = normalizeJoinedLines(
    asArray(nodes)
      .filter((n) => normalizeNodeType(n) === REQUEST_NODE_TEXT)
      .map((n) => asString(pick(pick(n, ["text_node", "textNode"]), ["content"])))
  );
  const fb = asString(fallback);
  return joined.trim() ? joined : fb;
}

function buildExchangeRenderCtx(ex) {
  const r = asRecord(ex);
  const requestMessage = asString(pick(r, ["request_message", "requestMessage"]));
  const requestNodes = asArray(pick(r, ["request_nodes", "requestNodes"]));
  const responseNodes = asArray(pick(r, ["response_nodes", "responseNodes"]));

  const toolResults = requestNodes
    .filter((n) => normalizeNodeType(n) === REQUEST_NODE_TOOL_RESULT)
    .map((n) => asRecord(pick(n, ["tool_result_node", "toolResultNode"])))
    .filter((tr) => asString(pick(tr, ["tool_use_id", "toolUseId"])).trim())
    .map((tr) => ({ id: asString(pick(tr, ["tool_use_id", "toolUseId"])), content: asString(pick(tr, ["content"])), is_error: Boolean(pick(tr, ["is_error", "isError"])) }));

  const thinking = normalizeJoinedLines(
    responseNodes
      .filter((n) => normalizeNodeType(n) === RESPONSE_NODE_THINKING)
      .map((n) => asString(pick(pick(n, ["thinking", "thinking_node", "thinkingNode"]), ["summary"])))
      .filter((s) => s.trim())
  );

  const responseText = normalizeJoinedLines(
    responseNodes
      .filter((n) => normalizeNodeType(n) === RESPONSE_NODE_RAW_RESPONSE)
      .map((n) => asString(pick(n, ["content"])))
      .filter((s) => s.trim())
  );

  const toolUses = responseNodes
    .filter((n) => normalizeNodeType(n) === RESPONSE_NODE_TOOL_USE)
    .map((n) => asRecord(pick(n, ["tool_use", "toolUse"])))
    .filter((tu) => asString(pick(tu, ["tool_use_id", "toolUseId"])).trim() && asString(pick(tu, ["tool_name", "toolName"])).trim())
    .map((tu) => ({ name: asString(pick(tu, ["tool_name", "toolName"])), id: asString(pick(tu, ["tool_use_id", "toolUseId"])), input: asString(pick(tu, ["input_json", "inputJson"])) }));

  return {
    user_message: extractUserMessageFromRequestNodes(requestNodes, requestMessage),
    tool_results: toolResults,
    thinking,
    response_text: responseText,
    tool_uses: toolUses,
    has_response: Boolean(thinking || responseText || toolUses.length)
  };
}

function renderExchangeFull(ctx) {
  const out = [];
  out.push("<exchange>");
  out.push("  <user_request_or_tool_results>");
  const userMessage = asString(ctx?.user_message).replace(/\n+$/g, "");
  if (userMessage.trim()) out.push(userMessage);
  for (const tr of asArray(ctx?.tool_results)) {
    const id = asString(tr?.id).trim();
    if (!id) continue;
    out.push(`    <tool_result tool_use_id="${id}" is_error="${tr?.is_error ? "true" : "false"}">`);
    const content = asString(tr?.content).replace(/\n+$/g, "");
    if (content.trim()) out.push(content);
    out.push("    </tool_result>");
  }
  out.push("  </user_request_or_tool_results>");
  if (ctx?.has_response) {
    out.push("  <agent_response_or_tool_uses>");
    const thinking = asString(ctx?.thinking).replace(/\n+$/g, "");
    if (thinking.trim()) {
      out.push("    <thinking>");
      out.push(thinking);
      out.push("    </thinking>");
    }
    const responseText = asString(ctx?.response_text).replace(/\n+$/g, "");
    if (responseText.trim()) out.push(responseText);
    for (const tu of asArray(ctx?.tool_uses)) {
      const name = asString(tu?.name).trim();
      const id = asString(tu?.id).trim();
      if (!name || !id) continue;
      out.push(`    <tool_use name="${name}" tool_use_id="${id}">`);
      const input = asString(tu?.input).replace(/\n+$/g, "");
      if (input.trim()) out.push(input);
      out.push("    </tool_use>");
    }
    out.push("  </agent_response_or_tool_uses>");
  }
  out.push("</exchange>");
  return out.join("\n");
}

function replacePlaceholders(template, repl) {
  let out = asString(template);
  for (const [k, v] of Array.isArray(repl) ? repl : []) {
    if (!out.includes(k)) continue;
    out = out.split(k).join(asString(v));
  }
  return out;
}

function normalizeHistoryEndExchange(raw) {
  const r = asRecord(raw);
  return {
    request_message: asString(pick(r, ["request_message", "requestMessage"])),
    response_text: asString(pick(r, ["response_text", "responseText"])),
    request_nodes: asArray(pick(r, ["request_nodes", "requestNodes"])),
    response_nodes: asArray(pick(r, ["response_nodes", "responseNodes"]))
  };
}

function renderHistorySummaryNodeValue(v, extraToolResults) {
  const r = asRecord(v);
  const messageTemplate = asString(pick(r, ["message_template", "messageTemplate"]));
  if (!messageTemplate.trim()) return null;

  const summaryText = asString(pick(r, ["summary_text", "summaryText"]));
  const summarizationRequestId = asString(pick(r, ["summarization_request_id", "summarizationRequestId"]));
  const historyBeginningDroppedNumExchanges = Number(pick(r, ["history_beginning_dropped_num_exchanges", "historyBeginningDroppedNumExchanges"])) || 0;
  const historyMiddleAbridgedText = asString(pick(r, ["history_middle_abridged_text", "historyMiddleAbridgedText"]));
  const historyEnd = asArray(pick(r, ["history_end", "historyEnd"])).map(normalizeHistoryEndExchange);
  const extra = asArray(extraToolResults);
  if (extra.length) historyEnd.push({ request_message: "", response_text: "", request_nodes: extra, response_nodes: [] });

  const endPartFull = historyEnd.map(buildExchangeRenderCtx).map(renderExchangeFull).join("\n");
  const abridged = historyMiddleAbridgedText;

  return replacePlaceholders(messageTemplate, [
    ["{summary}", summaryText],
    ["{summarization_request_id}", summarizationRequestId],
    ["{beginning_part_dropped_num_exchanges}", String(historyBeginningDroppedNumExchanges)],
    ["{middle_part_abridged}", abridged],
    ["{end_part_full}", endPartFull],
    ["{abridged_history}", abridged]
  ]);
}

function hasHistorySummaryNode(nodes) {
  return asArray(nodes).some((n) => normalizeNodeType(n) === REQUEST_NODE_HISTORY_SUMMARY && pick(n, ["history_summary_node", "historySummaryNode"]) != null);
}

function chatHistoryItemHasSummary(item) {
  const it = asRecord(item);
  return hasHistorySummaryNode(pick(it, ["request_nodes", "requestNodes"])) || hasHistorySummaryNode(pick(it, ["structured_request_nodes", "structuredRequestNodes"])) || hasHistorySummaryNode(pick(it, ["nodes"]));
}

function compactAugmentChatHistory(chatHistory) {
  const list = Array.isArray(chatHistory) ? chatHistory : null;
  if (!list || !list.length) return;

  let start = -1;
  for (let i = list.length - 1; i >= 0; i--) {
    if (chatHistoryItemHasSummary(list[i])) { start = i; break; }
  }
  if (start < 0) return;
  if (start > 0) list.splice(0, start);

  const first = list[0];
  if (!first || typeof first !== "object") return;

  const reqNodes = [...asArray(first.request_nodes), ...asArray(first.structured_request_nodes), ...asArray(first.nodes)];
  first.request_nodes = [];
  first.structured_request_nodes = [];
  first.nodes = [];

  const summaryPos = reqNodes.findIndex((n) => normalizeNodeType(n) === REQUEST_NODE_HISTORY_SUMMARY && pick(n, ["history_summary_node", "historySummaryNode"]) != null);
  if (summaryPos < 0) { first.request_nodes = reqNodes; return; }

  const summaryNode = asRecord(reqNodes[summaryPos]);
  const summaryId = Number(pick(summaryNode, ["id"])) || 0;
  const summaryValue = pick(summaryNode, ["history_summary_node", "historySummaryNode"]);

  const toolResults = reqNodes.filter((n) => normalizeNodeType(n) === REQUEST_NODE_TOOL_RESULT && pick(n, ["tool_result_node", "toolResultNode"]) != null);
  const otherNodes = reqNodes.filter((n) => {
    const t = normalizeNodeType(n);
    return t !== REQUEST_NODE_HISTORY_SUMMARY && t !== REQUEST_NODE_TOOL_RESULT;
  });

  const text = renderHistorySummaryNodeValue(summaryValue, toolResults);
  if (!text) { first.request_nodes = otherNodes; return; }

  const summaryTextNode = { id: summaryId, type: REQUEST_NODE_TEXT, content: "", text_node: { content: asString(text) } };
  first.request_nodes = [summaryTextNode, ...otherNodes];
}

module.exports = { compactAugmentChatHistory, renderHistorySummaryNodeValue };

