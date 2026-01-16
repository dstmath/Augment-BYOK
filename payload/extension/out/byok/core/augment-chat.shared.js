"use strict";

const { normalizeString } = require("../infra/util");
const { RESPONSE_NODE_RAW_RESPONSE, RESPONSE_NODE_MAIN_TEXT_FINISHED, RESPONSE_NODE_TOOL_USE, RESPONSE_NODE_TOOL_USE_START, IMAGE_FORMAT_JPEG, IMAGE_FORMAT_GIF, IMAGE_FORMAT_WEBP } = require("./augment-protocol");
const { compactAugmentChatHistory } = require("./augment-history-summary");
const { asRecord, asArray, asString, pick, normalizeNodeType } = require("./augment-struct");
const {
  personaTypeToLabel,
  formatIdeStateForPrompt,
  formatEditEventsForPrompt,
  formatCheckpointRefForPrompt,
  formatChangePersonalityForPrompt,
  formatImageIdForPrompt,
  formatFileIdForPrompt,
  formatFileNodeForPrompt,
  formatHistorySummaryForPrompt
} = require("./augment-node-format");

function isPlaceholderMessage(message) {
  const s = String(message || "").trim();
  if (!s) return false;
  if (s.length > 16) return false;
  for (const ch of s) if (ch !== "-") return false;
  return true;
}

function mapImageFormatToMimeType(format) {
  const f = Number(format);
  if (f === IMAGE_FORMAT_JPEG) return "image/jpeg";
  if (f === IMAGE_FORMAT_GIF) return "image/gif";
  if (f === IMAGE_FORMAT_WEBP) return "image/webp";
  return "image/png";
}

function normalizeToolDefinitions(raw) {
  const list = asArray(raw);
  const out = [];
  for (const it of list) {
    const r = asRecord(it);
    const name = normalizeString(pick(r, ["name"]));
    if (!name) continue;
    const description = asString(pick(r, ["description"])) || "";
    const input_schema = pick(r, ["input_schema", "inputSchema"]);
    const input_schema_json = asString(pick(r, ["input_schema_json", "inputSchemaJson"])) || "";
    const mcp_server_name = asString(pick(r, ["mcp_server_name", "mcpServerName"])) || "";
    const mcp_tool_name = asString(pick(r, ["mcp_tool_name", "mcpToolName"])) || "";
    out.push({ name, description, input_schema: input_schema && typeof input_schema === "object" ? input_schema : null, input_schema_json, mcp_server_name, mcp_tool_name });
  }
  return out;
}

function resolveToolSchema(def) {
  if (def && def.input_schema && typeof def.input_schema === "object" && !Array.isArray(def.input_schema)) return def.input_schema;
  const raw = normalizeString(def && def.input_schema_json);
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
    } catch {}
  }
  return { type: "object", properties: {} };
}

function convertOpenAiTools(toolDefs) {
  const defs = normalizeToolDefinitions(toolDefs);
  return defs.map((d) => ({ type: "function", function: { name: d.name, ...(normalizeString(d.description) ? { description: d.description } : {}), parameters: resolveToolSchema(d) } }));
}

function convertAnthropicTools(toolDefs) {
  const defs = normalizeToolDefinitions(toolDefs);
  return defs.map((d) => ({ name: d.name, ...(normalizeString(d.description) ? { description: d.description } : {}), input_schema: resolveToolSchema(d) }));
}

function buildToolMetaByName(toolDefs) {
  const defs = normalizeToolDefinitions(toolDefs);
  const map = new Map();
  for (const d of defs) {
    const toolName = normalizeString(d.name);
    if (!toolName) continue;
    const mcpServerName = normalizeString(d.mcp_server_name);
    const mcpToolName = normalizeString(d.mcp_tool_name);
    if (!mcpServerName && !mcpToolName) continue;
    map.set(toolName, { mcpServerName: mcpServerName || undefined, mcpToolName: mcpToolName || undefined });
  }
  return map;
}

function normalizeChatHistoryItem(raw) {
  const r = asRecord(raw);
  const request_id = asString(pick(r, ["request_id", "requestId", "requestID", "id"]));
  const request_message = asString(pick(r, ["request_message", "requestMessage", "message"]));
  const response_text = asString(pick(r, ["response_text", "responseText", "response", "text"]));
  const request_nodes = asArray(pick(r, ["request_nodes", "requestNodes"]));
  const structured_request_nodes = asArray(pick(r, ["structured_request_nodes", "structuredRequestNodes"]));
  const nodes = asArray(pick(r, ["nodes"]));
  const response_nodes = asArray(pick(r, ["response_nodes", "responseNodes"]));
  const structured_output_nodes = asArray(pick(r, ["structured_output_nodes", "structuredOutputNodes"]));
  return { request_id, request_message, response_text, request_nodes, structured_request_nodes, nodes, response_nodes, structured_output_nodes };
}

function normalizeAugmentChatRequest(body) {
  const b = asRecord(body);
  const rawMessage = asString(pick(b, ["message"]));
  const rawPrompt = asString(pick(b, ["prompt"]));
  const rawInstruction = asString(pick(b, ["instruction"]));
  const useMessage = normalizeString(rawMessage) && !isPlaceholderMessage(rawMessage);
  const usePrompt = !useMessage && normalizeString(rawPrompt);
  const message = useMessage ? rawMessage : usePrompt ? rawPrompt : rawInstruction;
  const message_source = useMessage ? "message" : usePrompt ? "prompt" : normalizeString(rawInstruction) ? "instruction" : "";
  const conversation_id = asString(pick(b, ["conversation_id", "conversationId", "conversationID"]));
  const chat_history = asArray(pick(b, ["chat_history", "chatHistory"])).map(normalizeChatHistoryItem);
  compactAugmentChatHistory(chat_history);
  const blobs = asRecord(pick(b, ["blobs"]));
  const external_source_ids = asArray(pick(b, ["external_source_ids", "externalSourceIds"]));
  const user_guided_blobs = asArray(pick(b, ["user_guided_blobs", "userGuidedBlobs", "user_specified_files", "userSpecifiedFiles"]));
  const disable_auto_external_sources = Boolean(pick(b, ["disable_auto_external_sources", "disableAutoExternalSources"]));
  const disable_retrieval = Boolean(pick(b, ["disable_retrieval", "disableRetrieval"]));
  const context_code_exchange_request_id = asString(pick(b, ["context_code_exchange_request_id", "contextCodeExchangeRequestId"]));
  const tool_definitions = asArray(pick(b, ["tool_definitions", "toolDefinitions"]));
  const nodes = asArray(pick(b, ["nodes"]));
  const structured_request_nodes = asArray(pick(b, ["structured_request_nodes", "structuredRequestNodes"]));
  const request_nodes = asArray(pick(b, ["request_nodes", "requestNodes"]));
  let agent_memories = asString(pick(b, ["agent_memories", "agentMemories"]));
  if (!normalizeString(agent_memories)) {
    const mi = pick(b, ["memories_info", "memoriesInfo"]);
    if (typeof mi === "string") agent_memories = mi;
    else {
      const r = asRecord(mi);
      const direct = asString(pick(r, ["agent_memories", "agentMemories", "memories", "memory", "text", "content"]));
      if (normalizeString(direct)) agent_memories = direct;
      else {
        const arr = asArray(pick(r, ["items", "memories", "memory"]));
        const joined = arr.map((x) => normalizeString(String(x))).filter(Boolean).join("\n");
        if (normalizeString(joined)) agent_memories = joined;
      }
    }
  }
  const mode = asString(pick(b, ["mode"]));
  const prefix = asString(pick(b, ["prefix"]));
  const selected_code = asString(pick(b, ["selected_code", "selectedCode", "selected_text", "selectedText", "selected_code_snippet", "selectedCodeSnippet"]));
  const disable_selected_code_details = Boolean(pick(b, ["disable_selected_code_details", "disableSelectedCodeDetails"]));
  const suffix = asString(pick(b, ["suffix"]));
  const diff = asString(pick(b, ["diff"]));
  const lang = asString(pick(b, ["lang", "language"]));
  const path = asString(pick(b, ["path"]));
  const user_guidelines = asString(pick(b, ["user_guidelines", "userGuidelines"]));
  const workspace_guidelines = asString(pick(b, ["workspace_guidelines", "workspaceGuidelines"]));
  const persona_type = Number(pick(b, ["persona_type", "personaType"]));
  const silent = Boolean(pick(b, ["silent"]));
  const canvas_id = asString(pick(b, ["canvas_id", "canvasId"]));
  const request_id_override = asString(pick(b, ["request_id_override", "requestIdOverride"]));
  const rules = pick(b, ["rules"]);
  const feature_detection_flags = asRecord(pick(b, ["feature_detection_flags", "featureDetectionFlags"]));
  return { message, message_source, conversation_id, chat_history, blobs, external_source_ids, user_guided_blobs, disable_auto_external_sources, disable_retrieval, context_code_exchange_request_id, tool_definitions, nodes, structured_request_nodes, request_nodes, agent_memories, mode, prefix, selected_code, disable_selected_code_details, suffix, diff, lang, path, user_guidelines, workspace_guidelines, persona_type, silent, canvas_id, request_id_override, rules, feature_detection_flags };
}

function coerceRulesText(rules) {
  if (Array.isArray(rules)) return rules.map((x) => normalizeString(String(x))).filter(Boolean).join("\n");
  return normalizeString(rules);
}

function buildInlineCodeContextText(req) {
  if (req && typeof req === "object" && req.disable_selected_code_details === true) return "";
  const prefix = typeof req?.prefix === "string" ? req.prefix : "";
  const selected = typeof req?.selected_code === "string" ? req.selected_code : "";
  const suffix = typeof req?.suffix === "string" ? req.suffix : "";
  return `${prefix}${selected}${suffix}`.trim();
}

function buildUserExtraTextParts(req, { hasNodes } = {}) {
  if (hasNodes) return [];
  if (req && typeof req === "object" && req.message_source === "prompt") return [];
  if (req && typeof req === "object" && req.disable_selected_code_details === true) return [];
  const main = typeof req?.message === "string" ? req.message.trim() : "";
  const out = [];
  const code = buildInlineCodeContextText(req);
  if (normalizeString(code) && code.trim() !== main) out.push(code);
  const diff = typeof req?.diff === "string" ? req.diff.trim() : "";
  if (normalizeString(diff) && diff !== code && diff !== main) out.push(diff);
  return out;
}

function buildSystemPrompt(req) {
  const parts = [];
  const persona = personaTypeToLabel(req && typeof req === "object" ? req.persona_type : 0);
  if (persona && persona !== "DEFAULT") parts.push(`Persona: ${persona}`);
  if (normalizeString(req.user_guidelines)) parts.push(req.user_guidelines.trim());
  if (normalizeString(req.workspace_guidelines)) parts.push(req.workspace_guidelines.trim());
  const rulesText = coerceRulesText(req.rules);
  if (rulesText) parts.push(rulesText);
  if (normalizeString(req.agent_memories)) parts.push(req.agent_memories.trim());
  if (normalizeString(req.mode).toUpperCase() === "AGENT") parts.push("You are an AI coding assistant with access to tools. Use tools when needed to complete tasks.");
  if (normalizeString(req.lang)) parts.push(`The user is working with ${req.lang.trim()} code.`);
  if (normalizeString(req.path)) parts.push(`Current file path: ${req.path.trim()}`);
  return parts.join("\n\n").trim();
}

function extractAssistantTextFromOutputNodes(nodes) {
  const list = asArray(nodes);
  let finished = "";
  let raw = "";
  for (const n of list) {
    const r = asRecord(n);
    const t = normalizeNodeType(r);
    const content = asString(pick(r, ["content"]));
    if (t === RESPONSE_NODE_MAIN_TEXT_FINISHED && normalizeString(content)) finished = content;
    else if (t === RESPONSE_NODE_RAW_RESPONSE && content) raw += content;
  }
  return normalizeString(finished) ? finished.trim() : raw.trim();
}

function extractToolCallsFromOutputNodes(nodes) {
  const list = asArray(nodes);
  const toolUse = [];
  const toolUseStart = [];
  for (const n of list) {
    const r = asRecord(n);
    const t = normalizeNodeType(r);
    if (t === RESPONSE_NODE_TOOL_USE) toolUse.push(r);
    else if (t === RESPONSE_NODE_TOOL_USE_START) toolUseStart.push(r);
  }
  const chosen = toolUse.length ? toolUse : toolUseStart;
  const seen = new Set();
  const out = [];
  for (const n of chosen) {
    const tu = asRecord(pick(n, ["tool_use", "toolUse"]));
    const toolName = normalizeString(pick(tu, ["tool_name", "toolName"]));
    if (!toolName) continue;
    let id = normalizeString(pick(tu, ["tool_use_id", "toolUseId"]));
    if (!id) id = `tool-${out.length + 1}`;
    if (seen.has(id)) continue;
    seen.add(id);
    const args = normalizeString(pick(tu, ["input_json", "inputJson"])) || "{}";
    out.push({ id, type: "function", function: { name: toolName, arguments: args } });
  }
  return out;
}

function parseJsonObjectOrEmpty(json) {
  const raw = normalizeString(json) || "{}";
  try {
    const v = JSON.parse(raw);
    if (v && typeof v === "object" && !Array.isArray(v)) return v;
  } catch {}
  return {};
}

module.exports = {
  asRecord,
  asArray,
  asString,
  pick,
  isPlaceholderMessage,
  mapImageFormatToMimeType,
  normalizeToolDefinitions,
  resolveToolSchema,
  convertOpenAiTools,
  convertAnthropicTools,
  buildToolMetaByName,
  normalizeNodeType,
  normalizeAugmentChatRequest,
  coerceRulesText,
  buildInlineCodeContextText,
  buildUserExtraTextParts,
  formatIdeStateForPrompt,
  formatEditEventsForPrompt,
  formatCheckpointRefForPrompt,
  formatChangePersonalityForPrompt,
  formatImageIdForPrompt,
  formatFileIdForPrompt,
  formatFileNodeForPrompt,
  formatHistorySummaryForPrompt,
  buildSystemPrompt,
  extractAssistantTextFromOutputNodes,
  extractToolCallsFromOutputNodes,
  parseJsonObjectOrEmpty
};
