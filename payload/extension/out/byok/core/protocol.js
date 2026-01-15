"use strict";

const { normalizeString, randomId } = require("../infra/util");
const { ensureModelRegistryFeatureFlags } = require("./model-registry");
const { buildChatPrompt } = require("../prompts/chat");
const { buildPromptEnhancerPrompt } = require("../prompts/prompt-enhancer");
const { buildCompletionPrompt } = require("../prompts/completion");
const { buildChatInputCompletionPrompt } = require("../prompts/chat-input-completion");
const { buildEditPrompt } = require("../prompts/edit");
const { buildInstructionStreamPrompt } = require("../prompts/instruction-stream");
const { buildSmartPasteStreamPrompt } = require("../prompts/smart-paste-stream");
const { buildCommitMessageStreamPrompt } = require("../prompts/commit-message-stream");
const { buildConversationTitlePrompt } = require("../prompts/conversation-title");
const { buildNextEditStreamPrompt } = require("../prompts/next-edit-stream");

function buildMessagesForEndpoint(endpoint, body) {
  const ep = normalizeString(endpoint);
  if (ep === "/completion") return buildCompletionPrompt(body);
  if (ep === "/chat-input-completion") return buildChatInputCompletionPrompt(body);
  if (ep === "/edit") return buildEditPrompt(body);
  if (ep === "/instruction-stream") return buildInstructionStreamPrompt(body);
  if (ep === "/smart-paste-stream") return buildSmartPasteStreamPrompt(body);
  if (ep === "/generate-commit-message-stream") return buildCommitMessageStreamPrompt(body);
  if (ep === "/generate-conversation-title") return buildConversationTitlePrompt(body);
  if (ep === "/next-edit-stream") return buildNextEditStreamPrompt(body);
  if (ep === "/prompt-enhancer") return buildPromptEnhancerPrompt(body);
  return buildChatPrompt(ep, body);
}

function coerceText(text) {
  return typeof text === "string" ? text : String(text ?? "");
}

function makeBackTextResult(text, extra) {
  return { text: coerceText(text), unknown_blob_names: [], checkpoint_not_found: false, ...(extra && typeof extra === "object" ? extra : null) };
}

function makeBackChatResult(text, { nodes, includeNodes = true } = {}) {
  const out = {
    text: coerceText(text),
    unknown_blob_names: [],
    checkpoint_not_found: false,
    workspace_file_chunks: []
  };
  if (includeNodes) out.nodes = Array.isArray(nodes) ? nodes : [];
  return out;
}

function makeBackCompletionResult(text, { timeoutMs } = {}) {
  const out = makeBackTextResult(text);
  if (Number.isFinite(Number(timeoutMs))) out.completion_timeout_ms = Number(timeoutMs);
  return out;
}

function makeBackNextEditGenerationChunk({ path, blobName, charStart, charEnd, existingCode, suggestedCode }) {
  const p = normalizeString(path) || "(unknown)";
  const b = normalizeString(blobName) || "(unknown)";
  const cs = Number.isFinite(Number(charStart)) ? Number(charStart) : 0;
  const ce = Number.isFinite(Number(charEnd)) && Number(charEnd) >= cs ? Number(charEnd) : cs;
  const ex = typeof existingCode === "string" ? existingCode : "";
  const su = typeof suggestedCode === "string" ? suggestedCode : "";
  return {
    unknown_blob_names: [],
    checkpoint_not_found: false,
    next_edit: {
      suggestion_id: `byok:${randomId()}`,
      path: p,
      blob_name: b,
      char_start: cs,
      char_end: ce,
      existing_code: ex,
      suggested_code: su,
      change_description: "BYOK suggestion",
      editing_score: 1,
      localization_score: 1,
      editing_score_threshold: 1
    }
  };
}

function makeBackNextEditLocationResult(candidate_locations) {
  return {
    candidate_locations: Array.isArray(candidate_locations) ? candidate_locations : [],
    unknown_blob_names: [],
    checkpoint_not_found: false,
    critical_errors: []
  };
}

function buildByokModelsFromConfig(cfg) {
  const out = new Set();
  const providers = Array.isArray(cfg?.providers) ? cfg.providers : [];
  for (const p of providers) {
    const pid = normalizeString(p?.id);
    if (!pid) continue;
    const models = Array.isArray(p?.models) ? p.models : [];
    for (const m of models) {
      const mid = normalizeString(m);
      if (mid) out.add(`byok:${pid}:${mid}`);
    }
    const dm = normalizeString(p?.defaultModel);
    if (dm) out.add(`byok:${pid}:${dm}`);
  }

  const rules = cfg?.routing?.rules && typeof cfg.routing.rules === "object" ? cfg.routing.rules : null;
  if (rules) {
    for (const r of Object.values(rules)) {
      if (!r || typeof r !== "object") continue;
      const pid = normalizeString(r.providerId);
      const mid = normalizeString(r.model);
      if (pid && mid) out.add(`byok:${pid}:${mid}`);
    }
  }
  return Array.from(out);
}

function makeBackGetModelsResult({ defaultModel, models }) {
  const dm = normalizeString(defaultModel) || (Array.isArray(models) && models.length ? models[0].name : "unknown");
  const ms = Array.isArray(models) ? models : [];
  const byokIds = ms.map((m) => normalizeString(m?.name)).filter(Boolean);
  return {
    default_model: dm,
    models: ms,
    feature_flags: ensureModelRegistryFeatureFlags({}, { byokModelIds: byokIds, defaultModel: dm })
  };
}

function makeModelInfo(name) {
  return { name, suggested_prefix_char_count: 0, suggested_suffix_char_count: 0, completion_timeout_ms: 120000 };
}

module.exports = {
  buildMessagesForEndpoint,
  makeBackTextResult,
  makeBackChatResult,
  makeBackCompletionResult,
  makeBackNextEditGenerationChunk,
  makeBackNextEditLocationResult,
  buildByokModelsFromConfig,
  makeBackGetModelsResult,
  makeModelInfo
};
