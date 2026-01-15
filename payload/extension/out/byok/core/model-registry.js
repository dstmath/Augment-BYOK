"use strict";

const { normalizeString, parseByokModelId } = require("../infra/util");

function safeParseJsonObject(raw) {
  try {
    const v = JSON.parse(typeof raw === "string" ? raw : "");
    return v && typeof v === "object" && !Array.isArray(v) ? v : {};
  } catch {
    return {};
  }
}

function pickFirstString(obj, keys) {
  const o = obj && typeof obj === "object" ? obj : {};
  for (const k of Array.isArray(keys) ? keys : []) {
    const v = o[k];
    if (typeof v === "string" && v) return v;
  }
  return "";
}

function ensureModelRegistryFeatureFlags(existingFlags, { byokModelIds, defaultModel, agentChatModel } = {}) {
  const dm = normalizeString(defaultModel) || "unknown";
  const flags =
    existingFlags && typeof existingFlags === "object" && !Array.isArray(existingFlags) ? { ...existingFlags } : {};

  const registry = safeParseJsonObject(pickFirstString(flags, ["model_registry", "modelRegistry"]));
  const infoRegistry = safeParseJsonObject(pickFirstString(flags, ["model_info_registry", "modelInfoRegistry"]));

  for (const raw of Array.isArray(byokModelIds) ? byokModelIds : []) {
    const parsed = parseByokModelId(raw);
    if (!parsed) continue;
    const displayName = `${parsed.providerId}: ${parsed.modelId}`;
    if (!registry[displayName]) registry[displayName] = raw;
    if (!infoRegistry[raw]) infoRegistry[raw] = { description: "", disabled: false, displayName, shortName: displayName };
  }

  const registryJson = JSON.stringify(registry);
  const infoRegistryJson = JSON.stringify(infoRegistry);
  const acm = normalizeString(agentChatModel) || pickFirstString(flags, ["agent_chat_model", "agentChatModel"]) || dm;

  flags.additional_chat_models = registryJson;
  flags.additionalChatModels = registryJson;
  flags.agent_chat_model = acm;
  flags.agentChatModel = acm;
  flags.enable_model_registry = true;
  flags.enableModelRegistry = true;
  flags.model_registry = registryJson;
  flags.modelRegistry = registryJson;
  flags.model_info_registry = infoRegistryJson;
  flags.modelInfoRegistry = infoRegistryJson;
  flags.show_thinking_summary = true;
  flags.showThinkingSummary = true;

  const fraudSign = typeof flags.fraud_sign_endpoints === "boolean" ? flags.fraud_sign_endpoints : typeof flags.fraudSignEndpoints === "boolean" ? flags.fraudSignEndpoints : false;
  flags.fraud_sign_endpoints = fraudSign;
  flags.fraudSignEndpoints = fraudSign;

  return flags;
}

module.exports = { ensureModelRegistryFeatureFlags };
