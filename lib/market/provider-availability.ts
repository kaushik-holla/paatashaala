/**
 * Providers offered by the Paatashaala setup UI for learners in the US and India.
 * Local runtimes are available everywhere they can be self-hosted. Custom
 * OpenAI-compatible connections remain an explicit advanced-user choice.
 *
 * Keep legacy adapters in the codebase for imported courses, but do not offer
 * region-specific services to new users or choose them as automatic fallbacks.
 */
export const AVAILABLE_PROVIDERS = {
  llm: new Set([
    'openai',
    'chatgpt',
    'azure',
    'anthropic',
    'bedrock',
    'google',
    'openrouter',
    'grok',
    'ollama',
    'lemonade',
  ]),
  image: new Set([
    'openai-image',
    'nano-banana',
    'grok-image',
    'openrouter-image',
    'comfyui-image',
    'lemonade',
  ]),
  video: new Set(['veo', 'grok-video', 'openrouter-video']),
  tts: new Set(['openai-tts', 'azure-tts', 'elevenlabs-tts', 'browser-native-tts', 'lemonade-tts']),
  asr: new Set(['openai-whisper', 'azure-asr', 'browser-native', 'lemonade-asr']),
  pdf: new Set(['unpdf']),
  webSearch: new Set(['tavily', 'exa', 'brave', 'claude', 'searxng']),
} as const;

export type ProviderCategory = keyof typeof AVAILABLE_PROVIDERS;

export function isAvailableProvider(category: ProviderCategory, providerId: string): boolean {
  if (providerId.startsWith('custom-')) return true;
  return (AVAILABLE_PROVIDERS[category] as ReadonlySet<string>).has(providerId);
}

const OPENROUTER_MODEL_PREFIXES = [
  'openai/',
  'anthropic/',
  'google/',
  'x-ai/',
  'meta-llama/',
  'mistralai/',
];

export function isAvailableLLMModel(providerId: string, modelId: string): boolean {
  if (providerId !== 'openrouter') return true;
  return OPENROUTER_MODEL_PREFIXES.some((prefix) => modelId.toLowerCase().startsWith(prefix));
}
