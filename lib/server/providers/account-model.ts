import { getModelInfo } from '@/lib/ai/providers';
import type { ThinkingConfig } from '@/lib/types/provider';
import type { ResolvedModel } from '@/lib/server/resolve-model';
import { getChatGPTCredentials } from '@/lib/server/chatgpt/auth';
import { createChatGPTLanguageModel } from '@/lib/server/chatgpt/model';

/** Account-backed providers resolve here so ordinary model routing stays provider-neutral. */
export async function resolveAccountBackedModel(params: {
  providerId: string;
  modelId: string;
  modelString: string;
  thinkingConfig?: ThinkingConfig;
}): Promise<ResolvedModel | null> {
  if (params.providerId !== 'chatgpt') return null;
  const session = await getChatGPTCredentials();
  if (!session) throw new Error('Sign in with ChatGPT in Settings before using this model.');
  return {
    model: createChatGPTLanguageModel(params.modelId, session.credentials.access),
    modelInfo: getModelInfo('chatgpt', params.modelId) ?? null,
    modelString: params.modelString,
    providerId: params.providerId,
    modelId: params.modelId,
    apiKey: '',
    thinkingConfig: params.thinkingConfig,
  };
}
