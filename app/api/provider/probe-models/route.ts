import { NextRequest } from 'next/server';
import { createLogger } from '@/lib/logger';
import { apiError, apiSuccess } from '@/lib/server/api-response';
import { validateUrlForSSRF } from '@/lib/server/ssrf-guard';
import { fetchModels, ModelFetchError } from '@/lib/server/model-fetch';
import {
  isServerConfiguredProvider,
  resolveApiKey,
  resolveBaseUrl,
} from '@/lib/server/provider-config';
import { PROVIDERS } from '@/lib/ai/providers';
import type { ProviderId } from '@/lib/types/provider';

const log = createLogger('ProbeModels');

/** Model ids that are not chat models — filtered out of probe results. */
const NON_CHAT_PATTERN = /(tts|asr|whisper|embedding|rerank|mineru|image|video|voxcpm|moderation)/i;

async function fetchNativeModels(providerId: string, baseUrl: string, apiKey: string) {
  const url =
    providerId === 'anthropic'
      ? `${baseUrl.replace(/\/+$/, '')}/models?limit=1000`
      : `${baseUrl.replace(/\/+$/, '').replace(/\/v1beta$/, '')}/v1beta/models?pageSize=1000`;
  const ssrfError = await validateUrlForSSRF(url);
  if (ssrfError) throw new Error(ssrfError);
  const response = await fetch(url, {
    headers:
      providerId === 'anthropic'
        ? { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' }
        : { 'x-goog-api-key': apiKey },
    redirect: 'manual',
    signal: AbortSignal.timeout(15_000),
  });
  if (response.status >= 300 && response.status < 400) {
    throw new ModelFetchError(response.status, 'Redirects are not allowed');
  }
  if (!response.ok) throw new ModelFetchError(response.status, `HTTP ${response.status}`);
  const body = await response.json();
  if (providerId === 'anthropic') {
    return (body.data as Array<{ id: string; ownedBy?: string }> | undefined) ?? [];
  }
  return (
    (body.models as Array<{ name: string; supportedGenerationMethods?: string[] }> | undefined) ??
    []
  )
    .filter((model) => model.supportedGenerationMethods?.includes('generateContent'))
    .map((model) => ({
      id: model.name.replace(/^models\//, ''),
      ownedBy: undefined as string | undefined,
    }));
}

/**
 * POST /api/provider/probe-models
 *
 * Discovers the chat models a base URL + key exposes, via the OpenAI-compatible
 * /models endpoint (with multi-candidate fallback). Returns the lit-up list, or
 * a typed status so the UI can fall back to manual model entry.
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { providerId, baseUrl, apiKey, modelsUrl } = body as {
      providerId?: string;
      baseUrl?: string;
      apiKey?: string;
      modelsUrl?: string;
    };

    const managed = !!providerId && isServerConfiguredProvider('providers', providerId);
    // Never let a caller redirect an operator-owned key to their own URL.
    const effectiveBaseUrl = managed
      ? resolveBaseUrl(providerId) || PROVIDERS[providerId as ProviderId]?.defaultBaseUrl
      : baseUrl;
    const effectiveModelsUrl = managed ? undefined : modelsUrl;
    if (!effectiveBaseUrl) {
      return apiError('MISSING_REQUIRED_FIELD', 400, 'baseUrl is required');
    }

    // SSRF guard on both the base URL and an explicit models URL override.
    for (const url of [effectiveBaseUrl, effectiveModelsUrl].filter(Boolean) as string[]) {
      const ssrfError = await validateUrlForSSRF(url);
      if (ssrfError) return apiError('INVALID_REQUEST', 400, ssrfError);
    }

    const effectiveKey = providerId ? resolveApiKey(providerId, apiKey || '') : apiKey || '';
    const models =
      !effectiveModelsUrl && (providerId === 'anthropic' || providerId === 'google')
        ? await fetchNativeModels(providerId, effectiveBaseUrl, effectiveKey)
        : await fetchModels(effectiveBaseUrl, effectiveKey, {
            modelsUrlOverride: effectiveModelsUrl,
          });
    const chatModels = models.filter((m) => !NON_CHAT_PATTERN.test(m.id));

    return apiSuccess({
      models: chatModels.map((m) => ({ id: m.id, ownedBy: m.ownedBy })),
      total: models.length,
      filtered: models.length - chatModels.length,
    });
  } catch (error) {
    if (error instanceof ModelFetchError) {
      if (error.status >= 300 && error.status < 400) {
        return apiError('REDIRECT_NOT_ALLOWED', 403, 'Redirects are not allowed');
      }
      if (error.status === 401 || error.status === 403) {
        return apiError('INVALID_REQUEST', 401, 'API key is invalid or expired');
      }
      if (error.status === 404) {
        // No /models endpoint — signal the UI (via 404) to use manual model entry.
        return apiError('INVALID_REQUEST', 404, 'This provider does not expose a model list');
      }
      return apiError('INTERNAL_ERROR', 502, error.message);
    }
    log.error('Model probe failed:', error);
    return apiError(
      'INTERNAL_ERROR',
      500,
      error instanceof Error ? error.message : 'Failed to probe models',
    );
  }
}
