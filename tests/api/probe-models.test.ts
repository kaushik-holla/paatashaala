import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import type { NextRequest } from 'next/server';

const mocks = vi.hoisted(() => ({
  validateUrlForSSRF: vi.fn(),
  isServerConfiguredProvider: vi.fn(),
  resolveApiKey: vi.fn(),
  resolveBaseUrl: vi.fn(),
}));

vi.mock('@/lib/server/ssrf-guard', () => ({
  validateUrlForSSRF: mocks.validateUrlForSSRF,
}));

vi.mock('@/lib/server/provider-config', () => ({
  isServerConfiguredProvider: mocks.isServerConfiguredProvider,
  resolveApiKey: mocks.resolveApiKey,
  resolveBaseUrl: mocks.resolveBaseUrl,
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  }),
}));

async function postProbeModels(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/provider/probe-models/route');
  const request = new Request('http://localhost/api/provider/probe-models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return POST(request as unknown as NextRequest);
}

describe('POST /api/provider/probe-models', () => {
  beforeEach(() => {
    vi.resetModules();
    mocks.validateUrlForSSRF.mockReset();
    mocks.validateUrlForSSRF.mockResolvedValue(null);
    mocks.isServerConfiguredProvider.mockReset();
    mocks.isServerConfiguredProvider.mockReturnValue(false);
    mocks.resolveApiKey.mockReset();
    mocks.resolveApiKey.mockImplementation((_id: string, key: string) => key);
    mocks.resolveBaseUrl.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps an upstream redirect to the exact redirect-not-allowed contract without reading it', async () => {
    const text = vi.fn().mockResolvedValue('redirect response body');
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 302,
      text,
    } as unknown as Response);
    vi.stubGlobal('fetch', fetchMock);

    const res = await postProbeModels({
      baseUrl: 'https://api.example.com',
      apiKey: 'test-key',
    });
    const json = await res.json();

    expect(res.status).toBe(403);
    expect(json).toEqual({
      success: false,
      errorCode: 'REDIRECT_NOT_ALLOWED',
      error: 'Redirects are not allowed',
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(text).not.toHaveBeenCalled();
  });

  it('preserves successful model filtering and response metadata', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            data: [
              { id: 'chat-model', owned_by: 'provider' },
              { id: 'text-embedding-3-small', owned_by: 'provider' },
            ],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    );

    const res = await postProbeModels({
      baseUrl: 'https://api.example.com',
      apiKey: 'test-key',
    });
    const json = await res.json();

    expect(res.status).toBe(200);
    expect(json).toEqual({
      success: true,
      models: [{ id: 'chat-model', ownedBy: 'provider' }],
      total: 2,
      filtered: 1,
    });
  });

  it('loads Claude models with Anthropic authentication', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ data: [{ id: 'claude-sonnet-5' }] }), { status: 200 }),
      );
    vi.stubGlobal('fetch', fetchMock);

    const res = await postProbeModels({
      providerId: 'anthropic',
      baseUrl: 'https://api.anthropic.com/v1',
      apiKey: 'anthropic-test',
    });

    expect(res.status).toBe(200);
    expect((await res.json()).models).toEqual([{ id: 'claude-sonnet-5' }]);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://api.anthropic.com/v1/models?limit=1000',
      expect.objectContaining({
        headers: { 'x-api-key': 'anthropic-test', 'anthropic-version': '2023-06-01' },
      }),
    );
  });

  it('loads only generative Gemini models with Google authentication', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          models: [
            { name: 'models/gemini-3.6-flash', supportedGenerationMethods: ['generateContent'] },
            { name: 'models/text-embedding-004', supportedGenerationMethods: ['embedContent'] },
          ],
        }),
        { status: 200 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const res = await postProbeModels({
      providerId: 'google',
      baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
      apiKey: 'google-test',
    });

    expect(res.status).toBe(200);
    expect((await res.json()).models).toEqual([{ id: 'gemini-3.6-flash' }]);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000',
      expect.objectContaining({ headers: { 'x-goog-api-key': 'google-test' } }),
    );
  });

  it('does not send a managed key to a caller supplied models URL', async () => {
    mocks.isServerConfiguredProvider.mockReturnValue(true);
    mocks.resolveApiKey.mockReturnValue('server-secret');
    mocks.resolveBaseUrl.mockReturnValue(undefined);
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ data: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await postProbeModels({
      providerId: 'openai',
      baseUrl: 'https://attacker.example/v1',
      modelsUrl: 'https://attacker.example/models',
    });

    expect(res.status).toBe(200);
    expect(fetchMock.mock.calls[0][0]).toBe('https://api.openai.com/v1/models');
    expect(fetchMock.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer server-secret' });
  });

  it.each([401, 403])('preserves the API-key error contract for upstream %i', async (status) => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status,
        text: vi.fn().mockResolvedValue('invalid key'),
      } as unknown as Response),
    );

    const res = await postProbeModels({
      baseUrl: 'https://api.example.com',
      apiKey: 'bad-key',
    });
    const json = await res.json();

    expect(res.status).toBe(401);
    expect(json).toEqual({
      success: false,
      errorCode: 'INVALID_REQUEST',
      error: 'API key is invalid or expired',
    });
  });

  it('preserves the manual-entry response when no model endpoint exists', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        text: vi.fn(),
      } as unknown as Response),
    );

    const res = await postProbeModels({
      baseUrl: 'https://api.example.com',
      apiKey: 'test-key',
    });
    const json = await res.json();

    expect(res.status).toBe(404);
    expect(json).toEqual({
      success: false,
      errorCode: 'INVALID_REQUEST',
      error: 'This provider does not expose a model list',
    });
  });
});
