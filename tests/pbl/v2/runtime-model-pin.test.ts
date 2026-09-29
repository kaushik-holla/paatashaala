import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { NextRequest } from 'next/server';

/**
 * Wiring guard: every PBL v2 *runtime* route must forward its MODEL_ROUTES
 * stage into model resolution, so operators can pin the shared runtime model
 * through `pbl-v2-runtime` or override a specific endpoint via
 * `pbl-v2-runtime:<endpoint>`.
 */
const mocks = vi.hoisted(() => ({
  resolveModelFromRequest: vi.fn(),
  runInstructorTurn: vi.fn(),
}));

vi.mock('@/lib/server/resolve-model', () => ({
  resolveModelFromRequest: mocks.resolveModelFromRequest,
}));
vi.mock('@/lib/pbl/v2/api/sse', () => ({
  createSSEResponse: vi.fn(() => new Response('ok')),
}));
vi.mock('@/lib/pbl/v2/api/locale', () => ({
  applyRequestLocaleToProject: vi.fn(),
}));
vi.mock('@/lib/pbl/v2/agents/instructor', () => ({
  runInstructorTurn: mocks.runInstructorTurn,
}));
vi.mock('@/lib/pbl/v2/agents/simulator', () => ({
  runSimulatorTurn: vi.fn(),
}));
vi.mock('@/lib/pbl/v2/agents/evaluator', () => ({
  runTaskEvaluation: vi.fn(),
  runMilestoneEvaluation: vi.fn(),
  runFinalEvaluation: vi.fn(),
}));
vi.mock('@/lib/pbl/v2/operations/runtime/quiz-snapshot', () => ({
  applyQuizSignalsToProject: vi.fn(() => ({ updated: false, tierChanged: false })),
}));
vi.mock('@/lib/logger', () => ({
  createLogger: () => ({ info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() }),
}));

function makeRequest(body: Record<string, unknown>): NextRequest {
  return new Request('http://localhost/api/pbl/v2', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }) as unknown as NextRequest;
}

describe('PBL v2 runtime routes forward MODEL_ROUTES stages', () => {
  beforeEach(() => {
    mocks.resolveModelFromRequest.mockReset();
    mocks.resolveModelFromRequest.mockResolvedValue({
      model: {},
      thinkingConfig: undefined,
      modelInfo: null,
    });
    mocks.runInstructorTurn.mockReset();
  });

  it('instructor route forwards its runtime stage', async () => {
    const { POST } = await import('@/app/api/pbl/v2/instructor/route');
    await POST(makeRequest({ project: { id: 'p' }, userMessage: 'hi' }));
    expect(mocks.resolveModelFromRequest).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      'pbl-v2-runtime:instructor',
    );
  });

  it('passes a screenshot to a vision-capable instructor model', async () => {
    mocks.resolveModelFromRequest.mockResolvedValue({
      model: {},
      thinkingConfig: undefined,
      modelInfo: { capabilities: { vision: true } },
    });
    const { POST } = await import('@/app/api/pbl/v2/instructor/route');
    const imageUrl = 'data:image/png;base64,AAAA';
    const response = await POST(
      makeRequest({ project: { id: 'p' }, userMessage: 'What is shown?', imageUrl }),
    );
    expect(response.status).toBe(200);
    expect(mocks.runInstructorTurn).toHaveBeenCalledWith(
      expect.objectContaining({ imageUrl, userMessage: 'What is shown?' }),
    );
  });

  it('rejects a screenshot when the selected instructor model lacks vision', async () => {
    const { POST } = await import('@/app/api/pbl/v2/instructor/route');
    const response = await POST(
      makeRequest({
        project: { id: 'p' },
        userMessage: 'What is shown?',
        imageUrl: 'data:image/png;base64,AAAA',
      }),
    );
    expect(response.status).toBe(400);
    expect(mocks.runInstructorTurn).not.toHaveBeenCalled();
  });

  it('evaluate route forwards its runtime stage', async () => {
    const { POST } = await import('@/app/api/pbl/v2/evaluate/route');
    await POST(makeRequest({ project: { id: 'p' }, kind: 'final' }));
    expect(mocks.resolveModelFromRequest).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      'pbl-v2-runtime:evaluate',
    );
  });

  it('open-task route forwards its runtime stage', async () => {
    const { POST } = await import('@/app/api/pbl/v2/open-task/route');
    await POST(makeRequest({ project: { id: 'p' }, phase: 'greeting' }));
    expect(mocks.resolveModelFromRequest).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      'pbl-v2-runtime:open-task',
    );
  });

  it('simulator route forwards its runtime stage', async () => {
    const { POST } = await import('@/app/api/pbl/v2/simulator/route');
    await POST(makeRequest({ project: { id: 'p' }, userMessage: 'hi' }));
    expect(mocks.resolveModelFromRequest).toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      'pbl-v2-runtime:simulator',
    );
  });
});
