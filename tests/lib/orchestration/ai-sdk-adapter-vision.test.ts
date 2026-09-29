import { HumanMessage } from '@langchain/core/messages';
import { describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ streamLLM: vi.fn() }));

vi.mock('@/lib/ai/llm', () => ({
  callLLM: vi.fn(),
  streamLLM: mocks.streamLLM,
}));

vi.mock('@/lib/logger', () => ({
  createLogger: () => ({ info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() }),
}));

import { AISdkLangGraphAdapter } from '@/lib/orchestration/ai-sdk-adapter';

describe('AISdkLangGraphAdapter screenshot input', () => {
  it('attaches a pasted image to the latest learner message', async () => {
    mocks.streamLLM.mockReturnValue({
      textStream: (async function* () {
        yield 'ok';
      })(),
    });
    const adapter = new AISdkLangGraphAdapter({} as never);
    const chunks = [];
    for await (const chunk of adapter.streamGenerate([new HumanMessage('What is this?')], {
      visionImage: 'data:image/png;base64,AAAA',
    })) {
      chunks.push(chunk);
    }

    const request = mocks.streamLLM.mock.calls[0]?.[0] as {
      messages: Array<{ role: string; content: Array<Record<string, unknown>> }>;
    };
    expect(request.messages[0]?.role).toBe('user');
    expect(request.messages[0]?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'text', text: 'What is this?' }),
        expect.objectContaining({ type: 'image', image: 'AAAA', mimeType: 'image/png' }),
      ]),
    );
    expect(chunks).toEqual([
      { type: 'delta', content: 'ok' },
      { type: 'done', content: 'ok' },
    ]);
  });
});
