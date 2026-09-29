import { describe, expect, it, vi } from 'vitest';
import { generateText, streamText } from 'ai';
import type { LanguageModel } from 'ai';
import { createChatGPTLanguageModel } from '@/lib/server/chatgpt/model';

vi.mock('@earendil-works/pi-ai/openai-codex-responses', () => ({
  streamSimpleOpenAICodexResponses: vi.fn((_model, context, options) =>
    (async function* () {
      expect(options.apiKey).toBe('user-specific-token');
      const usage = {
        input: 5,
        output: 2,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 7,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      };
      if (context.tools?.length) {
        const toolCall = {
          type: 'toolCall',
          id: 'call_1',
          name: context.tools[0].name,
          arguments: { city: 'Boston' },
        };
        yield { type: 'toolcall_end', contentIndex: 0, toolCall };
        yield {
          type: 'done',
          reason: 'toolUse',
          message: { role: 'assistant', content: [toolCall], stopReason: 'toolUse', usage },
        };
        return;
      }
      yield { type: 'text_start', contentIndex: 0 };
      yield { type: 'text_delta', contentIndex: 0, delta: 'Hello' };
      yield { type: 'text_end', contentIndex: 0, content: 'Hello' };
      yield {
        type: 'done',
        reason: 'stop',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'Hello' }],
          stopReason: 'stop',
          usage,
        },
      };
    })(),
  ),
}));

describe('ChatGPT model bridge', () => {
  it('supports non-streaming generation through the selected model', async () => {
    const result = await generateText({
      model: createChatGPTLanguageModel('gpt-5.4', 'user-specific-token'),
      prompt: 'Say hello',
    });
    expect(result.text).toBe('Hello');
    expect(result.usage.inputTokens).toBe(5);
  });

  it('streams text through the same signed-in account', async () => {
    const result = streamText({
      model: createChatGPTLanguageModel('gpt-5.4', 'user-specific-token'),
      prompt: 'Say hello',
    });
    let text = '';
    for await (const chunk of result.textStream) text += chunk;
    expect(text).toBe('Hello');
  });

  it('passes tools through the account-backed model', async () => {
    const model = createChatGPTLanguageModel('gpt-5.4', 'user-specific-token') as Extract<
      LanguageModel,
      { specificationVersion: 'v3' }
    >;
    const result = await model.doGenerate({
      prompt: [{ role: 'user', content: [{ type: 'text', text: 'Check weather' }] }],
      tools: [
        {
          type: 'function',
          name: 'weather',
          inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
        },
      ],
    });
    expect(result.content).toContainEqual({
      type: 'tool-call',
      toolCallId: 'call_1',
      toolName: 'weather',
      input: '{"city":"Boston"}',
    });
    expect(result.finishReason.unified).toBe('tool-calls');
  });
});
