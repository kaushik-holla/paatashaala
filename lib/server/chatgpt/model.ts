import type { LanguageModel } from 'ai';
import {
  getModels,
  type AssistantMessage,
  type Context,
  type Message,
  type Model,
  type Tool,
} from '@earendil-works/pi-ai';
import { streamSimpleOpenAICodexResponses } from '@earendil-works/pi-ai/openai-codex-responses';

type ModelV3 = Extract<LanguageModel, { specificationVersion: 'v3' }>;
type CallOptions = Parameters<ModelV3['doGenerate']>[0];
type StreamPart =
  Awaited<ReturnType<ModelV3['doStream']>>['stream'] extends ReadableStream<infer T> ? T : never;

function toContext(options: CallOptions): Context {
  const system: string[] = [];
  const messages: Message[] = [];
  for (const item of options.prompt) {
    if (item.role === 'system') {
      system.push(item.content);
      continue;
    }
    if (item.role === 'tool') {
      for (const part of item.content) {
        if (part.type !== 'tool-result') continue;
        messages.push({
          role: 'toolResult',
          toolCallId: part.toolCallId,
          toolName: part.toolName,
          content: [
            {
              type: 'text',
              text: part.output.type === 'text' ? part.output.value : JSON.stringify(part.output),
            },
          ],
          isError: false,
          timestamp: Date.now(),
        });
      }
      continue;
    }
    if (item.role === 'user') {
      const content: Array<
        { type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }
      > = [];
      for (const part of item.content) {
        if (part.type === 'text') {
          content.push({ type: 'text', text: part.text });
          continue;
        }
        if (part.type === 'file' && part.mediaType.startsWith('image/')) {
          const data =
            part.data instanceof Uint8Array
              ? Buffer.from(part.data).toString('base64')
              : String(part.data);
          content.push({ type: 'image', data, mimeType: part.mediaType });
        }
      }
      messages.push({ role: 'user', content, timestamp: Date.now() });
      continue;
    }
    const content: AssistantMessage['content'] = [];
    for (const part of item.content) {
      if (part.type === 'text') content.push({ type: 'text', text: part.text });
      if (part.type === 'tool-call')
        content.push({
          type: 'toolCall',
          id: part.toolCallId,
          name: part.toolName,
          arguments: (part.input && typeof part.input === 'object' ? part.input : {}) as Record<
            string,
            unknown
          >,
        });
    }
    messages.push({
      role: 'assistant',
      content,
      api: 'openai-codex-responses',
      provider: 'openai-codex',
      model: '',
      timestamp: Date.now(),
      stopReason: 'stop',
      usage: emptyPiUsage(),
    });
  }
  if (options.responseFormat?.type === 'json') {
    system.push(
      `Respond with valid JSON only.${options.responseFormat.schema ? ` Match this JSON schema: ${JSON.stringify(options.responseFormat.schema)}` : ''}`,
    );
  }
  const tools: Tool[] = (options.tools ?? [])
    .filter((tool) => tool.type === 'function')
    .map((tool) => ({
      name: tool.name,
      description: tool.description ?? '',
      parameters: tool.inputSchema as Tool['parameters'],
    }));
  return { systemPrompt: system.join('\n\n'), messages, tools: tools.length ? tools : undefined };
}

function emptyPiUsage(): AssistantMessage['usage'] {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

function toUsage(message: AssistantMessage) {
  return {
    inputTokens: {
      total: message.usage.input,
      noCache: message.usage.input - message.usage.cacheRead,
      cacheRead: message.usage.cacheRead,
      cacheWrite: message.usage.cacheWrite,
    },
    outputTokens: { total: message.usage.output, text: message.usage.output, reasoning: undefined },
  };
}

function finishReason(message: AssistantMessage) {
  return {
    unified:
      message.stopReason === 'toolUse'
        ? ('tool-calls' as const)
        : message.stopReason === 'length'
          ? ('length' as const)
          : message.stopReason === 'error'
            ? ('error' as const)
            : ('stop' as const),
    raw: message.stopReason,
  };
}

export function createChatGPTLanguageModel(modelId: string, accessToken: string): LanguageModel {
  const catalog = getModels('openai-codex');
  const model = (catalog.find((entry) => entry.id === modelId) ?? {
    ...catalog.find((entry) => entry.id === 'gpt-5.4')!,
    id: modelId,
    name: modelId,
  }) as Model<'openai-codex-responses'>;
  if (!model || !/^[a-zA-Z0-9._-]{1,100}$/.test(modelId))
    throw new Error('Invalid ChatGPT model ID');
  const run = (options: CallOptions) =>
    streamSimpleOpenAICodexResponses(model, toContext(options), {
      apiKey: accessToken,
      signal: options.abortSignal,
      maxTokens: options.maxOutputTokens,
    });
  const implementation: ModelV3 = {
    specificationVersion: 'v3',
    provider: 'chatgpt',
    modelId,
    supportedUrls: {},
    async doGenerate(options) {
      let result: AssistantMessage | undefined;
      for await (const event of run(options)) {
        if (event.type === 'done') result = event.message;
        if (event.type === 'error')
          throw new Error(event.error.errorMessage ?? 'ChatGPT generation failed');
      }
      if (!result) throw new Error('ChatGPT returned no response');
      const content: Awaited<ReturnType<ModelV3['doGenerate']>>['content'] = [];
      for (const part of result.content) {
        if (part.type === 'text') content.push({ type: 'text', text: part.text });
        else if (part.type === 'thinking') content.push({ type: 'reasoning', text: part.thinking });
        else
          content.push({
            type: 'tool-call',
            toolCallId: part.id,
            toolName: part.name,
            input: JSON.stringify(part.arguments),
          });
      }
      return { content, finishReason: finishReason(result), usage: toUsage(result), warnings: [] };
    },
    async doStream(options) {
      const stream = new ReadableStream<StreamPart>({
        async start(controller) {
          controller.enqueue({ type: 'stream-start', warnings: [] });
          try {
            for await (const event of run(options)) {
              const id = String('contentIndex' in event ? event.contentIndex : 0);
              if (event.type === 'text_start') controller.enqueue({ type: 'text-start', id });
              if (event.type === 'text_delta')
                controller.enqueue({ type: 'text-delta', id, delta: event.delta });
              if (event.type === 'text_end') controller.enqueue({ type: 'text-end', id });
              if (event.type === 'thinking_start')
                controller.enqueue({ type: 'reasoning-start', id });
              if (event.type === 'thinking_delta')
                controller.enqueue({ type: 'reasoning-delta', id, delta: event.delta });
              if (event.type === 'thinking_end') controller.enqueue({ type: 'reasoning-end', id });
              if (event.type === 'toolcall_end')
                controller.enqueue({
                  type: 'tool-call',
                  toolCallId: event.toolCall.id,
                  toolName: event.toolCall.name,
                  input: JSON.stringify(event.toolCall.arguments),
                });
              if (event.type === 'done')
                controller.enqueue({
                  type: 'finish',
                  finishReason: finishReason(event.message),
                  usage: toUsage(event.message),
                });
              if (event.type === 'error')
                throw new Error(event.error.errorMessage ?? 'ChatGPT generation failed');
            }
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        },
      });
      return { stream };
    },
  };
  return implementation;
}
