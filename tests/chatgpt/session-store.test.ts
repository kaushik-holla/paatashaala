import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import {
  deleteChatGPTSession,
  readChatGPTSession,
  saveChatGPTSession,
} from '@/lib/server/chatgpt/session-store';

describe('ChatGPT session storage', () => {
  const previousSecret = process.env.CHATGPT_SESSION_SECRET;
  const previousDatabase = process.env.DATABASE_URL;
  const ids: string[] = [];

  beforeAll(() => {
    process.env.CHATGPT_SESSION_SECRET = 'test-only-secret-with-at-least-thirty-two-characters';
    delete process.env.DATABASE_URL;
  });

  afterAll(async () => {
    await Promise.all(ids.map(deleteChatGPTSession));
    if (previousSecret === undefined) delete process.env.CHATGPT_SESSION_SECRET;
    else process.env.CHATGPT_SESSION_SECRET = previousSecret;
    if (previousDatabase === undefined) delete process.env.DATABASE_URL;
    else process.env.DATABASE_URL = previousDatabase;
  });

  it('keeps each account separate and encrypts the stored tokens', async () => {
    const first = await saveChatGPTSession({
      access: 'first-access-secret',
      refresh: 'first-refresh-secret',
      expires: Date.now() + 60_000,
    });
    const second = await saveChatGPTSession({
      access: 'second-access-secret',
      refresh: 'second-refresh-secret',
      expires: Date.now() + 60_000,
    });
    ids.push(first, second);

    expect((await readChatGPTSession(first))?.access).toBe('first-access-secret');
    expect((await readChatGPTSession(second))?.access).toBe('second-access-secret');
    expect(await readChatGPTSession('invalid')).toBeNull();

    const filename = path.join(
      process.cwd(),
      '.data',
      'chatgpt-sessions',
      createHash('sha256').update(first).digest('hex'),
    );
    const disk = await readFile(filename, 'utf8');
    expect(disk).not.toContain('first-access-secret');
    expect(disk).not.toContain('first-refresh-secret');

    await deleteChatGPTSession(first);
    expect(await readChatGPTSession(first)).toBeNull();
    expect((await readChatGPTSession(second))?.access).toBe('second-access-secret');
  });
});
