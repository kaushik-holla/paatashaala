import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
let route: typeof import('@/app/api/course-library/route');
let root: string;
beforeAll(async () => {
  root = await mkdtemp(path.join(tmpdir(), 'paatashaala-route-'));
  vi.stubEnv('PAATASHAALA_MEDIA_DIR', root);
  route = await import('@/app/api/course-library/route');
});
afterAll(async () => {
  vi.unstubAllEnvs();
  await rm(root, { recursive: true, force: true });
});
const url = 'http://localhost:3000/api/course-library';
function post(operation: string, id?: string, payload?: unknown) {
  return route.POST(
    new Request(url, {
      method: 'POST',
      headers: { origin: 'http://localhost:3000', 'content-type': 'application/json' },
      body: JSON.stringify({ operation, id, payload }),
    }),
  );
}
describe('localhost disk API', () => {
  it('rejects cross-origin writes and remote host access', async () => {
    const cross = await route.POST(
      new Request(url, {
        method: 'POST',
        headers: { origin: 'https://attacker.test' },
        body: JSON.stringify({ operation: 'list' }),
      }),
    );
    expect(cross.status).toBe(403);
    expect((await route.GET(new Request('https://public.test/api/course-library'))).status).toBe(
      403,
    );
  });
  it('accepts a browser-facing loopback Host when Next normalizes the request URL', async () => {
    const result = await route.POST(
      new Request(url, {
        method: 'POST',
        headers: { host: '127.0.0.1:3000', origin: 'http://127.0.0.1:3000' },
        body: JSON.stringify({ operation: 'list' }),
      }),
    );
    expect(result.status).toBe(200);
  });
  it('round-trips a course and binary asset without an owner cookie', async () => {
    const put = await route.PUT(
      new Request(url + '?asset=ast_test', {
        method: 'PUT',
        headers: { 'content-type': 'audio/wav' },
        body: new Uint8Array([1, 2, 3]),
      }),
    );
    expect(put.status).toBe(200);
    const bytes = await route.GET(new Request(url + '?asset=ast_test'));
    expect(bytes.headers.get('content-type')).toBe('audio/wav');
    expect([...new Uint8Array(await bytes.arrayBuffer())]).toEqual([1, 2, 3]);
    const doc = {
      stage: { id: 'course-1', name: 'My course', createdAt: 100, updatedAt: 200 },
      scenes: [],
    };
    expect((await post('save', 'course-1', doc)).status).toBe(200);
    expect(await (await post('load', 'course-1')).json()).toMatchObject({
      result: { stage: { name: 'My course' } },
    });
    expect((await post('delete', 'course-1')).status).toBe(200);
    expect(await (await post('list')).json()).toMatchObject({ result: [] });
    expect((await post('restore', 'course-1')).status).toBe(200);
    expect(await (await post('list')).json()).toMatchObject({ result: [{ id: 'course-1' }] });
  });
  it('rejects malformed documents and traversal IDs', async () => {
    expect((await post('save', 'bad', {})).status).toBe(400);
    expect((await post('load', '../secret')).status).toBe(400);
  });
});
