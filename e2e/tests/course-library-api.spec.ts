import { randomUUID } from 'node:crypto';
import { test, expect } from '@playwright/test';

test('disk courses and media survive an independent client, with recoverable deletion', async ({
  request,
  playwright,
}, testInfo) => {
  const id = `disk_e2e_${randomUUID().replaceAll('-', '')}`;
  const asset = `ast_${randomUUID().replaceAll('-', '')}`;
  const bytes = Buffer.from('durable recorded media');
  const document = {
    stage: { id, name: 'Durable storage test', createdAt: Date.now(), updatedAt: Date.now() },
    scenes: [],
    outline: { recordedMedia: asset },
  };
  const uploaded = await request.put(`/api/course-library?asset=${asset}`, {
    data: bytes,
    headers: { 'content-type': 'audio/wav' },
  });
  expect(uploaded.ok()).toBeTruthy();
  const saved = await request.post('/api/course-library', {
    data: { operation: 'save', id, payload: document },
  });
  expect(saved.ok()).toBeTruthy();

  const independent = await playwright.request.newContext({
    baseURL: testInfo.project.use.baseURL,
  });
  try {
    const loaded = await independent.post('/api/course-library', {
      data: { operation: 'load', id },
    });
    expect(loaded.ok()).toBeTruthy();
    expect((await loaded.json()).result).toMatchObject(document);
    const media = await independent.get(`/api/course-library?asset=${asset}`);
    expect(await media.body()).toEqual(bytes);
    expect(media.headers()['x-content-type-options']).toBe('nosniff');

    const refused = await independent.post('/api/course-library', {
      headers: { origin: 'https://untrusted.example' },
      data: { operation: 'delete', id },
    });
    expect(refused.status()).toBe(403);
    expect(
      (
        await independent.post('/api/course-library', {
          data: { operation: 'delete', id },
        })
      ).ok(),
    ).toBeTruthy();
    expect(
      (
        await (
          await independent.post('/api/course-library', {
            data: { operation: 'load', id },
          })
        ).json()
      ).result,
    ).toBeNull();
    expect(
      (
        await independent.post('/api/course-library', {
          data: { operation: 'restore', id },
        })
      ).ok(),
    ).toBeTruthy();
    expect(
      (
        await (
          await independent.post('/api/course-library', {
            data: { operation: 'load', id },
          })
        ).json()
      ).result,
    ).toMatchObject(document);
  } finally {
    // The test owns only this unique course; production uses recoverable Trash.
    await request.post('/api/course-library', { data: { operation: 'delete', id } });
    await independent.dispose();
  }
});
