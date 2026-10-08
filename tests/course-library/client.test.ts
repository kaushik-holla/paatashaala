import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { IDBFactory, IDBKeyRange } from 'fake-indexeddb';
import { BrowserDocumentStore, BrowserAssetStore } from '@paatashaala/storage';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DiskAssetPool, DiskDocumentStore } from '@/lib/course-library/client';
import { FileCourseLibrary } from '@/lib/course-library/server';
import { validateAppScene, validateAppStage } from '@/lib/document-store/validators';
import type { AppDocument } from '@/lib/document-store/persistence-types';
import type { AppScene } from '@/lib/types/stage';
import type { AppStage } from '@/lib/document-store/persistence-types';
const validators = { validateScene: validateAppScene, validateStage: validateAppStage };
let root: string;
let server: FileCourseLibrary;
const nativeFetch = globalThis.fetch;
beforeEach(async () => {
  vi.stubGlobal('indexedDB', new IDBFactory());
  vi.stubGlobal('IDBKeyRange', IDBKeyRange);
  root = await mkdtemp(path.join(tmpdir(), 'paatashaala-client-'));
  server = new FileCourseLibrary(root);
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string, init?: RequestInit) => {
      if (input.startsWith('blob:')) return nativeFetch(input, init);
      const url = new URL(input, 'http://localhost:3000');
      const ref = url.searchParams.get('asset');
      if (ref) {
        if (init?.method === 'PUT') {
          const blob = init.body as Blob;
          await server.putAsset(ref, new Uint8Array(await blob.arrayBuffer()), blob.type);
          return Response.json({ saved: true });
        }
        return new Response(null, { status: (await server.assetInfo(ref)) ? 200 : 404 });
      }
      const { operation, id, payload } = JSON.parse(init!.body as string);
      let result: unknown;
      if (operation === 'save') await server.save(payload);
      else if (operation === 'load') result = await server.load(id);
      else if (operation === 'list') result = await server.list();
      else if (operation === 'trash') result = await server.list(true);
      else if (operation === 'delete') await server.remove(id);
      else await server.mutate(id, operation, payload);
      return Response.json({ result: result ?? null });
    }),
  );
});
afterEach(async () => {
  vi.unstubAllGlobals();
  await rm(root, { recursive: true, force: true });
});
function document(): AppDocument {
  return {
    stage: { id: 'course-1', name: 'Existing course', createdAt: 100, updatedAt: 200 },
    scenes: [],
  };
}
describe('browser-to-disk migration', () => {
  it('copies old documents and pool media without deleting browser copies', async () => {
    const browser = new BrowserDocumentStore<AppScene, AppStage>({
      dbName: 'maic-documents',
      ...validators,
    });
    const assets = new BrowserAssetStore({ dbName: 'maic-asset-pool' });
    const ref = await assets.put(new Blob(['recorded narration'], { type: 'audio/wav' }));
    const doc = { ...document(), outline: { preservedAsset: ref } };
    await browser.saveDocument(doc);
    const disk = new DiskDocumentStore(validators);
    expect(await disk.listDocuments()).toMatchObject([{ name: 'Existing course' }]);
    expect(await server.assetInfo(ref)).toMatchObject({ mime: 'audio/wav' });
    expect(await browser.loadDocument('course-1')).not.toBeNull();
    expect(await assets.exists(ref)).toBe(true);
    // A fresh browser profile, with no cookie, still sees the saved disk course.
    vi.stubGlobal('indexedDB', new IDBFactory());
    expect(await new DiskDocumentStore(validators).listDocuments()).toMatchObject([
      { id: 'course-1' },
    ]);
    await assets.close();
  });
  it('copies legacy narration bytes and rewrites the disk document to durable IDs', async () => {
    const { db } = await import('@/lib/utils/database');
    await db.audioFiles.put({
      id: 'legacy-audio',
      stageId: 'course-1',
      blob: new Blob(['narration'], { type: 'audio/wav' }),
      format: 'wav',
      createdAt: 100,
    } as never);
    const browser = new BrowserDocumentStore<AppScene, AppStage>({
      dbName: 'maic-documents',
      ...validators,
    });
    const doc = {
      ...document(),
      scenes: [
        {
          id: 'scene-1',
          stageId: 'course-1',
          title: 'Lesson',
          order: 0,
          type: 'slide',
          content: { type: 'slide', canvas: { id: 'canvas-1', elements: [] } },
          actions: [{ id: 'speech-1', type: 'speech', text: 'Hello', audioId: 'legacy-audio' }],
        },
      ],
    } as unknown as AppDocument;
    await browser.saveDocument(doc);
    await new DiskDocumentStore(validators).listDocuments();
    const saved = await server.load('course-1');
    const ref = (saved?.scenes[0].actions?.[0] as { audioId: string }).audioId;
    expect(ref).toMatch(/^ast_/);
    expect(await server.assetInfo(ref)).toMatchObject({ mime: 'audio/wav' });
    expect((await browser.loadDocument('course-1'))?.scenes[0].actions?.[0]).toMatchObject({
      audioId: 'legacy-audio',
    });
    db.close();
  });
  it('does not resurrect a trashed course from its retained browser copy', async () => {
    const browser = new BrowserDocumentStore<AppScene, AppStage>({
      dbName: 'maic-documents',
      ...validators,
    });
    await browser.saveDocument(document());
    const disk = new DiskDocumentStore(validators);
    await disk.listDocuments();
    await disk.deleteDocument('course-1');
    expect(await new DiskDocumentStore(validators).listDocuments()).toEqual([]);
  });
  it('stores a structural BinaryBlob and resolves it through the durable pool', async () => {
    const pool = new DiskAssetPool();
    const data = new Blob(['image bytes'], { type: 'image/png' });
    const ref = await pool.put({
      size: data.size,
      type: data.type,
      arrayBuffer: () => data.arrayBuffer(),
    });
    expect(await pool.resolve(ref)).toBe(`/api/course-library?asset=${ref}`);
    expect(new TextDecoder().decode(await server.assetBytes(ref))).toBe('image bytes');
    await pool.close();
  });
});
