import {
  BrowserDocumentStore,
  type DocumentStore,
  type DocumentSummary,
} from '@paatashaala/storage';
import type { AssetMeta, BinaryBlob } from '@paatashaala/dsl';
import type { AppScene } from '@/lib/types/stage';
import type { AppDocument, AppStage } from '@/lib/document-store/persistence-types';
import type { DocumentStorageValidators } from '@/lib/document-store/config';
import type { AssetPoolStore } from '@/lib/media/asset-pool-config';
import { beginCourseSave, finishCourseSave } from './status';
import { mayNameAPoolAsset } from '@/lib/media/media-placeholder';

export function isDiskLibraryEnabled() {
  if (
    typeof window === 'undefined' ||
    process.env.NODE_ENV === 'test' ||
    process.env.NEXT_PUBLIC_PERSISTENCE === '1'
  )
    return false;
  const mode = process.env.NEXT_PUBLIC_COURSE_STORAGE;
  return (
    mode !== 'browser' && ['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)
  );
}
export async function libraryRequest<T>(
  operation: string,
  id?: string,
  payload?: unknown,
): Promise<T> {
  const response = await fetch('/api/course-library', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ operation, id, payload }),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Course library request failed');
  return body.result;
}
async function saving<T>(work: () => Promise<T>): Promise<T> {
  beginCourseSave();
  try {
    const result = await work();
    finishCourseSave(true);
    return result;
  } catch (error) {
    finishCourseSave(false);
    throw error;
  }
}

export class DiskAssetPool implements AssetPoolStore {
  private legacy?: Promise<AssetPoolStore>;
  private legacySource() {
    return (this.legacy ??= import('@/lib/media/asset-pool').then(
      ({ createLegacyBrowserAssetPool }) => createLegacyBrowserAssetPool(),
    ));
  }
  private migrations = new Map<string, Promise<boolean>>();
  private address(ref: string) {
    return `/api/course-library?asset=${encodeURIComponent(ref)}`;
  }
  async put(data: BinaryBlob, meta?: AssetMeta) {
    const ref = `ast_${crypto.randomUUID().replaceAll('-', '')}`;
    await this.upload(
      ref,
      new Blob([await data.arrayBuffer()], { type: data.type || meta?.contentType || '' }),
    );
    return ref;
  }
  private async upload(ref: string, blob: Blob) {
    const response = await fetch(this.address(ref), {
      method: 'PUT',
      body: blob,
      headers: { 'content-type': blob.type || 'application/octet-stream' },
    });
    if (!response.ok)
      throw new Error((await response.json()).error || 'Could not save course media');
  }
  async exists(ref: string): Promise<boolean> {
    if (!mayNameAPoolAsset(ref) || !/^ast_[a-zA-Z0-9_-]+$/.test(ref)) return false;
    const response = await fetch(this.address(ref), { method: 'HEAD', cache: 'no-store' });
    if (response.ok) return true;
    if (response.status !== 404) throw new Error('Could not access saved course media');
    const prior = this.migrations.get(ref);
    if (prior) return prior;
    const work = (async () => {
      const { withAssetUrl } = await import('@/lib/media/use-asset-url');
      return withAssetUrl(
        ref,
        async (url) => {
          if (!url) return false;
          const response = await fetch(url);
          if (!response.ok) throw new Error('Could not read browser media');
          await this.upload(ref, await response.blob());
          return true;
        },
        await this.legacySource(),
      );
    })();
    this.migrations.set(ref, work);
    try {
      return await work;
    } finally {
      this.migrations.delete(ref);
    }
  }
  async resolve(ref: string) {
    return (await this.exists(ref)) ? this.address(ref) : null;
  }
  async invalidate() {}
  // Course versions can still reference these immutable bytes. Reclamation is explicit backup administration.
  async remove() {}
  async release() {}
  async close() {
    if (this.legacy) await (await this.legacy).close();
  }
}
function poolRefs(value: unknown): string[] {
  const refs = new Set<string>();
  function visit(v: unknown) {
    if (typeof v === 'string' && /^ast_[a-zA-Z0-9_-]+$/.test(v)) refs.add(v);
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  }
  visit(value);
  return [...refs];
}

/** Browser copies are retained until users have verified their disk backups. */
export class DiskDocumentStore implements DocumentStore<AppScene, AppStage> {
  private legacy: BrowserDocumentStore<AppScene, AppStage>;
  private assets = new DiskAssetPool();
  private migration?: Promise<void>;
  private legacyMappings = new Map<string, string>();
  private async durableDocument(doc: AppDocument) {
    const { materializeLegacyCourseAssets } = await import('./migrate-assets');
    const converted = await materializeLegacyCourseAssets(doc, this.assets, this.legacyMappings);
    await this.preserveAssets(converted);
    return converted;
  }
  constructor(validators: DocumentStorageValidators) {
    this.legacy = new BrowserDocumentStore({ dbName: 'maic-documents', ...validators });
  }
  private async preserveAssets(doc: AppDocument) {
    for (const ref of poolRefs(doc))
      if (!(await this.assets.exists(ref))) throw new Error(`Missing course media: ${ref}`);
  }
  private migrateLibrary() {
    return (this.migration ??= (async () => {
      const remote = await libraryRequest<DocumentSummary[]>('list');
      const deleted = await libraryRequest<DocumentSummary[]>('trash');
      const known = new Set([...remote, ...deleted].map((c) => c.id));
      for (const course of await this.legacy.listDocuments()) {
        if (known.has(course.id)) continue;
        const doc = await this.legacy.loadDocument(course.id);
        if (doc) {
          await libraryRequest('save', course.id, await this.durableDocument(doc));
        }
      }
    })().catch((error) => {
      this.migration = undefined;
      throw error;
    }));
  }
  async saveDocument(doc: AppDocument) {
    return saving(async () => {
      await libraryRequest('save', doc.stage.id, await this.durableDocument(doc));
    });
  }
  async loadDocument(id: string) {
    await this.migrateLibrary();
    return libraryRequest<AppDocument | null>('load', id);
  }
  async listDocuments() {
    await this.migrateLibrary();
    return libraryRequest<DocumentSummary[]>('list');
  }
  async deleteDocument(id: string) {
    return saving(() => libraryRequest<void>('delete', id));
  }
  async putStage(id: string, stage: AppStage) {
    return saving(async () => {
      const converted = await this.durableDocument({ stage, scenes: [] });
      await libraryRequest<void>('putStage', id, converted.stage);
    });
  }
  async putScene(id: string, scene: AppScene) {
    return saving(async () => {
      const converted = await this.durableDocument({ stage: { id } as AppStage, scenes: [scene] });
      await libraryRequest('putScene', id, converted.scenes[0]);
    });
  }
  async getScene(id: string, sceneId: string) {
    return (await this.loadDocument(id))?.scenes.find((s) => s.id === sceneId) ?? null;
  }
  async deleteScene(id: string, sceneId: string) {
    return saving(() => libraryRequest<void>('deleteScene', id, sceneId));
  }
}
