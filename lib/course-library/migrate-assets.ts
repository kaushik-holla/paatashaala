import { enumerateAssetManifest } from '@paatashaala/dsl';
import type { AppDocument } from '@/lib/document-store/persistence-types';
import type { DiskAssetPool } from './client';
import { db, mediaFileKey } from '@/lib/utils/database';
import { fetchMediaUrl } from '@/lib/media/fetch-media-url';

/** Copy legacy browser bytes before claiming a document is durable. Pending media stays pending. */
export async function materializeLegacyCourseAssets(
  document: AppDocument,
  pool: DiskAssetPool,
  mappings: Map<string, string>,
): Promise<AppDocument> {
  const replacements = new Map<string, string>();
  for (const entry of enumerateAssetManifest(document).entries) {
    if (entry.ref.startsWith('ast_')) continue;
    const key = `${document.stage.id}:${entry.ref}`;
    const previous = mappings.get(key);
    if (previous) {
      replacements.set(entry.ref, previous);
      continue;
    }
    const row =
      entry.kind === 'audio'
        ? await db.audioFiles.get(entry.ref)
        : await db.mediaFiles.get(mediaFileKey(document.stage.id, entry.ref));
    let blob = row?.blob?.size ? row.blob : undefined;
    if (!blob && row?.ossKey) {
      const response = await fetchMediaUrl(row.ossKey, 15000);
      if (!response.ok) throw new Error('Could not copy course media to disk');
      blob = await response.blob();
    }
    // data/blob URLs are browser-owned bytes; remote links remain explicit dependencies.
    if (!blob && /^(?:data:|blob:)/i.test(entry.ref)) {
      const response = await fetch(entry.ref);
      if (response.ok) blob = await response.blob();
    }
    if (!blob?.size) continue;
    const ref = await pool.put(blob, { contentType: blob.type });
    mappings.set(key, ref);
    replacements.set(entry.ref, ref);
  }
  if (!replacements.size) return document;
  function rewrite(value: unknown): unknown {
    if (typeof value === 'string') return replacements.get(value) ?? value;
    if (Array.isArray(value)) return value.map(rewrite);
    if (value && typeof value === 'object')
      return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, rewrite(child)]));
    return value;
  }
  return rewrite(document) as AppDocument;
}
