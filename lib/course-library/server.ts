import { randomUUID } from 'node:crypto';
import {
  mkdir,
  readFile,
  readdir,
  rename,
  access,
  open,
  link,
  copyFile,
  rm,
} from 'node:fs/promises';
import path from 'node:path';
import { DSL_VERSION, migrate, needsMigration, dslVersionOf } from '@paatashaala/dsl';
import type { AppDocument } from '@/lib/document-store/persistence-types';
import { sanitizeSceneContent } from '@/lib/sanitize/scene-content';
import { validateAppScene, validateAppStage } from '@/lib/document-store/validators';

export class LibraryError extends Error {
  constructor(
    message: string,
    readonly status = 400,
  ) {
    super(message);
  }
}

export function libraryId(id: unknown): string {
  if (typeof id !== 'string' || !/^[a-zA-Z0-9_-]{1,160}$/.test(id))
    throw new LibraryError('Invalid library ID');
  return id;
}

export function courseFolderName(name: string, id: string): string {
  const slug =
    name
      .normalize('NFKC')
      .replace(/[^\p{L}\p{N}_-]+/gu, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 70) || 'course';
  return `${slug}--${libraryId(id)}`;
}

/** Also finds pool refs in app-owned fields outside the DSL's inventory. */
export function courseAssetIds(value: unknown): string[] {
  const refs = new Set<string>();
  function visit(v: unknown) {
    if (typeof v === 'string' && /^ast_[a-zA-Z0-9_-]+$/.test(v)) refs.add(v);
    else if (Array.isArray(v)) v.forEach(visit);
    else if (v && typeof v === 'object') Object.values(v).forEach(visit);
  }
  visit(value);
  return [...refs];
}

async function json<T>(file: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(file, 'utf8')) as T;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null;
    throw error;
  }
}

/** Temp + fsync + rename: a failed write leaves the previous complete snapshot. */
async function atomic(file: string, bytes: string | Uint8Array) {
  await mkdir(path.dirname(file), { recursive: true });
  const temp = `${file}.${randomUUID()}.tmp`;
  const handle = await open(temp, 'wx', 0o600);
  try {
    await handle.writeFile(bytes);
    await handle.sync();
  } finally {
    await handle.close();
  }
  try {
    await rename(temp, file);
  } finally {
    await rm(temp, { force: true });
  }
}

export class FileCourseLibrary {
  constructor(
    readonly root = path.resolve(
      process.env.PAATASHAALA_MEDIA_DIR ||
        path.join(process.env.PAATASHAALA_PROJECT_DIR || process.cwd(), 'media'),
    ),
  ) {}
  private get courses() {
    return path.join(this.root, 'courses');
  }
  private get trash() {
    return path.join(this.root, '.trash');
  }
  private get assets() {
    return path.join(this.root, '.assets');
  }

  // Disk locks serialize requests across Next workers, not just this module instance.
  async locked<T>(id: string, work: () => Promise<T>): Promise<T> {
    libraryId(id);
    const lock = path.join(this.root, '.locks', id);
    await mkdir(path.dirname(lock), { recursive: true });
    const deadline = Date.now() + 15000;
    for (;;) {
      try {
        await mkdir(lock);
        break;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
        if (Date.now() > deadline)
          throw new LibraryError(
            'Library is busy. Retry saving. If the server crashed, remove its stale media/.locks directory after stopping it.',
            503,
          );
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
    }
    try {
      return await work();
    } finally {
      await rm(lock, { recursive: true, force: true });
    }
  }

  private async folders(base = this.courses) {
    await mkdir(base, { recursive: true });
    return (await readdir(base, { withFileTypes: true }))
      .filter((e) => e.isDirectory() && !e.name.startsWith('.'))
      .map((e) => path.join(base, e.name));
  }
  private async find(id: string, base = this.courses) {
    libraryId(id);
    return (await this.folders(base)).find((folder) => path.basename(folder).endsWith(`--${id}`));
  }
  async load(id: string): Promise<AppDocument | null> {
    const folder = await this.find(id);
    const doc = folder ? await json<AppDocument>(path.join(folder, 'tutorial.json')) : null;
    if (!doc) return null;
    const { outline, ...core } = doc;
    return {
      ...(migrate(core) as AppDocument),
      ...(outline === undefined ? {} : { outline }),
    } as AppDocument;
  }
  async list(deleted = false) {
    const results = [];
    for (const folder of await this.folders(deleted ? this.trash : this.courses)) {
      const doc = await json<AppDocument>(path.join(folder, 'tutorial.json'));
      if (!doc) continue;
      results.push({
        id: doc.stage.id,
        name: doc.stage.name,
        description: doc.stage.description,
        createdAt: doc.stage.createdAt,
        updatedAt: doc.stage.updatedAt,
        sceneCount: doc.scenes.length,
        interactiveMode: doc.stage.interactiveMode,
        taskEngineMode: doc.stage.taskEngineMode,
      });
    }
    return results.sort((a, b) => b.updatedAt - a.updatedAt);
  }
  private validate(doc: AppDocument) {
    if (!doc || !doc.stage || !Array.isArray(doc.scenes))
      throw new LibraryError('Invalid course document');
    libraryId(doc.stage.id);
    if (doc.dslVersion && !needsMigration(doc) && dslVersionOf(doc) !== DSL_VERSION)
      throw new LibraryError('Course was written by a newer app version');
    const checks = [validateAppStage(doc.stage), ...doc.scenes.map(validateAppScene)];
    const invalid = checks.find((c) => !c.valid);
    if (invalid && !invalid.valid)
      throw new LibraryError(invalid.errors.map((e) => `${e.path}: ${e.message}`).join('; '));
    if (
      new Set(doc.scenes.map((s) => s.id)).size !== doc.scenes.length ||
      doc.scenes.some((s) => s.stageId !== doc.stage.id)
    )
      throw new LibraryError('Invalid scene identity');
  }
  private async saveUnlocked(doc: AppDocument) {
    this.validate(doc);
    const id = doc.stage.id;
    if (await this.find(id, this.trash))
      throw new LibraryError('Course is in Trash. Restore it before saving.', 409);
    const folder =
      (await this.find(id)) || path.join(this.courses, courseFolderName(doc.stage.name, id));
    await mkdir(path.join(folder, 'assets'), { recursive: true });
    // Keep course-local bytes usable even when pool assets change later.
    for (const ref of courseAssetIds(doc)) {
      const info = await this.assetInfo(ref);
      if (!info)
        throw new LibraryError(
          `Course asset ${ref} has not been saved. Retry after media finishes.`,
          409,
        );
      const destination = path.join(folder, 'assets', `${ref}.bin`);
      try {
        await link(await this.assetFile(ref), destination);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'EXDEV')
          await copyFile(await this.assetFile(ref), destination);
        else if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      }
      await atomic(path.join(folder, 'assets', `${ref}.json`), JSON.stringify(info));
    }
    const stored = { ...sanitizeSceneContent(doc), dslVersion: DSL_VERSION };
    const text = JSON.stringify(stored, null, 2);
    const current = await json<AppDocument>(path.join(folder, 'tutorial.json'));
    if (current?.dslVersion && !needsMigration(current) && dslVersionOf(current) !== DSL_VERSION)
      throw new LibraryError('Stored course was written by a newer app version');
    if (current && JSON.stringify(current) === JSON.stringify(stored)) return;
    // Immutable checkpoints are never pruned automatically.
    const version = `${Date.now()}-${randomUUID()}.json`;
    await atomic(path.join(folder, 'versions', version), text);
    await atomic(
      path.join(folder, 'manifest.json'),
      JSON.stringify(
        {
          formatVersion: 1,
          courseId: id,
          name: doc.stage.name,
          createdAt: doc.stage.createdAt,
          updatedAt: doc.stage.updatedAt,
          assets: courseAssetIds(doc),
          version,
        },
        null,
        2,
      ),
    );
    await atomic(path.join(folder, 'tutorial.json'), text);
    const namedFolder = path.join(this.courses, courseFolderName(doc.stage.name, id));
    if (namedFolder !== folder) await rename(folder, namedFolder);
  }
  async save(doc: AppDocument) {
    return this.locked(doc?.stage?.id, () => this.saveUnlocked(doc));
  }
  async mutate(id: string, operation: string, payload: unknown) {
    return this.locked(id, async () => {
      const doc = await this.load(id);
      if (!doc) throw new LibraryError('Course not found', 404);
      if (doc.dslVersion !== DSL_VERSION)
        throw new LibraryError('Load and save the course before editing');
      if (operation === 'putStage') {
        const stage = payload as AppDocument['stage'];
        if (stage.id !== id) throw new LibraryError('Course ID mismatch');
        doc.stage = stage;
      } else if (operation === 'putScene') {
        const scene = payload as AppDocument['scenes'][number];
        doc.scenes = [...doc.scenes.filter((s) => s.id !== scene.id), scene].sort(
          (a, b) => a.order - b.order,
        );
      } else if (operation === 'deleteScene')
        doc.scenes = doc.scenes.filter((s) => s.id !== payload);
      else throw new LibraryError('Unknown operation');
      await this.saveUnlocked(doc);
    });
  }
  async remove(id: string) {
    return this.locked(id, async () => {
      const folder = await this.find(id);
      if (!folder) return;
      await mkdir(this.trash, { recursive: true });
      await rename(folder, path.join(this.trash, path.basename(folder)));
    });
  }
  async restore(id: string) {
    return this.locked(id, async () => {
      const folder = await this.find(id, this.trash);
      if (!folder) throw new LibraryError('Course not found in Trash', 404);
      if (await this.find(id)) throw new LibraryError('Course already exists', 409);
      await mkdir(this.courses, { recursive: true });
      await rename(folder, path.join(this.courses, path.basename(folder)));
    });
  }
  async versions(id: string) {
    const folder = await this.find(id);
    if (!folder) return [];
    return (await readdir(path.join(folder, 'versions')))
      .filter((f) => /^\d+-[a-f0-9-]+\.json$/.test(f))
      .sort()
      .reverse();
  }
  async restoreVersion(id: string, version: string) {
    return this.locked(id, async () => {
      if (!(await this.versions(id)).includes(version))
        throw new LibraryError('Version not found', 404);
      const folder = (await this.find(id))!;
      const doc = await json<AppDocument>(path.join(folder, 'versions', version));
      if (!doc || doc.stage.id !== id) throw new LibraryError('Invalid version');
      await this.saveUnlocked({ ...doc, stage: { ...doc.stage, updatedAt: Date.now() } });
    });
  }
  private async assetFile(ref: string) {
    libraryId(ref);
    const pooled = path.join(this.assets, `${ref}.bin`);
    try {
      await access(pooled);
      return pooled;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    }
    for (const folder of [...(await this.folders()), ...(await this.folders(this.trash))]) {
      const file = path.join(folder, 'assets', `${ref}.bin`);
      try {
        await access(file);
        return file;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
    }
    throw new LibraryError('Course asset not found', 404);
  }
  async assetInfo(ref: string) {
    libraryId(ref);
    const pooled = await json<{ mime: string; size: number; meta?: unknown }>(
      path.join(this.assets, `${ref}.json`),
    );
    if (pooled) return pooled;
    try {
      const file = await this.assetFile(ref);
      return json<{ mime: string; size: number; meta?: unknown }>(file.replace(/\.bin$/, '.json'));
    } catch (error) {
      if (error instanceof LibraryError && error.status === 404) return null;
      throw error;
    }
  }
  async putAsset(ref: string, bytes: Uint8Array, mime: string, meta?: unknown) {
    libraryId(ref);
    return this.locked(ref, async () => {
      if (await this.assetInfo(ref)) return; // IDs are immutable; migration retries are idempotent.
      await atomic(path.join(this.assets, `${ref}.bin`), bytes);
      await atomic(
        path.join(this.assets, `${ref}.json`),
        JSON.stringify({ mime, size: bytes.length, meta }),
      );
    });
  }
  async assetBytes(ref: string) {
    return readFile(await this.assetFile(ref));
  }
}
