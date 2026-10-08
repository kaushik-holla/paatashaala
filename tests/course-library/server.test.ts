import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { DSL_VERSION } from '@paatashaala/dsl';
import { FileCourseLibrary, courseFolderName } from '@/lib/course-library/server';
import type { AppDocument } from '@/lib/document-store/persistence-types';

function document(name = 'Python / Basics'): AppDocument {
  return {
    stage: { id: 'course-1', name, createdAt: 100, updatedAt: 200 },
    scenes: [
      {
        id: 'scene-1',
        stageId: 'course-1',
        title: 'Lesson',
        order: 0,
        type: 'slide',
        content: { type: 'slide', canvas: { id: 'canvas-1', elements: [] } },
      },
    ],
  } as unknown as AppDocument;
}
let root: string;
let library: FileCourseLibrary;
beforeEach(async () => {
  root = await mkdtemp(path.join(tmpdir(), 'paatashaala-library-'));
  library = new FileCourseLibrary(root);
});
afterEach(async () => {
  vi.unstubAllEnvs();
  await rm(root, { recursive: true, force: true });
});

describe('durable course library', () => {
  it('keeps the default library beside the project when a standalone server changes cwd', async () => {
    vi.stubEnv('PAATASHAALA_PROJECT_DIR', root);
    vi.stubEnv('PAATASHAALA_MEDIA_DIR', '');
    const standalone = new FileCourseLibrary();
    await standalone.save(document());
    expect(await new FileCourseLibrary(path.join(root, 'media')).list()).toMatchObject([
      { id: 'course-1' },
    ]);
  });
  it('survives a fresh server instance and retains the outline', async () => {
    await library.save({ ...document(), outline: { outlines: [], generationComplete: false } });
    const next = new FileCourseLibrary(root);
    expect((await next.load('course-1'))?.outline).toEqual({
      outlines: [],
      generationComplete: false,
    });
    expect(await next.list()).toMatchObject([
      { id: 'course-1', name: 'Python / Basics', sceneCount: 1 },
    ]);
    expect(await readdir(path.join(root, 'courses'))).toEqual(['Python-Basics--course-1']);
  });
  it('rejects invalid identities and a malformed scene without replacing the last good save', async () => {
    await library.save(document());
    await expect(library.load('../secret')).rejects.toThrow('Invalid library ID');
    const bad = document();
    bad.scenes[0].stageId = 'wrong';
    await expect(library.save(bad)).rejects.toThrow('Invalid scene identity');
    expect((await library.load('course-1'))?.scenes[0].stageId).toBe('course-1');
    expect(await library.versions('course-1')).toHaveLength(1);
  });
  it('makes independent course-local asset copies and refuses unresolved pool refs', async () => {
    const doc = document();
    doc.scenes[0].content = {
      type: 'slide',
      canvas: {
        id: 'canvas-1',
        elements: [
          {
            type: 'image',
            id: 'image-1',
            src: 'ast_test',
            left: 0,
            top: 0,
            width: 100,
            height: 100,
            fixedRatio: true,
            rotate: 0,
          },
        ],
      },
    } as AppDocument['scenes'][number]['content'];
    await expect(library.save(doc)).rejects.toThrow('has not been saved');
    expect(await library.list()).toEqual([]);
    await library.putAsset('ast_test', new Uint8Array([1, 2, 3]), 'image/png');
    await library.save(doc);
    const folder = path.join(root, 'courses', courseFolderName(doc.stage.name, doc.stage.id));
    await rm(path.join(root, '.assets'), { recursive: true });
    expect([...(await readFile(path.join(folder, 'assets', 'ast_test.bin')))]).toEqual([1, 2, 3]);
    expect([...(await library.assetBytes('ast_test'))]).toEqual([1, 2, 3]);
  });
  it('moves whole courses to Trash and restores them without overwriting an existing course', async () => {
    await library.save(document());
    await library.remove('course-1');
    expect(await library.load('course-1')).toBeNull();
    expect(await library.list(true)).toMatchObject([{ id: 'course-1' }]);
    await expect(library.save(document())).rejects.toThrow('in Trash');
    await library.restore('course-1');
    expect((await library.load('course-1'))?.stage.name).toBe('Python / Basics');
    expect(await library.list(true)).toEqual([]);
  });
  it('retains version history across edits and restores a previous version as a new checkpoint', async () => {
    await library.save(document());
    const first = (await library.versions('course-1'))[0];
    await library.mutate('course-1', 'putStage', {
      ...document('Renamed course').stage,
      updatedAt: 300,
    });
    expect((await library.load('course-1'))?.stage.name).toBe('Renamed course');
    expect(await readdir(path.join(root, 'courses'))).toEqual(['Renamed-course--course-1']);
    await library.restoreVersion('course-1', first);
    expect((await library.load('course-1'))?.stage.name).toBe('Python / Basics');
    expect(await library.versions('course-1')).toHaveLength(3);
    expect(await readdir(path.join(root, 'courses'))).toEqual(['Python-Basics--course-1']);
  });
  it('serializes concurrent edits from different server instances', async () => {
    await library.save(document());
    const next = new FileCourseLibrary(root);
    const scene = { ...document().scenes[0], id: 'scene-2', order: 1 };
    await Promise.all([
      library.mutate('course-1', 'putStage', document('Updated').stage),
      next.mutate('course-1', 'putScene', scene),
    ]);
    const saved = await library.load('course-1');
    expect(saved?.stage.name).toBe('Updated');
    expect(saved?.scenes).toHaveLength(2);
  });
  it('guards newer-format documents already on disk against downgrade', async () => {
    await library.save(document());
    const file = path.join(
      root,
      'courses',
      courseFolderName(document().stage.name, 'course-1'),
      'tutorial.json',
    );
    await writeFile(file, JSON.stringify({ ...document(), dslVersion: '999.0.0' }));
    await expect(library.save({ ...document(), dslVersion: DSL_VERSION })).rejects.toThrow(
      'newer app version',
    );
    expect(JSON.parse(await readFile(file, 'utf8')).dslVersion).toBe('999.0.0');
  });
});
