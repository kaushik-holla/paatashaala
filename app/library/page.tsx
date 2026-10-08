'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { toast } from 'sonner';
import type { DocumentSummary } from '@paatashaala/storage';
import { libraryRequest } from '@/lib/course-library/client';
import { listStages } from '@/lib/utils/stage-storage';
import { unmarkStageDeleted } from '@/lib/utils/deleted-stages';

export default function LibraryPage() {
  const [courses, setCourses] = useState<DocumentSummary[]>([]);
  const [trash, setTrash] = useState<DocumentSummary[]>([]);
  const [versions, setVersions] = useState<{ id: string; entries: string[] } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    try {
      const [saved, deleted] = await Promise.all([
        listStages(),
        libraryRequest<DocumentSummary[]>('trash'),
      ]);
      setCourses(saved);
      setTrash(deleted);
      setError('');
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Could not open library');
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  async function action(work: () => Promise<unknown>) {
    setBusy(true);
    try {
      await work();
      await refresh();
      toast.success('Library updated');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Library action failed');
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="mx-auto max-w-4xl p-6 space-y-6">
      <Link href="/" className="text-purple-600">
        ← Home
      </Link>
      <h1 className="text-3xl font-semibold">Course library</h1>
      <p className="text-gray-500">
        Your local courses are saved on disk, independently of browser cookies. Back up the entire
        media folder to another drive or your cloud backup. Download a course ZIP for editable
        transfer, or offline HTML to study on your phone.
      </p>
      {error && (
        <div role="alert" className="text-red-600">
          {error}
          <button onClick={refresh} className="ml-4 underline">
            Retry
          </button>
        </div>
      )}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Saved courses</h2>
        {!courses.length && !error && (
          <p>
            No saved courses yet. Open this page in the browser that holds your existing courses to
            copy them to disk.
          </p>
        )}
        {courses.map((course) => (
          <div key={course.id} className="flex flex-wrap items-center gap-4 rounded-xl border p-4">
            <Link
              className="flex-1 min-w-48 font-medium"
              href={`/classroom/${encodeURIComponent(course.id)}`}
            >
              {course.name}
              <span className="block text-sm text-gray-500">{course.sceneCount} lessons</span>
            </Link>
            <button
              disabled={busy}
              className="underline"
              onClick={() =>
                action(async () =>
                  setVersions({
                    id: course.id,
                    entries: await libraryRequest<string[]>('versions', course.id),
                  }),
                )
              }
            >
              Version history
            </button>
            <button
              disabled={busy}
              className="text-red-600"
              onClick={() => action(() => libraryRequest('delete', course.id))}
            >
              Move to Trash
            </button>
          </div>
        ))}
      </section>
      {versions && (
        <section className="rounded-xl border p-4 space-y-3">
          <h2 className="text-xl font-semibold">Version history</h2>
          <button onClick={() => setVersions(null)} className="underline">
            Close history
          </button>
          {versions.entries.map((version) => (
            <div key={version} className="flex gap-4">
              <span className="flex-1">
                {new Date(Number(version.split('-')[0])).toLocaleString()}
              </span>
              <button
                disabled={busy}
                className="underline"
                onClick={() => action(() => libraryRequest('restoreVersion', versions.id, version))}
              >
                Restore this version
              </button>
            </div>
          ))}
        </section>
      )}
      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Trash</h2>
        <p className="text-sm text-gray-500">
          Deleted courses and their assets stay here until you remove them from disk yourself.
        </p>
        {trash.map((course) => (
          <div key={course.id} className="flex gap-4 rounded-xl border p-4">
            <span className="flex-1">{course.name}</span>
            <button
              disabled={busy}
              className="underline"
              onClick={() =>
                action(async () => {
                  await libraryRequest('restore', course.id);
                  unmarkStageDeleted(course.id);
                })
              }
            >
              Restore course
            </button>
          </div>
        ))}
      </section>
    </main>
  );
}
