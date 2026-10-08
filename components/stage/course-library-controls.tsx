'use client';

import { useSyncExternalStore } from 'react';
import Link from 'next/link';
import { isDiskLibraryEnabled } from '@/lib/course-library/client';
import { courseSaveStatus, subscribeToCourseSave } from '@/lib/course-library/status';

export function CourseLibraryControls() {
  const status = useSyncExternalStore(
    subscribeToCourseSave,
    courseSaveStatus,
    () => 'idle' as const,
  );
  const enabled = useSyncExternalStore(
    () => () => {},
    isDiskLibraryEnabled,
    () => false,
  );
  if (!enabled) return null;
  const label = {
    idle: 'Disk library',
    saving: 'Saving…',
    saved: 'Saved to disk',
    failed: 'Save failed',
  }[status];
  return (
    <Link
      href="/library"
      className={`px-2 text-xs ${status === 'failed' ? 'text-red-600' : 'text-gray-500'}`}
      title="Course backups, version history, and Trash"
      aria-live="polite"
    >
      {label}
    </Link>
  );
}
