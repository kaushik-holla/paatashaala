'use client';

import { useMemo } from 'react';
import type { Scene } from '@/lib/types/stage';
import { buildLectureTimeline, formatLectureTime } from '@/lib/playback/lecture-timeline';

interface LectureTimelineProps {
  readonly scenes: readonly Scene[];
  readonly currentSceneIndex: number;
  readonly disabled: boolean;
  readonly onSeek: (sceneIndex: number) => void;
}

export function LectureTimeline({
  scenes,
  currentSceneIndex,
  disabled,
  onSeek,
}: LectureTimelineProps) {
  const { starts, durationSeconds } = useMemo(() => buildLectureTimeline(scenes), [scenes]);
  if (scenes.length < 2) return null;
  const index = Math.min(Math.max(0, currentSceneIndex), scenes.length - 1);

  return (
    <div className="flex w-full items-center gap-2 px-3 py-1 text-[10px] tabular-nums text-gray-500 dark:text-gray-400">
      <span className="shrink-0" title="Approximate elapsed narration time">
        ~{formatLectureTime(starts[index])}
      </span>
      <input
        aria-label="Seek course playback"
        title="Seek to a course page (times are approximate)"
        type="range"
        min={0}
        max={scenes.length - 1}
        step={1}
        value={index}
        disabled={disabled}
        onChange={(event) => onSeek(Number(event.target.value))}
        className="min-w-0 flex-1 cursor-pointer accent-[#456477] disabled:cursor-not-allowed disabled:opacity-40"
      />
      <span className="shrink-0" title="Approximate course duration">
        ~{formatLectureTime(durationSeconds)}
      </span>
    </div>
  );
}
