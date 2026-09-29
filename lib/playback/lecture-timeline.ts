import type { Scene } from '@/lib/types/stage';
import { estimateSpeechDurationMs } from '@/lib/choreography';

/** Approximate elapsed narration at each page boundary. */
export function buildLectureTimeline(scenes: readonly Scene[]): {
  starts: number[];
  durationSeconds: number;
} {
  let elapsedMs = 0;
  const starts = scenes.map((scene) => {
    const start = Math.round(elapsedMs / 1000);
    const speechMs = (scene.actions ?? []).reduce(
      (sum, action) => sum + (action.type === 'speech' ? estimateSpeechDurationMs(action.text) : 0),
      0,
    );
    elapsedMs += Math.max(2000, speechMs);
    return start;
  });
  return { starts, durationSeconds: Math.round(elapsedMs / 1000) };
}

export function formatLectureTime(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, '0')}`;
}
