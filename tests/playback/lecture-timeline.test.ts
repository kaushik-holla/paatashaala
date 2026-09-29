import { describe, expect, it } from 'vitest';
import type { Action } from '@/lib/types/action';
import type { Scene } from '@/lib/types/stage';
import { estimateSpeechDurationMs } from '@/lib/choreography';
import { buildLectureTimeline, formatLectureTime } from '@/lib/playback/lecture-timeline';

const speech = (text: string) => ({ id: text, type: 'speech', text }) as Action;
const scene = (actions: Action[]) => ({ actions }) as Scene;

describe('lecture timeline', () => {
  it('includes every course page, including pages without narration', () => {
    const firstText = 'A short introduction.';
    const firstDuration = Math.max(2000, estimateSpeechDurationMs(firstText));
    const timeline = buildLectureTimeline([
      scene([speech(firstText)]),
      scene([]),
      scene([speech('The next section.')]),
    ]);

    expect(timeline.starts).toEqual([
      0,
      Math.round(firstDuration / 1000),
      Math.round((firstDuration + 2000) / 1000),
    ]);
    expect(timeline.durationSeconds).toBeGreaterThan(timeline.starts[2]);
    expect(formatLectureTime(125)).toBe('2:05');
  });

  it('handles an empty course', () => {
    expect(buildLectureTimeline([])).toEqual({ starts: [], durationSeconds: 0 });
  });
});
