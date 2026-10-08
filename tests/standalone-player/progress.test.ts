import { describe, expect, it } from 'vitest';
import { parseStudyProgress } from '@/lib/standalone-player/progress';
const progress = {
  formatVersion: 1,
  courseId: 'course-1',
  sceneIndex: 1,
  quizzes: { '1': { answers: { q1: ['A', 'C'], q2: 'some answer' }, submitted: true } },
};
describe('portable study progress', () => {
  it('round-trips scene position and quiz answers', () => {
    expect(parseStudyProgress(JSON.stringify(progress), 'course-1', 3)).toEqual(progress);
  });
  it('refuses a progress file for another course', () => {
    expect(() => parseStudyProgress(JSON.stringify(progress), 'other', 3)).toThrow(
      'does not match',
    );
  });
  it('refuses invalid scene positions and quiz answer shapes', () => {
    expect(() =>
      parseStudyProgress(JSON.stringify({ ...progress, sceneIndex: 20 }), 'course-1', 3),
    ).toThrow();
    expect(() =>
      parseStudyProgress(
        JSON.stringify({ ...progress, quizzes: { '1': { answers: { q1: {} }, submitted: true } } }),
        'course-1',
        3,
      ),
    ).toThrow();
  });
});
