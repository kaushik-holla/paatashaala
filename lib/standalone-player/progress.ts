export interface StudyProgress {
  formatVersion: 1;
  courseId: string;
  sceneIndex: number;
  quizzes: Record<string, { answers: Record<string, string | string[]>; submitted: boolean }>;
}

export function parseStudyProgress(
  text: string,
  courseId: string,
  sceneCount: number,
): StudyProgress {
  const value = JSON.parse(text) as StudyProgress;
  if (
    !value ||
    value.formatVersion !== 1 ||
    value.courseId !== courseId ||
    !Number.isInteger(value.sceneIndex) ||
    value.sceneIndex < 0 ||
    value.sceneIndex >= sceneCount ||
    !value.quizzes ||
    typeof value.quizzes !== 'object' ||
    Array.isArray(value.quizzes)
  )
    throw new Error('This progress file does not match the course.');
  for (const quiz of Object.values(value.quizzes)) {
    if (
      !quiz ||
      typeof quiz.submitted !== 'boolean' ||
      !quiz.answers ||
      typeof quiz.answers !== 'object' ||
      Array.isArray(quiz.answers) ||
      Object.values(quiz.answers).some(
        (a) =>
          typeof a !== 'string' && !(Array.isArray(a) && a.every((v) => typeof v === 'string')),
      )
    )
      throw new Error('Invalid quiz progress.');
  }
  return value;
}
export function downloadStudyProgress(progress: StudyProgress) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(progress, null, 2)], { type: 'application/json' }),
  );
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'study-progress.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
