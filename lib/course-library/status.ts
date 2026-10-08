export type CourseSaveStatus = 'idle' | 'saving' | 'saved' | 'failed';
let status: CourseSaveStatus = 'idle';
let pending = 0;
let failed = false;
const listeners = new Set<() => void>();
export const subscribeToCourseSave = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
export const courseSaveStatus = () => status;
export function beginCourseSave() {
  if (!pending) failed = false;
  pending++;
  status = 'saving';
  listeners.forEach((l) => l());
}
export function finishCourseSave(success: boolean) {
  pending = Math.max(0, pending - 1);
  failed ||= !success;
  status = pending ? 'saving' : failed ? 'failed' : 'saved';
  listeners.forEach((l) => l());
}
