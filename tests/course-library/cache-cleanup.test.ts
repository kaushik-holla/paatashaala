import { describe, expect, it, vi } from 'vitest';
const mocks = vi.hoisted(() => ({
  migrate: vi.fn(async () => []),
  deleteDocument: vi.fn(),
  clearRuntime: vi.fn(async () => {}),
}));
vi.mock('@/lib/course-library/client', () => ({ isDiskLibraryEnabled: () => true }));
vi.mock('@/lib/utils/stage-storage', () => ({ listStages: mocks.migrate }));
vi.mock('@/lib/document-store/storage-generation', () => ({
  bumpGeneration: vi.fn(async () => {}),
}));
vi.mock('@/lib/document-store', () => ({
  getDocumentStore: () => ({
    listDocuments: async () => [{ id: 'disk-course' }],
    deleteDocument: mocks.deleteDocument,
  }),
}));
vi.mock('@/lib/runtime/store', () => ({
  getRuntimeStore: () => ({ deleteAllRuntime: mocks.clearRuntime }),
  beginStageRuntimeDeletionSafely: vi.fn(),
}));
vi.mock('@/lib/utils/chat-storage-lock', () => ({
  withRuntimeStorageExclusiveLock: (work: () => Promise<unknown>) => work(),
  withRuntimeStorageExclusiveLockUntilSettled: vi.fn(),
  withRuntimeStorageSharedLock: vi.fn(),
}));
vi.mock('@/lib/media/asset-pool', () => ({ clearAssetPool: vi.fn(async () => {}) }));
import { clearDatabase, db } from '@/lib/utils/database';
describe('disk tutorials survive cache clearing', () => {
  it('migrates remaining browser courses and clears browser state without deleting disk documents', async () => {
    const clearBrowser = vi.spyOn(db, 'delete').mockResolvedValue(undefined);
    await clearDatabase();
    expect(mocks.migrate).toHaveBeenCalledOnce();
    expect(mocks.clearRuntime).toHaveBeenCalledOnce();
    expect(clearBrowser).toHaveBeenCalledOnce();
    expect(mocks.deleteDocument).not.toHaveBeenCalled();
    clearBrowser.mockRestore();
  });
});
