import { resolve } from 'path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      '@': resolve(__dirname, '.'),
    },
  },
  test: {
    include: ['tests/**/*.test.ts'],
    testTimeout: 10_000,
    setupFiles: ['tests/setup-env.ts'],
  },
});
