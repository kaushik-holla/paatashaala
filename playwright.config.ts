import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e/tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // CI builds the production bundle in a dedicated workflow step, so this
  // process only drives Chromium. Two workers fit a 4-core runner.
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? 'html' : 'list',
  use: {
    baseURL: 'http://localhost:3002',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    // In CI the production build runs before Playwright. Start its standalone
    // server directly so package-manager shims cannot trigger an install.
    // The 120s budget covers startup, not the (much slower) build. Locally we
    // run the dev server.
    command: process.env.CI ? 'node scripts/start-standalone.mjs' : 'pnpm dev',
    url: 'http://localhost:3002',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    // Enable the MAIC Editor (Pro mode) so editor e2e can reach it. This is a
    // build-time NEXT_PUBLIC_* flag: in CI it must be set on the dedicated
    // `pnpm build` step; locally `pnpm dev` reads it here.
    env: {
      PORT: '3002',
      // Existing UI fixtures seed a separate browser database per test.
      // Durable server storage is exercised separately by course-library-api.spec.ts.
      NEXT_PUBLIC_COURSE_STORAGE: 'browser',
      NEXT_PUBLIC_MAIC_EDITOR_ENABLED: 'true',
      NEXT_PUBLIC_PI_CHAT_ENABLED: 'true',
      NEXT_PUBLIC_COURSEWARE_REFERENCE_ENABLED: 'true',
    },
  },
});
