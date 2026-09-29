import { test, expect } from '../fixtures/base';
import { HomePage } from '../pages/home.page';
import { createSettingsStorage } from '../fixtures/test-data/settings';

/**
 * #580 — "usable provider ⇒ a concrete model is always selected".
 *
 * State A: no usable provider → the home action guides the learner to AI
 *          setup without an unsolicited dialog or error toast.
 * State B: a server-configured provider → a concrete default model is
 *          selected in Settings and course generation is enabled.
 */

const SCREENSHOT_DIR = 'e2e/screenshots';

// fetchServerProviders reads data.tts/asr/pdf/image/video/webSearch via
// Object.keys(); omitting them throws and is silently swallowed by its
// try/catch, so the mock must return the full shape (like the unit helper).
function serverProvidersBody(providers: Record<string, { models?: string[] }>) {
  return JSON.stringify({
    providers,
    tts: {},
    asr: {},
    pdf: {},
    image: {},
    video: {},
    webSearch: {},
  });
}

// Run serially with one worker: a single shared dev server + async
// server-provider reconcile makes parallel runs flaky.
test.describe.configure({ mode: 'serial' });

test.describe('#580 model-selection invariant', () => {
  test('State A: no usable provider → clear setup action, no toast', async ({ page }) => {
    await page.route('**/api/server-providers', (route) =>
      route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: serverProvidersBody({}),
      }),
    );
    await page.addInitScript(
      (settings) => {
        localStorage.setItem('maic:account:settings-storage', settings);
      },
      createSettingsStorage({
        modelId: '',
        providerId: 'openai',
        providersConfig: { openai: { apiKey: '' } },
        autoConfigApplied: true,
      }),
    );

    const home = new HomePage(page);
    await Promise.all([page.waitForResponse('**/api/server-providers'), home.goto()]);
    await expect(home.textarea).toBeVisible();

    // The redesigned home keeps provider setup in Settings.
    await expect(page.getByRole('button', { name: 'Settings' })).toBeVisible();

    // After typing, the primary action guides the learner into setup without
    // an unsolicited modal or error toast.
    await home.fillRequirement('Explain how photosynthesis works');
    await expect(page.getByRole('button', { name: 'Set up your AI model' })).toBeVisible();
    await expect(page.locator('[data-sonner-toast]')).toHaveCount(0);
    await expect(page.getByRole('dialog')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/580-state-a-no-provider.png`,
      fullPage: true,
      animations: 'disabled',
      caret: 'hide',
    });
  });

  test('State B: server-configured provider → concrete model auto-selected, generation enabled', async ({
    page,
  }) => {
    await page.route('**/api/server-providers', (route) =>
      route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: serverProvidersBody({ openai: { models: ['gpt-4o', 'gpt-4o-mini'] } }),
      }),
    );
    await page.addInitScript(
      (settings) => {
        localStorage.setItem('maic:account:settings-storage', settings);
      },
      createSettingsStorage({
        modelId: '',
        providerId: 'openai',
        providersConfig: { openai: { apiKey: '' } },
        autoConfigApplied: true,
      }),
    );

    const home = new HomePage(page);
    await Promise.all([page.waitForResponse('**/api/server-providers'), home.goto()]);
    await expect(home.textarea).toBeVisible();

    // Reconcile resolves the first server model even though the home no longer
    // displays a model picker. The selected default is visible in Settings.
    await page.getByRole('button', { name: 'Settings' }).click();
    const settings = page.getByRole('dialog', { name: 'Settings' });
    await settings.getByRole('button', { name: /OpenAI OpenAI/ }).click();
    await expect(
      settings.getByRole('button', { name: 'Use gpt-4o as the default model' }),
    ).toContainText('Default');
    await settings.getByRole('button', { name: 'Close' }).click();

    await home.fillRequirement('Explain how photosynthesis works');
    await expect(home.enterButton).toBeEnabled();
    await expect(page.locator('[data-sonner-toast]')).toHaveCount(0);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/580-state-b-usable-provider.png`,
      fullPage: true,
      animations: 'disabled',
      caret: 'hide',
    });
  });
});
