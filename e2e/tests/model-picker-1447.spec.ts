import { test, expect } from '../fixtures/base';
import { createSettingsStorage } from '../fixtures/test-data/settings';

const serverProviders = {
  providers: {},
  tts: {},
  asr: {},
  pdf: {},
  image: {},
  video: {},
  webSearch: {},
};

async function openOpenAISettings(page: import('@playwright/test').Page) {
  await page.getByRole('button', { name: 'Settings' }).click();
  const dialog = page.getByRole('dialog', { name: 'Settings' });
  await dialog.getByRole('button', { name: /OpenAI OpenAI/ }).click();
  return dialog;
}

test.describe('model selection in Settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/server-providers', (route) =>
      route.fulfill({ json: serverProviders }),
    );
    await page.addInitScript(
      (settings) => localStorage.setItem('maic:account:settings-storage', settings),
      createSettingsStorage({
        modelId: 'alpha',
        providerId: 'openai',
        providersConfig: {
          openai: {
            apiKey: 'test-key',
            models: [
              { id: 'alpha', name: 'Alpha' },
              { id: 'beta', name: 'Beta' },
            ],
          },
        },
      }),
    );
  });

  test('chooses a default model and keeps it after reopening Settings', async ({ page }) => {
    await page.goto('/');
    let dialog = await openOpenAISettings(page);
    await dialog.getByRole('button', { name: 'Use Beta as the default model' }).click();
    await expect(
      dialog.getByRole('button', { name: 'Use Beta as the default model' }),
    ).toContainText('Default');
    await dialog.getByRole('button', { name: 'Close' }).click();

    dialog = await openOpenAISettings(page);
    await expect(
      dialog.getByRole('button', { name: 'Use Beta as the default model' }),
    ).toContainText('Default');
  });

  test('loads API-key models and offers them as defaults', async ({ page }) => {
    await page.route('**/api/provider/probe-models', (route) =>
      route.fulfill({ json: { success: true, models: [{ id: 'gpt-test-fetched' }] } }),
    );
    await page.goto('/');
    const dialog = await openOpenAISettings(page);
    await dialog.getByRole('button', { name: 'Fetch Models' }).click();
    await expect(dialog.getByText('gpt-test-fetched')).toBeVisible();
    const fetchedDefault = dialog.getByRole('button', {
      name: /Use .*gpt-test-fetched.* as the default model/i,
    });
    await expect(fetchedDefault).toBeEnabled();
    await fetchedDefault.click();
    await expect(fetchedDefault).toContainText('Default');
  });
});
