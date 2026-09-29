import type { Page, Locator } from '@playwright/test';

export class HomePage {
  readonly page: Page;
  readonly logo: Locator;
  readonly textarea: Locator;
  readonly enterButton: Locator;

  constructor(page: Page) {
    this.page = page;
    this.logo = page.getByRole('button', { name: /Paatashaala LEARN YOUR WAY/i });
    this.textarea = page.getByRole('textbox', { name: 'What would you like to learn?' });
    this.enterButton = page.getByRole('button', { name: /Create my course/i });
  }

  async goto() {
    await this.page.goto('/');
  }

  async fillRequirement(text: string) {
    await this.textarea.fill(text);
  }

  async submit() {
    await this.enterButton.click();
  }
}
