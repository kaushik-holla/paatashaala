import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, test } from 'vitest';

const css = readFileSync(resolve(process.cwd(), 'app/globals.css'), 'utf8');

describe('Paatashaala design system', () => {
  test.each([
    ['--bg-primary', '#f9f8f6'],
    ['--bg-surface', '#ffffff'],
    ['--text-primary', '#1a1a1a'],
    ['--text-secondary', '#626973'],
    ['--accent-primary', '#456477'],
    ['--accent-primary-soft', '#eaf0f3'],
    ['--accent-ai', '#8123d1'],
    ['--accent-secondary', '#a3b18a'],
    ['--accent-warning', '#ffb703'],
    ['--border-subtle', '#e5e0d8'],
  ])('defines %s as %s', (token, value) => {
    expect(css).toContain(`${token}: ${value};`);
  });

  test('loads and assigns the display and body typefaces', () => {
    expect(css).toContain('family=Lato:wght@400;700');
    expect(css).toContain('family=Playfair+Display:wght@600;700');
    expect(css).toContain("--font-body: 'Lato'");
    expect(css).toContain("--font-display: 'Playfair Display'");
  });
});
