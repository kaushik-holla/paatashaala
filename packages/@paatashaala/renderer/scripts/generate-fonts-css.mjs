/** Generate local-only slide font aliases from fonts.config.mjs. */
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { FONT_FAMILIES } from '../fonts.config.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const outFile = path.join(here, '..', 'fonts.css');

const header = `/**
 * GENERATED FILE — do not edit by hand.
 * Source of truth: fonts.config.mjs (run \`pnpm run genfonts\` to regenerate).
 * These aliases use locally installed fonts only and make no network requests.
 */`;
const blocks = Object.entries(FONT_FAMILIES)
  .map(([family, localNames]) => {
    const sources = localNames.map((name) => `local('${name}')`).join(', ');
    return `@font-face {\n  font-display: swap;\n  font-family: '${family}';\n  src: ${sources};\n}`;
  })
  .join('\n');

writeFileSync(outFile, `${header}\n${blocks}\n`);
console.log(`[genfonts] wrote ${Object.keys(FONT_FAMILIES).length} local font aliases → fonts.css`);
