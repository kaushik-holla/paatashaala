// @vitest-environment jsdom
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { assembleStandaloneHtml } from '@/lib/export/standalone-html/assemble';
import {
  prepareStandaloneManifest,
  collectStandaloneMediaReferences,
} from '@/lib/export/standalone-html/prepare-manifest';
import {
  standaloneFixtureStage,
  standaloneFixtureScenes,
  FIXTURE_PNG_BASE64,
} from '@/tests/fixtures/standalone-html-classroom';
import strings from '@/lib/i18n/locales/en-US.json';
import type { ClassroomManifest } from '@/lib/export/classroom-zip-types';
import type { StandalonePlayerStrings } from '@/lib/export/standalone-html/contract';

const bundle = path.resolve('public/vendor/standalone-player');
describe.skipIf(!existsSync(path.join(bundle, 'player.min.js')))('built offline player', () => {
  it('assembles the real bundle and a mixed course into one network-independent file', () => {
    const stage = standaloneFixtureStage('offline-preview');
    const manifest = {
      formatVersion: 1,
      exportedAt: new Date().toISOString(),
      appVersion: '0.1.0',
      stage,
      agents: [],
      scenes: standaloneFixtureScenes(stage.id),
      mediaIndex: {},
    } as unknown as ClassroomManifest;
    const dataUris = new Map(
      collectStandaloneMediaReferences(manifest).map(({ ref }) => [
        ref,
        `data:image/png;base64,${FIXTURE_PNG_BASE64}`,
      ]),
    );
    const prepared = prepareStandaloneManifest(manifest, { dataUris });
    const html = assembleStandaloneHtml({
      manifest: prepared.manifest,
      config: { strings: strings.export.htmlPlayer as StandalonePlayerStrings, courseId: stage.id },
      playerScript: readFileSync(path.join(bundle, 'player.min.js'), 'utf8'),
      playerStyle: readFileSync(path.join(bundle, 'player.min.css'), 'utf8'),
      extraScripts: [readFileSync(path.join(bundle, 'player-charts.min.js'), 'utf8')],
      extraStyles: [readFileSync(path.join(bundle, 'katex-fonts.min.css'), 'utf8')],
      lang: 'en-US',
    });
    expect(html).toContain("connect-src 'none'");
    expect(html).toContain('Save study progress');
    expect(html).toContain('data:image/png;base64,');
    expect(html).not.toMatch(/<script[^>]+src=/i);
    if (process.env.PAATASHAALA_PREVIEW_EXPORT)
      writeFileSync(process.env.PAATASHAALA_PREVIEW_EXPORT, html);
  });
});
