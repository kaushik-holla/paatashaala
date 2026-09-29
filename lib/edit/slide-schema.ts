/**
 * SlideContent schema versioning. Slide-surface PRs will iterate the
 * on-disk shape; this module is the single chokepoint for normalizing
 * any incoming SlideContent (API response, snapshot restore, future
 * localStorage restore, PPTX reimport) to the current version.
 *
 * Conventions:
 *   - `migrateSlideContent` is pure (returns a new reference only when
 *     it has to change something) and idempotent (running it twice is
 *     identical to running it once).
 *   - Each schema bump appends a step keyed by the previous version's
 *     number. v1 (current) needs no per-step migration body — just the
 *     guarantee that the field is present.
 */

import {
  makeScene,
  type InteractiveContent,
  type Scene,
  type SceneContent,
  type SlideContent,
} from '@/lib/types/stage';

export const CURRENT_SLIDE_CONTENT_SCHEMA_VERSION = 1;

const LEGACY_ENGLISH_UI_REPLACEMENTS = [
  ['重新开始', 'Restart'],
  ['开始游戏', 'Start Game'],
  ['再试一次', 'Try Again'],
  ['隐藏控制', 'Hide Controls'],
  ['显示控制', 'Show Controls'],
  ['上一步', 'Previous'],
  ['下一步', 'Next'],
  ['运行中', 'Running'],
  ['已暂停', 'Paused'],
  ['已结束', 'Finished'],
  ['继续', 'Resume'],
  ['暂停', 'Pause'],
  ['启动', 'Start'],
  ['重试', 'Retry'],
  ['重置', 'Reset'],
  ['运行', 'Run'],
  ['停止', 'Stop'],
  ['放大', 'Zoom in'],
  ['缩小', 'Zoom out'],
  ['成功', 'Success'],
  ['失败', 'Failed'],
  ['得分', 'Score'],
  ['关卡', 'Level'],
  ['开始', 'Start'],
] as const;

/**
 * Repair a specific legacy generation defect: older English interactives were
 * seeded with Chinese control-label examples, so otherwise-English tutorials
 * could render buttons such as “下一步” and “启动”. Translate only the
 * small known UI vocabulary and only when the page's visible prose is clearly
 * English-dominant. This keeps Chinese-language lessons byte-identical.
 *
 * Replacements cover the full HTML source (including scripts) because some
 * legacy widgets compare button text while changing state. Updating only DOM
 * text would make those controls display correctly but stop working.
 */
export function translateLegacyInteractiveUiToEnglish(html: string): string {
  if (!LEGACY_ENGLISH_UI_REPLACEMENTS.some(([source]) => html.includes(source))) return html;

  const visibleText = html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ');
  const latinCount = visibleText.match(/[A-Za-z]/g)?.length ?? 0;
  const hanCount = visibleText.match(/[\u3400-\u9fff]/gu)?.length ?? 0;

  if (latinCount < 40 || latinCount < hanCount * 3) return html;

  return LEGACY_ENGLISH_UI_REPLACEMENTS.reduce(
    (result, [source, target]) => result.replaceAll(source, target),
    html,
  );
}

export function migrateSlideContent(content: SlideContent): SlideContent {
  // Forward-compatibility: if a future client has written content with a
  // newer schemaVersion than we know about, return it untouched rather
  // than silently downgrading. The slide may not render correctly here,
  // but its on-disk shape stays intact for the next compatible client.
  if (
    content.schemaVersion !== undefined &&
    content.schemaVersion >= CURRENT_SLIDE_CONTENT_SCHEMA_VERSION
  ) {
    return content;
  }
  // Legacy data (no schemaVersion) and any older intermediate versions
  // fall through here. As schema versions accumulate, walk versions in
  // order and apply each step's body before stamping the final version.
  return {
    ...content,
    schemaVersion: CURRENT_SLIDE_CONTENT_SCHEMA_VERSION,
  };
}

/**
 * InteractiveContent migration. The legacy widget-actions pipeline persisted a
 * `teacherActions` authoring layer alongside the materialized `actions` stream;
 * that field is now dead (playback reads only `scene.actions`). Legacy documents
 * carry it in two places — at the top level and nested inside `widgetConfig`
 * (every WidgetConfig variant used to declare it) — so drop both on load. The
 * same pure migration seam repairs known Chinese UI labels accidentally seeded
 * into English interactives. Both operations are idempotent and preserve the
 * original reference when no repair is needed, so no schema stamp is required.
 */
export function migrateInteractiveContent(content: InteractiveContent): InteractiveContent {
  const legacy = content as InteractiveContent & {
    teacherActions?: unknown;
    widgetConfig?: Record<string, unknown> & { teacherActions?: unknown };
  };
  const hasTop = 'teacherActions' in legacy;
  const hasNested = legacy.widgetConfig != null && 'teacherActions' in legacy.widgetConfig;
  const translatedHtml = legacy.html
    ? translateLegacyInteractiveUiToEnglish(legacy.html)
    : legacy.html;
  const hasTranslatedHtml = translatedHtml !== legacy.html;
  if (!hasTop && !hasNested && !hasTranslatedHtml) {
    return content;
  }
  const { teacherActions: _top, widgetConfig, ...rest } = legacy;
  const next = rest as InteractiveContent & { widgetConfig?: Record<string, unknown> };
  if (widgetConfig !== undefined) {
    if (hasNested) {
      const { teacherActions: _nested, ...widgetRest } = widgetConfig;
      next.widgetConfig = widgetRest;
    } else {
      next.widgetConfig = widgetConfig;
    }
  }
  if (hasTranslatedHtml) next.html = translatedHtml;
  return next;
}

/**
 * Top-level scene migrator — dispatches by scene-content type. SlideContent is
 * versioned; InteractiveContent drops its legacy `teacherActions` field and
 * repairs English UI labels; other content types pass through. Future surfaces
 * declare their own migrators and wire them in here.
 */
export function migrateScene(scene: Scene): Scene {
  const migratedContent = migrateSceneContent(scene.content);
  if (migratedContent === scene.content) {
    return scene;
  }
  return makeScene(scene, migratedContent);
}

function migrateSceneContent(content: SceneContent): SceneContent {
  if (content.type === 'slide') {
    return migrateSlideContent(content);
  }
  if (content.type === 'interactive') {
    return migrateInteractiveContent(content);
  }
  return content;
}
