# OpenMAIC integration — October 8, 2026

Compared this independently branded fork with THU-MAIC/OpenMAIC main at
`32f59237`. The Git histories have no shared merge base, and this fork renames
`@openmaic/*` to `@paatashaala/*` and keeps a local, lecture-first workflow.
Changes were ported individually, preserving the existing provider selection,
branding, local setup, and browser-driven generation.

## Included

- `08e1447a` / #1813: shared export snapshot and single-file offline HTML player.
  Adapted package names and the export menu; extended the player with recorded
  narration/video and downloadable/importable study progress.
- `deb24913` / #1815: reject escaped CSS tokens at the sanitization boundary.
- `6bfdfa98` / #1713: reject duplicate quiz option values during generation.
- `877bf792` / #1706: preserve JavaScript template literals in interactive HTML.
- `e79d00c9` / #1816: bound user-authored skill text with a content-derived fence
  and neutralize closing tags that could escape its enclosing skill block.
- `8b451a33` / #1825: repaint animated SVG chart clipping on WebKit; offline
  charts remain static, and renderer tests cover Safari-specific behavior.

The new local disk library is specific to Paatashaala. It implements the
existing document/asset seams, keeps browser source copies during migration,
uses atomic JSON checkpoints and disk locks, and retains course-local media,
version history, and recoverable Trash. It does not require PostgreSQL.

## Deferred

The v1.2.0 server-first persistence/auth/configuration/generation migration is a
breaking platform change: it requires PostgreSQL and a long-running server,
removes the browser storage default, and changes ownership and model policy.
It should be a separate migration if this fork later needs multi-user hosting.

The broad provider-transport and MinerU Cloud hardening updates depend on a
shared transport refactor affecting dozens of provider routes and tests; they
were not partially cherry-picked into this storage/export change. That code
needs a separate focused compatibility pass before adopting the newer
transport as a whole. This integration is not a claim of complete upstream
security parity.

Preview/video worker isolation changes require render-service deployment and
configuration changes; they were left for a coordinated render-service update.
New provider catalogs, Token Plans, settings rewrites, and PPTX placeholder
choices were deferred to preserve this fork's deliberately narrowed setup and
existing course UI. The newer interactive-observation prompt patch did not
apply cleanly to this fork's prompt/snapshot baseline and was left unchanged.

## Verification

Focused document, migration, asset, export, player, sanitizer, and skill tests;
generation quiz/template-literal regressions; renderer clipping tests; TypeScript;
local browser verification of the built standalone player at phone dimensions.
The test preview uses synthetic course content, not private courses.
