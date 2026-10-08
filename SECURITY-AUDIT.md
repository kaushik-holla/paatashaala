# Course library and offline export security review

Review date: 2026-10-08. Baseline: public main at `a7e45f67`.

## Publication and private data

- Gitleaks scanned the complete public branch history with redaction enabled: zero findings.
- Generated courses, media, `.env.local`, and local logs are excluded from source control. Course migration copies user data into the ignored media directory; no course data is included in this change.
- Only the three feature commits are applied onto public main. The former archive repository history is not published.

## Storage and exports

- Disk routes require a loopback Host and same-origin requests; cross-site requests are rejected. The existing access-code gate still applies.
- Course and asset IDs are constrained before filesystem operations. Documents are validated and scene markup is sanitized.
- Stored assets use sandbox and nosniff response headers. Course writes use atomic replacement, and deleted courses remain recoverable.
- Offline exports contain course content and selected media, not application provider settings or credentials. JSON script payloads escape HTML delimiters; interactive content runs in an opaque sandbox. The offline policy blocks network requests.
- Exported course files may contain the user's course content and should be shared only with intended recipients.

## Dependencies

The registry audit found existing vulnerabilities on public main: 6 critical, 145 high, 169 moderate, and 24 low findings. These are dependency/path findings, not proof that every advisory is exploitable in this application's configuration.

The initial direct esbuild dependency added another path for a known low-severity Windows development-server file-read advisory (GHSA-g7r4-m6w7-qqqr). It was upgraded to patched esbuild 0.28.1 before publication. The offline builder uses the build API, rather than serving files through esbuild's development server.

Broader existing dependency upgrades require a separate compatibility review. This audit does not certify the repository as vulnerability-free. Repeat secret scanning and dependency auditing before future pushes and merges.
