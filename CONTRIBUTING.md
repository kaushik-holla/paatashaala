# Contributing to Paatashaala

Thanks for helping improve Paatashaala. Please open issues and pull requests in [this repository](https://github.com/kaushik-holla/paatashaala).

## Get started

Use Node.js 22.19 or newer and pnpm 10. From a fresh clone:

```bash
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev
```

The [sample lesson](http://localhost:3000/sample-lesson) works without a provider key. For AI features, follow the [README](README.md) and keep real credentials in `.env.local`. Never include credentials, personal course material, or generated user data in a pull request.

## Branch and pull request names

Name branches after the work, using a short prefix such as `feature/`, `fix/`, `docs/`, or `chore/`. For example, use `fix/timeline-scrubbing` or `docs/project-site`. Give each pull request a title that describes the change. Keep assistant and tool branding out of branch names and pull request titles.

## Make a change

Open an issue for a bug or feature request, or submit a focused pull request against `main`. Describe the user-facing behavior, include screenshots for UI changes, and say what you tested. Update `.env.example` and local setup instructions when adding a configuration setting. Keep user-facing copy in the existing translation system.

Before submitting, run:

```bash
pnpm check
pnpm lint
pnpm check:i18n-keys
pnpm exec tsc --noEmit
pnpm test
pnpm build
```

Before pushing or merging, run a redacted secret scan of the branch history (`gitleaks git --redact`) and a dependency audit (`pnpm audit`). Compare dependency findings with `main`, fix newly introduced vulnerable dependencies, and document remaining findings. Confirm that environment files, credentials, generated courses, media, and local logs are excluded from the published commits. Use the public Paatashaala repository for branches and merges.

Some tests need local services or a browser; note any check you could not run. The internal workspace packages are not published to npm.

## Security and license

Please use the process in [SECURITY.md](SECURITY.md) for vulnerabilities; do not disclose exploit details in a public issue. Contributions to the main application are under [MIT](LICENSE), while bundled components retain the terms in their own license files. The [README](README.md#credits-and-license) identifies the original foundation and bundled license notices.
