# Paatashaala local setup

Paatashaala is an independent open-source application. It runs locally at <http://127.0.0.1:3000>. Use the same address and browser profile to keep access to courses stored in browser IndexedDB. The [sample lesson](http://127.0.0.1:3000/sample-lesson) works without an AI provider.

## First setup

Install Node.js 22.19 or newer and pnpm 10 or newer, then run:

```sh
corepack enable
corepack pnpm install --frozen-lockfile
cp .env.example .env.local
```

Choose an AI connection in **Settings → AI providers**. For ChatGPT device-code sign-in, first enable it in your [ChatGPT Security Settings](https://learn.chatgpt.com/docs/auth#preferred-device-code-authentication-beta), set `CHATGPT_SESSION_SECRET` in `.env.local` to a random value from `openssl rand -hex 32`, start the app, and select **ChatGPT** in Settings. You can instead enter an API key in Settings or configure a local server-managed provider in `.env.local`. API usage billed through a provider key is separate from a ChatGPT subscription. See [README.md](README.md) for model selection.

## Start and stop on macOS

Double-click **Start Paatashaala.command**. It starts the development server on `127.0.0.1:3000` and opens the app. Double-click **Stop Paatashaala.command** when finished. Restart after changing `.env.local`.

The local log is `local-setup/server.log`. Review it before sharing because prompts or provider error details may appear there.

## Courses and backups

Courses in the default setup live in the browser's IndexedDB. They survive an app restart but are tied to the browser profile and exact origin `http://127.0.0.1:3000`. Clearing browser data can delete them.

For every course you want to keep, use **Export Classroom ZIP** in the course Download menu. Store private exports outside this Git repository. Other exports include PPTX and narration scripts.

## Privacy and source

Provider requests can send prompts and source material to the service you choose. Keep API keys, personal materials, course exports, logs, and browser databases out of Git; `.env.local` is ignored. This repository's `origin` is the independent [Paatashaala repository](https://github.com/kaushik-holla/paatashaala). The original attribution and MIT copyright notice are linked from [Credits and license](README.md#credits-and-license).
