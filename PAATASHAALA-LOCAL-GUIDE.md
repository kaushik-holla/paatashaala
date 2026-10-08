# Paatashaala local setup

Paatashaala is an independent open-source application. It runs locally at <http://127.0.0.1:3000>. Courses are automatically saved in the local media folder and can be retrieved after clearing browser data. The [sample lesson](http://127.0.0.1:3000/sample-lesson) works without an AI provider.

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

Courses on localhost are automatically saved under `media/courses/<course-name>--<course-id>/`. Each folder contains `tutorial.json`, `manifest.json`, `assets/`, and immutable `versions/` snapshots. Names can change without breaking course IDs. The shared immutable asset pool is in `media/.assets/`. Set `PAATASHAALA_MEDIA_DIR` in `.env.local` to an absolute path to relocate the library; restart the app afterwards. A custom location starts a separate library: copy the existing media folder there first.

Open **Disk library / Saved to disk** from the home page or course header for saved courses, version history, and **Trash**. Deleting a course moves its entire folder to `media/.trash/`; restoration brings back its assets and history. Snapshots and Trash are never automatically pruned. After an unexpected server crash, stop all app processes before removing stale directories under `media/.locks/` if saves report that the library is busy.

Existing IndexedDB courses are copied when the library is loaded in the original browser profile and origin. Their browser copies are retained. Open old courses once to finish legacy-media conversion, verify the saved course in another browser, and back up `media/` before clearing browser data. Generation still runs in the browser in this fork: leave the tab open until it finishes; completed saves and intermediate course checkpoints survive closing it.

Back up the **entire media directory**, including hidden directories, to a separate drive or your backup service. A local disk copy protects against browser cleanup; a separate backup protects against disk loss. Custom media directories should also be excluded from any source-control repository.

### Study on your phone

From the course Download menu:

- **Classroom ZIP** (`.maic.zip`) bundles the editable course, media, quizzes, and activities. Transfer it through AirDrop, Files, or a cloud drive and use **Import Classroom** in Paatashaala to import it as a new course. Imports intentionally create new IDs and do not overwrite an existing course or its study progress.
- **Export as HTML** creates a single self-contained file with a responsive offline player, slides, locally graded choice quizzes, bundled interactive HTML, recorded narration, and available recorded videos. Transfer it to your phone and open it in a browser or HTML viewer that executes local HTML. Some phone Files previewers, particularly iOS Quick Look, do not execute these files; use a compatible HTML viewer. No native mobile app or automatic cloud sync is included.
- In the offline player, use **Save study progress** to download a small JSON file containing your current lesson and quiz answers. **Load study progress** restores it for the same course. The player also remembers progress when the browser permits local storage; keep the JSON file for reliable transfer and recovery. AI chat, AI grading, live project coaching, and network-dependent widgets require the online app. Short-answer quizzes show reference answers offline.

Watch for partial-export warnings: missing or unembeddable resources cannot work offline. Interactive activities have a restrictive network policy in the HTML player. Full ZIP export remains available for editable transfer, alongside PPTX, video, and narration-script exports.

The disk API accepts only localhost requests and same-origin writes. Shared/public deployments should use the existing PostgreSQL persistence mode with appropriate authentication. Set `NEXT_PUBLIC_COURSE_STORAGE=browser` to opt out of the local disk library; `NEXT_PUBLIC_PERSISTENCE=1` selects the existing database mode instead.

## Privacy and source

Provider requests can send prompts and source material to the service you choose. Keep API keys, personal materials, course exports, logs, and browser databases out of Git; `.env.local` is ignored. This repository's `origin` is the independent [Paatashaala repository](https://github.com/kaushik-holla/paatashaala). The original attribution and MIT copyright notice are linked from [Credits and license](README.md#credits-and-license).
