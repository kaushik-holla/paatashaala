# Paatashaala security policy

Please report a suspected vulnerability privately through [GitHub's vulnerability reporting page](https://github.com/kaushik-holla/paatashaala/security/advisories/new) when it is available. If private reporting is not enabled, open an issue asking the maintainers for a private contact method **without posting exploit details or sensitive data**. Include the affected version or commit, deployment configuration, reproduction steps, and potential impact in the private report.

The `main` branch is the currently maintained source. A published security advisory will identify affected releases and the fix when one is available. We do not promise a fixed response or release time.

## Local data and keys

- Keep `.env.local`, provider keys, session secrets, generated course data, exports, and logs out of Git.
- ChatGPT browser sign-in creates an encrypted per-browser session when `CHATGPT_SESSION_SECRET` is set. It does not authenticate a user to Paatashaala or provide an OpenAI Platform API key.
- The default course library lives in the browser profile. Export courses you need to preserve before clearing browser data.
- The optional render service processes untrusted documents and HTML. See [render-service/README.md](render-service/README.md) for its isolation settings.

If you find an issue in inherited code that affects Paatashaala, report it through the same private channel.
