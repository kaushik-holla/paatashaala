'use client';

import { useCallback, useEffect, useState } from 'react';
import { CheckCircle2, ExternalLink, Loader2, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';

type AuthStatus = {
  connected: boolean;
  pending?: boolean;
  verificationUri?: string;
  userCode?: string;
  models?: Array<{ id: string; name: string }>;
  error?: string;
};

export function ChatGPTConnection({
  connectedInSettings,
  onConnectionChange,
  onModelsFetched,
}: {
  connectedInSettings: boolean;
  onConnectionChange: (connected: boolean) => void;
  onModelsFetched?: (ids: string[]) => number;
}) {
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    const response = await fetch('/api/chatgpt/auth', { cache: 'no-store' });
    const next = (await response.json()) as AuthStatus;
    setStatus(next);
    if (next.connected !== connectedInSettings) onConnectionChange(next.connected);
    if (next.connected && next.models?.length)
      onModelsFetched?.(next.models.map((model) => model.id));
    if (next.error) setError(next.error);
    return next;
  }, [connectedInSettings, onConnectionChange, onModelsFetched]);

  useEffect(() => {
    void refresh().catch(() => setError('Could not check ChatGPT sign-in.'));
    // The initial status is enough; polling below handles an active sign-in.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!status?.pending) return;
    const timer = window.setInterval(
      () => void refresh().catch(() => setError('Could not check sign-in.')),
      2500,
    );
    return () => window.clearInterval(timer);
  }, [status?.pending, refresh]);

  async function start() {
    setBusy(true);
    setError('');
    try {
      const response = await fetch('/api/chatgpt/auth/start', { method: 'POST' });
      const data = (await response.json()) as AuthStatus;
      if (!response.ok) throw new Error(data.error ?? 'Unable to start sign-in');
      setStatus({
        connected: false,
        pending: true,
        verificationUri: data.verificationUri,
        userCode: data.userCode,
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to start sign-in');
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    setBusy(true);
    try {
      const response = await fetch('/api/chatgpt/auth', { method: 'DELETE' });
      if (!response.ok) throw new Error('Could not sign out');
      setStatus({ connected: false });
      onConnectionChange(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not sign out');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-card to-card p-5 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold">Use your ChatGPT account</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Connect through OpenAI’s device-code sign-in. Your account stays private to this browser
            session.
          </p>
        </div>
        {status?.connected && <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600" />}
      </div>
      {!status?.connected && (
        <p className="rounded-xl border border-primary/20 bg-background/70 p-3 text-sm text-muted-foreground">
          First, enable <span className="font-medium text-foreground">device code sign-in</span> in
          your ChatGPT Security Settings. If your account belongs to a managed workspace, an admin
          may need to allow it.{' '}
          <a
            href="https://learn.chatgpt.com/docs/auth#preferred-device-code-authentication-beta"
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-primary hover:underline"
          >
            View OpenAI’s instructions
          </a>
          .
        </p>
      )}
      {status?.pending && status.verificationUri && (
        <div className="rounded-xl border bg-background/80 p-4 space-y-3">
          <p className="text-sm">Open the secure sign-in page and enter this one-time code:</p>
          <div className="rounded-lg border bg-muted/40 px-4 py-2 font-mono text-lg font-semibold tracking-widest select-all w-fit">
            {status.userCode}
          </div>
          <a
            href={status.verificationUri}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline"
          >
            Open OpenAI device sign-in <ExternalLink className="h-3.5 w-3.5" />
          </a>
          <p className="text-xs text-muted-foreground">Waiting for you to finish in the browser…</p>
          <button
            type="button"
            onClick={start}
            disabled={busy}
            className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
          >
            Get a new code
          </button>
        </div>
      )}
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {status?.connected ? (
        <div className="flex items-center justify-between gap-3">
          <span className="text-sm font-medium text-emerald-700 dark:text-emerald-400">
            Connected to ChatGPT
          </span>
          <Button type="button" variant="outline" size="sm" onClick={signOut} disabled={busy}>
            <LogOut className="mr-2 h-4 w-4" />
            Sign out
          </Button>
        </div>
      ) : (
        !status?.pending && (
          <Button type="button" onClick={start} disabled={busy} className="rounded-xl">
            {busy && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}Continue with ChatGPT
          </Button>
        )
      )}
    </div>
  );
}
