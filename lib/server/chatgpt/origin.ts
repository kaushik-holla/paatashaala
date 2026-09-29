import type { NextRequest } from 'next/server';

export function isSameOrigin(request: NextRequest): boolean {
  const origin = request.headers.get('origin');
  const host = request.headers.get('host');
  if (!origin || !host) return false;
  try {
    const parsed = new URL(origin);
    return (
      parsed.host === host &&
      (parsed.protocol === 'https:' ||
        (parsed.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(parsed.hostname)))
    );
  } catch {
    return false;
  }
}
