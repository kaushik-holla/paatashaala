import { NextRequest, NextResponse } from 'next/server';
import { getModels } from '@earendil-works/pi-ai';
import {
  clearPendingLogin,
  cookieOptions,
  getChatGPTCredentials,
  getPendingLogin,
  PENDING_COOKIE,
  removeChatGPTSession,
  SESSION_COOKIE,
} from '@/lib/server/chatgpt/auth';
import { saveChatGPTSession } from '@/lib/server/chatgpt/session-store';
import { isSameOrigin } from '@/lib/server/chatgpt/origin';

export const runtime = 'nodejs';

const noStore = { 'Cache-Control': 'no-store' };

export async function GET(request: NextRequest) {
  const pendingId = request.cookies.get(PENDING_COOKIE)?.value;
  const pending = pendingId ? getPendingLogin(pendingId) : null;
  if (pending?.state === 'complete' && pendingId) {
    const id = await saveChatGPTSession(pending.credentials);
    clearPendingLogin(pendingId);
    const response = NextResponse.json(
      { connected: true, models: availableModels() },
      { headers: noStore },
    );
    response.cookies.set(SESSION_COOKIE, id, cookieOptions());
    response.cookies.delete(PENDING_COOKIE);
    return response;
  }
  if (pending?.state === 'error') {
    const response = NextResponse.json(
      { connected: false, error: pending.message },
      { headers: noStore },
    );
    response.cookies.delete(PENDING_COOKIE);
    if (pendingId) clearPendingLogin(pendingId);
    return response;
  }
  if (pending?.state === 'waiting') {
    return NextResponse.json(
      {
        connected: false,
        pending: true,
        verificationUri: pending.info?.verificationUri,
        userCode: pending.info?.userCode,
      },
      { headers: noStore },
    );
  }
  try {
    const session = await getChatGPTCredentials();
    return NextResponse.json(
      { connected: Boolean(session), models: session ? availableModels() : [] },
      { headers: noStore },
    );
  } catch {
    return NextResponse.json(
      { connected: false, error: 'Session expired. Sign in again.' },
      { headers: noStore },
    );
  }
}

export async function DELETE(request: NextRequest) {
  if (!isSameOrigin(request)) {
    return NextResponse.json({ error: 'Invalid request origin' }, { status: 403 });
  }
  await removeChatGPTSession();
  return NextResponse.json({ connected: false }, { headers: noStore });
}

function availableModels() {
  return getModels('openai-codex')
    .filter((model) => !model.id.includes('codex'))
    .map((model) => ({ id: model.id, name: model.name }));
}
