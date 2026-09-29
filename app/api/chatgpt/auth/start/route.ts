import { NextRequest, NextResponse } from 'next/server';
import { cookieOptions, PENDING_COOKIE, startChatGPTLogin } from '@/lib/server/chatgpt/auth';
import { assertChatGPTSessionConfigured } from '@/lib/server/chatgpt/session-store';
import { isSameOrigin } from '@/lib/server/chatgpt/origin';

export const runtime = 'nodejs';
export const maxDuration = 30;

export async function POST(request: NextRequest) {
  if (!isSameOrigin(request)) {
    return NextResponse.json({ error: 'Invalid request origin' }, { status: 403 });
  }
  try {
    assertChatGPTSessionConfigured();
    const { id, ready } = startChatGPTLogin();
    const info = await ready;
    const response = NextResponse.json({
      verificationUri: info.verificationUri,
      userCode: info.userCode,
      intervalSeconds: info.intervalSeconds,
    });
    response.cookies.set(PENDING_COOKIE, id, cookieOptions(10 * 60));
    return response;
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unable to start sign-in' },
      { status: 502 },
    );
  }
}
