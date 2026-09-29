import { randomBytes } from 'node:crypto';
import { cookies } from 'next/headers';
import {
  loginOpenAICodexDeviceCode,
  refreshOpenAICodexToken,
  type OAuthCredentials,
  type OAuthDeviceCodeInfo,
} from '@earendil-works/pi-ai/oauth';
import { deleteChatGPTSession, readChatGPTSession, saveChatGPTSession } from './session-store';

export const SESSION_COOKIE = 'paatashaala_chatgpt_session';
export const PENDING_COOKIE = 'paatashaala_chatgpt_pending';

type Pending =
  | { state: 'waiting'; info?: OAuthDeviceCodeInfo; expires: number }
  | { state: 'complete'; credentials: OAuthCredentials; expires: number }
  | { state: 'error'; message: string; expires: number };
const pending = new Map<string, Pending>();

export function cookieOptions(maxAge = 60 * 60 * 24 * 30) {
  return {
    httpOnly: true,
    sameSite: 'lax' as const,
    secure: process.env.NODE_ENV === 'production',
    path: '/',
    maxAge,
  };
}

export function startChatGPTLogin(): { id: string; ready: Promise<OAuthDeviceCodeInfo> } {
  for (const [key, value] of pending) if (value.expires < Date.now()) pending.delete(key);
  const id = randomBytes(32).toString('hex');
  const expires = Date.now() + 10 * 60_000;
  pending.set(id, { state: 'waiting', expires });
  const ready = new Promise<OAuthDeviceCodeInfo>((resolve, reject) => {
    loginOpenAICodexDeviceCode({
      onDeviceCode(info) {
        pending.set(id, { state: 'waiting', info, expires });
        resolve(info);
      },
    })
      .then((credentials) => {
        pending.set(id, { state: 'complete', credentials, expires });
      })
      .catch((error) => {
        const message = error instanceof Error ? error.message : 'ChatGPT sign-in failed';
        pending.set(id, { state: 'error', message, expires });
        reject(new Error(message));
      });
  });
  return { id, ready };
}

export function getPendingLogin(id: string): Pending | null {
  const entry = pending.get(id);
  if (!entry) return null;
  if (entry.expires < Date.now()) {
    pending.delete(id);
    return null;
  }
  return entry;
}

export function clearPendingLogin(id: string): void {
  pending.delete(id);
}

export async function getChatGPTCredentials(): Promise<{
  id: string;
  credentials: OAuthCredentials;
} | null> {
  const id = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!id) return null;
  const credentials = await readChatGPTSession(id);
  if (!credentials) return null;
  if (credentials.expires > Date.now() + 60_000) return { id, credentials };
  const refreshed = await refreshOpenAICodexToken(credentials.refresh);
  await saveChatGPTSession(refreshed, id);
  return { id, credentials: refreshed };
}

export async function removeChatGPTSession(): Promise<void> {
  const store = await cookies();
  const id = store.get(SESSION_COOKIE)?.value;
  if (id) await deleteChatGPTSession(id);
  store.delete(SESSION_COOKIE);
  store.delete(PENDING_COOKIE);
}
