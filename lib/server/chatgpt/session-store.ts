import { createCipheriv, createDecipheriv, createHash, randomBytes } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Pool } from 'pg';
import type { OAuthCredentials } from '@earendil-works/pi-ai';

const SESSION_DIR = path.join(process.cwd(), '.data', 'chatgpt-sessions');
let pool: Pool | undefined;
let tableReady: Promise<void> | undefined;

export function assertChatGPTSessionConfigured(): Buffer {
  const secret = process.env.CHATGPT_SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error(
      'CHATGPT_SESSION_SECRET must be set to a random value of at least 32 characters.',
    );
  }
  return createHash('sha256').update(secret).digest();
}

function encrypt(credentials: OAuthCredentials): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', assertChatGPTSessionConfigured(), iv);
  const payload = Buffer.concat([cipher.update(JSON.stringify(credentials)), cipher.final()]);
  return Buffer.concat([iv, cipher.getAuthTag(), payload]).toString('base64url');
}

function decrypt(value: string): OAuthCredentials {
  const data = Buffer.from(value, 'base64url');
  const decipher = createDecipheriv(
    'aes-256-gcm',
    assertChatGPTSessionConfigured(),
    data.subarray(0, 12),
  );
  decipher.setAuthTag(data.subarray(12, 28));
  return JSON.parse(
    Buffer.concat([decipher.update(data.subarray(28)), decipher.final()]).toString(),
  );
}

function database(): Pool | undefined {
  if (!process.env.DATABASE_URL) return undefined;
  pool ??= new Pool({ connectionString: process.env.DATABASE_URL });
  tableReady ??= pool
    .query(
      `CREATE TABLE IF NOT EXISTS chatgpt_sessions (
    id_hash text PRIMARY KEY, credentials text NOT NULL, created_at timestamptz NOT NULL DEFAULT now()
  )`,
    )
    .then(() => undefined);
  return pool;
}

function idHash(id: string): string {
  return createHash('sha256').update(id).digest('hex');
}

function validId(id: string): boolean {
  return /^[a-f0-9]{64}$/.test(id);
}

export async function saveChatGPTSession(
  credentials: OAuthCredentials,
  id = randomBytes(32).toString('hex'),
): Promise<string> {
  if (!validId(id)) throw new Error('Invalid session ID');
  const value = encrypt(credentials);
  const db = database();
  if (db) {
    await tableReady;
    await db.query(
      'INSERT INTO chatgpt_sessions (id_hash, credentials) VALUES ($1, $2) ON CONFLICT (id_hash) DO UPDATE SET credentials = EXCLUDED.credentials',
      [idHash(id), value],
    );
  } else {
    await mkdir(SESSION_DIR, { recursive: true, mode: 0o700 });
    const filename = path.join(SESSION_DIR, idHash(id));
    const temporary = `${filename}.${randomBytes(8).toString('hex')}.tmp`;
    await writeFile(temporary, value, { mode: 0o600 });
    await rename(temporary, filename);
  }
  return id;
}

export async function readChatGPTSession(id: string): Promise<OAuthCredentials | null> {
  if (!validId(id)) return null;
  const db = database();
  let value: string | undefined;
  if (db) {
    await tableReady;
    const result = await db.query<{ credentials: string }>(
      'SELECT credentials FROM chatgpt_sessions WHERE id_hash = $1',
      [idHash(id)],
    );
    value = result.rows[0]?.credentials;
  } else {
    value = await readFile(path.join(SESSION_DIR, idHash(id)), 'utf8').catch(() => undefined);
  }
  if (!value) return null;
  try {
    return decrypt(value);
  } catch {
    return null;
  }
}

export async function deleteChatGPTSession(id: string): Promise<void> {
  if (!validId(id)) return;
  const db = database();
  if (db) {
    await tableReady;
    await db.query('DELETE FROM chatgpt_sessions WHERE id_hash = $1', [idHash(id)]);
  } else {
    await rm(path.join(SESSION_DIR, idHash(id)), { force: true });
  }
}
