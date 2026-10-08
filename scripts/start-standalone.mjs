import { spawn } from 'node:child_process';
import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';

const buildDir = resolve(process.env.PAATASHAALA_BUILD_DIR || '.next');
const standaloneDir = join(buildDir, 'standalone');
const server = join(standaloneDir, 'server.js');

if (!existsSync(server)) {
  console.error('Standalone build not found. Run pnpm build first.');
  process.exit(1);
}

// Next.js leaves static assets and public files outside standalone output.
for (const [source, destination] of [
  [resolve('public'), join(standaloneDir, 'public')],
  [join(buildDir, 'static'), join(standaloneDir, '.next', 'static')],
]) {
  if (!existsSync(source)) continue;
  mkdirSync(destination, { recursive: true });
  cpSync(source, destination, { recursive: true, force: true });
}

// The standalone server changes cwd to its generated checkout. Keep local
// courses beside the project, so rebuilding never replaces the library.
const child = spawn(process.execPath, [server], {
  stdio: 'inherit',
  env: { ...process.env, PAATASHAALA_PROJECT_DIR: process.cwd() },
});
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
child.on('exit', (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
