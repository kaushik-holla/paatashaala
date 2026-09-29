#!/usr/bin/env python3
"""Local process controls. Never reads or prints API keys."""
import json, os, shutil, signal, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'local-setup'
PIDFILE = STATE / 'server.pid'
URL = 'http://127.0.0.1:3000'
def owned_pid():
    try:
        pid = int(PIDFILE.read_text())
        os.kill(pid, 0)
        cwd = subprocess.run(['/usr/sbin/lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn'], capture_output=True, text=True).stdout
        return pid if ('n' + str(ROOT)) in cwd.splitlines() else None
    except (OSError, ValueError):
        return None

def stop():
    pid = owned_pid()
    if pid:
        os.killpg(pid, signal.SIGTERM)
        for _ in range(30):
            if not owned_pid(): break
            time.sleep(0.2)
        print('Stopped the local Paatashaala server.')
    else:
        print('No managed Paatashaala server is running.')
    PIDFILE.unlink(missing_ok=True)

def start():
    if not owned_pid():
        import socket
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', 3000)) == 0:
                raise SystemExit('Port 3000 is already in use. No existing process was stopped.')
        env = os.environ.copy()
        env['NEXT_TELEMETRY_DISABLED'] = '1'
        env['npm_package_version'] = json.loads((ROOT / 'package.json').read_text())['version']
        # Limit memory use while compiling this large application.
        env['NODE_OPTIONS'] = '--max-old-space-size=6144'
        node = shutil.which('node')
        next_bin = ROOT / 'node_modules/next/dist/bin/next'
        if not node:
            raise SystemExit('Node.js is not installed or is not on PATH.')
        if not next_bin.exists():
            raise SystemExit('Dependencies are missing. Run: corepack pnpm install --frozen-lockfile')
        with (STATE / 'server.log').open('ab') as log:
            process = subprocess.Popen([node, str(next_bin), 'dev', '--hostname', '127.0.0.1', '--port', '3000'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        PIDFILE.write_text(str(process.pid))
    print('Starting Paatashaala. First launch can take a minute...', flush=True)
    for _ in range(90):
        try:
            with urllib.request.urlopen(URL + '/api/health', timeout=3) as response:
                if response.status == 200:
                    print('Paatashaala is ready at ' + URL)
                    if '--no-browser' not in sys.argv: subprocess.run(['open', URL])
                    return
        except Exception:
            if not owned_pid():
                raise SystemExit('Server exited; see local-setup/server.log.')
            time.sleep(1)
    raise SystemExit('Server is still compiling. Check local-setup/server.log and try the URL shortly.')

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'start'
    if action in ('stop', 'restart'): stop()
    if action in ('start', 'restart'): start()
