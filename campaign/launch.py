"""Own one fresh Bench author session and its original bounded final confirmation on C500."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROVIDER = Path.home() / '.cake-c500-run007-claude-restricted'


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    os.chdir(ROOT)
    binding = json.loads((ROOT / 'campaign/binding.json').read_text())
    if sys.argv[1:] not in ([], ['--check']):
        raise ValueError('launch takes only optional --check')

    if not PROVIDER.is_file() or not os.access(PROVIDER, os.X_OK):
        raise RuntimeError('Author provider unavailable: ' + str(PROVIDER))
    subprocess.run([str(PROVIDER), '--version'], check=True, timeout=30,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if sys.argv[1:] == ['--check']:
        print('Author provider is available; binding loaded')
        return

    plan = json.loads((ROOT / 'campaign/frozen-plan.json').read_text())
    author = binding['author']
    wall = author['wall_time_seconds']
    reserve = author['confirmation_seconds']
    now = time.time()
    deadline = {
        'started_at_epoch': now,
        'search_stop_at_epoch': now + wall - reserve,
        'stop_at_epoch': now + wall,
        'wall_time_seconds': wall,
        'confirmation_seconds': reserve,
        'selected_physical_device': binding['physical_device'],
        'preparation_cost_included': True,
    }
    write_new(ROOT / 'campaign/deadline.json', deadline)

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(binding['physical_device'])
    env['LD_LIBRARY_PATH'] = '/opt/maca/lib:/opt/maca/mxgpu_llvm/lib'
    env['MACA_PATH'] = '/opt/maca'
    for k in ('http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        env.pop(k, None)
    import glob as _glob
    _pypath_parts = (
        _glob.glob(str(ROOT / '.venv/lib/python*/site-packages'))
        + [str(ROOT / '.deps/sol-execbench/src')]
    )
    _existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = ':'.join(p for p in _pypath_parts + [_existing] if p)

    task_text = (ROOT / 'campaign/TASK.md').read_text()
    scaffold_text = (ROOT / author['scaffold']).read_text()
    initial_prompt = '\n\n'.join([scaffold_text, task_text])

    model_code, error = None, None
    (ROOT / 'campaign/logs').mkdir(exist_ok=True)
    try:
        remaining = int(deadline['search_stop_at_epoch'] - time.time())
        if remaining <= 0:
            raise ValueError('No search time remains after preparation')
        with (ROOT / 'campaign/logs/ralph.log').open('x') as log:
            child = subprocess.Popen(
                [str(PROVIDER), '--restricted', '--output-format', 'stream-json',
                 '--verbose', '-p', initial_prompt],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                model_code = child.wait(timeout=max(1, remaining))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                model_code = 124

        with (ROOT / 'campaign/logs/final-confirmation-owner.log').open('x') as log:
            subprocess.run(
                [sys.executable, '-m', 'campaign.final_confirmation'],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=max(1, deadline['stop_at_epoch'] - time.time()),
            )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        write_new(ROOT / 'campaign/controller-status.json', {
            'model_exit_code': model_code,
            'error': error,
            'done_exists': (ROOT / 'campaign/DONE.json').exists(),
            'owner_finalized_at_epoch': time.time(),
        })


if __name__ == '__main__':
    main()
