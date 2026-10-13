"""Pass evaluation requests from an isolated author to its task controller."""
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def validate(root, request):
    if set(request) != {'candidate', 'id', 'phase', 'route'}:
        raise ValueError('evaluation request fields differ')
    identity = request['id']
    if (not isinstance(identity, str) or not identity
            or not identity.replace('-', '').replace('_', '').isalnum()):
        raise ValueError('invalid evaluation id')
    if request['phase'] not in {'search', 'attribution'}:
        raise ValueError('the controller owns final confirmation')
    if request['route'] not in {'cake', 'native_triton'}:
        raise ValueError('unknown authoring route')
    candidate = (root / request['candidate']).resolve(strict=True)
    if root.resolve() not in candidate.parents or not candidate.is_file():
        raise ValueError('candidate is outside the task')
    return candidate


def submit(root, candidate, identity, phase, route):
    request = {'candidate': str(candidate), 'id': identity, 'phase': phase, 'route': route}
    validate(root, request)
    deadline = json.loads((root / 'campaign/deadline.json').read_text())
    if time.time() >= deadline['search_stop_at_epoch']:
        raise TimeoutError('search budget is closed')
    folder = root / 'campaign/evaluation-requests' / identity
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / 'request.json', request)
    while time.time() < deadline['search_stop_at_epoch']:
        response = folder / 'response.json'
        if response.exists():
            value = json.loads(response.read_text())
            if 'error' in value:
                raise RuntimeError(value['error'])
            return value['outcome']
        if (root / 'campaign/SEARCH_CLOSE.json').exists():
            raise RuntimeError('controller closed search; request remains retained')
        time.sleep(0.2)
    raise TimeoutError('evaluation response not received before deadline; do not retry')


def execute(root, request, env, remaining, folder):
    with (folder / 'controller.log').open('x') as log:
        process = subprocess.Popen([
            sys.executable, '-m', 'development.owner_evaluation',
            '--candidate', request['candidate'], '--id', request['id'],
            '--phase', request['phase'], '--route', request['route'],
        ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,start_new_session=True)
        try:return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            finally:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            raise


def abandon_pending(root, reason):
    for folder in (root/'campaign/evaluation-requests').glob('*'):
        if folder.is_dir() and not (folder/'response.json').exists():
            write_new(folder/'response.json',{'error':reason,'no_retry':True})


def service(root, env, runner=execute):
    for folder in sorted((root / 'campaign/evaluation-requests').glob('*')):
        request_path = folder / 'request.json'
        if not folder.is_dir() or (folder/'response.json').exists() or (folder / 'claimed.json').exists():
            continue
        if not request_path.is_file() and time.time()-folder.stat().st_mtime<1:
            continue
        write_new(folder / 'claimed.json', {'pid': os.getpid(), 'claimed_at_epoch': time.time()})
        try:
            if not request_path.is_file():
                raise RuntimeError('incomplete author request; outcome unknown')
            request = json.loads(request_path.read_text())
            validate(root, request)
            if folder.name != request['id']:
                raise ValueError('request id differs from its folder')
            deadline = json.loads((root / 'campaign/deadline.json').read_text())
            remaining = deadline['search_stop_at_epoch'] - time.time()
            if remaining <= 0 or (root / 'campaign/SEARCH_CLOSE.json').exists():
                raise TimeoutError('controller search is closed')
            code = runner(root, request, env, remaining, folder)
            outcome_path = root / 'campaign/evaluations' / request['id'] / 'outcome.json'
            if code != 0 or not outcome_path.is_file():
                raise RuntimeError(f'controller evaluation failed with exit {code}; see controller.log')
            response = {'outcome': json.loads(outcome_path.read_text())}
        except Exception as error:
            response = {'error': f'{type(error).__name__}: {error}', 'no_retry': True}
            close = root / 'campaign/SEARCH_CLOSE.json'
            if not close.exists():
                write_new(close, {'reason': 'controller_evaluation_fault', 'request': folder.name})
        write_new(folder / 'response.json', response)
