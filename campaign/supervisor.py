"""One assignment's author, request execution, fixed nomination and confirmation."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

from .binding import clean_checkout, read_object
from .custody import Request, nominate, read_request, snapshot
from .owner import claim_confirmation, publish_new
from .processes import clean_environment, run_bounded, sandbox, runtime_environment

SOURCE = Path(__file__).resolve().parents[1]


class AssignmentSupervisor:
    def __init__(self, root: Path, state: Path):
        self.root, self.state = root, state
        self.binding = read_object(root / 'campaign/binding.json')
        self.identity = self.binding['assignment_id']
        self.claim = read_object(state / 'claims' / f'{self.identity}.json')
        self.folder = state / 'assignments' / self.identity
        self.protocol = read_object(root / self.binding['measurement_protocol'])
        self.results = []
        self.search_closed = False

    def execute(self, identity: str, phase: str, candidate: str) -> dict:
        output = self.folder / 'evaluations' / identity
        output.mkdir(parents=True, exist_ok=False)
        stop = (self.claim['execution_stop_at_epoch'] if phase == 'confirmation'
                else self.claim['search_stop_at_epoch'])
        request = {'id': identity, 'phase': phase, 'candidate': candidate,
                   'snapshot': str(self.folder / 'snapshots' / candidate), 'stop_at_epoch': stop,
                   'hard_stop_at_epoch': self.claim['stop_at_epoch']}
        publish_new(output / 'request.json', request)
        command = sandbox(self.root, SOURCE, writable=[output],
                          extra_readonly=[self.folder / 'snapshots' / candidate],
                          gpu=True, protocol=self.protocol)
        # Only this attempt is visible to the worker, not other candidates or results.
        command += [self.protocol['deployment']['assay_python'], '-I', str(SOURCE / 'campaign/entry.py'),
                    'assay', '--root', str(self.root), '--state', str(self.state),
                    '--request', str(output / 'request.json')]
        code = run_bounded(command, cwd=self.root, env=runtime_environment(self.root), stop=stop,
                           hard_stop=self.claim['stop_at_epoch'],
                           log=output / 'process.log', process_record=output / 'process.json')
        if code != 0 or not (output / 'result.json').is_file():
            raise RuntimeError(f'assay outcome unknown, exit {code}; this opportunity is consumed')
        result = read_object(output / 'result.json')
        if any(result.get(key) != request[key] for key in ('id', 'phase', 'candidate')):
            raise ValueError('assay result identity differs')
        self.results.append(result)
        return result

    def service(self):
        incoming = self.root / 'campaign/requests'
        for path in sorted(incoming.iterdir()):
            identity = path.stem
            from .custody import NAME
            if path.suffix != '.json' or not NAME.fullmatch(identity):
                self.search_closed = True
                raise ValueError('author request filename differs')
            claim = self.folder / 'requests' / f'{identity}.json'
            if claim.exists():
                continue
            publish_new(claim, {'claimed_at_epoch': time.time(), 'opportunity_consumed': True})
            try:
                request = Request.parse(read_request(path), identity)
                if self.search_closed or time.time() >= self.claim['search_stop_at_epoch']:
                    raise TimeoutError('search closed before request execution')
                if request.phase == 'search':
                    snapshot(self.root / 'campaign/candidates', request.candidate,
                             self.folder / 'snapshots' / request.candidate)
                elif not any(row['phase'] == 'search' and row['status'] == 'passed'
                             and row['candidate'] == request.candidate for row in self.results):
                    raise ValueError('profile requires an already evaluated immutable candidate')
                response = self.execute(request.identity, request.phase, request.candidate)
            except Exception as error:
                response = {'id': identity, 'status': 'unknown', 'no_retry': True,
                            'reason': f'{type(error).__name__}: {error}'}
                self.search_closed = True
            publish_new(self.root / 'campaign/responses' / f'{identity}.json', response)
            if self.search_closed:
                raise RuntimeError('search closed after unknown request; no author replacement')

    def author(self):
        deployment = self.protocol['deployment']
        spec = importlib.util.spec_from_file_location('selected_compiler_tools',
                                                     self.root / 'campaign/hmz_compiler_tools.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        catalog = adapter.catalog(self.root)
        publish_new(self.folder / 'catalog-intake.json', catalog)
        writable = [self.root / 'campaign' / name for name in
                    ('candidates', 'schedules', 'compiler-actions', 'requests')]
        home = self.folder / 'author-home'
        home.mkdir()
        command = sandbox(self.root, SOURCE, writable=writable, protocol=self.protocol, author=True)
        command += ['--bind', str(home), '/home/author',
                    str(Path(deployment['hmz_root']) / 'bin/hmz'), 'exec',
                    '-f', str(SOURCE / 'campaign/author_flow.py'),
                    '-a', 'agent=claude/glm-5.3:high',
                    '-p', f"budget.duration={max(1, self.claim['search_stop_at_epoch'] - time.time())},budget.graceful=false",
                    (self.root / self.binding['author']['scaffold']).read_text()]
        credential = Path(deployment['provider_credential']).read_text().strip()
        if not credential:
            raise ValueError('provider credential is empty')
        env = clean_environment()
        env.update(ANTHROPIC_AUTH_TOKEN=credential, ANTHROPIC_API_KEY=credential,
                   ANTHROPIC_BASE_URL='https://cloud.infini-ai.com/maas', ANTHROPIC_MODEL='glm-5.3',
                   ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.3', ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.3',
                   ANTHROPIC_DEFAULT_HAIKU_MODEL='glm-5.3')
        env['PATH'] = os.pathsep.join((str(Path(deployment['hmz_root']) / 'bin'),
                                      deployment['provider_native'], env['PATH']))
        return run_bounded(command, cwd=self.root, env=env, stop=self.claim['search_stop_at_epoch'],
                           hard_stop=self.claim['stop_at_epoch'],
                           log=self.folder / 'author.log', process_record=self.folder / 'author-process.json',
                           service=self.service)

    def run(self):
        self.folder.mkdir(parents=True, exist_ok=False)
        clean_checkout(self.root, self.binding_with_commit())
        publish_new(self.root / 'campaign/deadline.json', self.claim)
        publish_new(self.root / 'campaign/intake.json', {
            'source_commit': self.binding_with_commit(), 'bench_commit': self.binding['bench_commit'],
            'compiler_commit': self.binding['compiler_commit'], 'claim': self.claim})
        for name in ('requests', 'responses', 'candidates', 'schedules', 'compiler-actions'):
            (self.root / 'campaign' / name).mkdir(exist_ok=True)
        author_exit, failure = None, None
        confirmation = {'status': 'missing'}
        nominee = None
        try:
            author_exit = self.author()
            self.service()
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
        finally:
            self.search_closed = True
            publish_new(self.folder / 'search-closed.json', {'at_epoch': time.time(), 'reason': failure})
            for path in (self.root / 'campaign/requests').glob('*.json'):
                response = self.root / 'campaign/responses' / path.name
                if not response.exists():
                    publish_new(response, {'status': 'unknown', 'no_retry': True,
                                           'reason': 'author exited or deadline closed'})
        if failure is None or failure.startswith('TimeoutError: process deadline'):
            nominee = nominate(self.results, search_stop=self.claim['search_stop_at_epoch'])
        if nominee is not None:
            publish_new(self.folder / 'nominee.json', nominee)
            try:
                claim_confirmation(self.state, self.identity, nominee, now=time.time())
                confirmation = self.execute('final-confirmation', 'confirmation', nominee['id'])
            except Exception as error:
                confirmation = {'status': 'unknown', 'no_retry': True,
                                'reason': f'{type(error).__name__}: {error}'}
        selected = next((row for row in self.results if nominee and row['id'] == nominee['search_evaluation']), {})
        if not selected:
            selected = next((row for row in reversed(self.results)
                             if row.get('phase') == 'search' and row.get('correctness')), {})
        confirmed = (confirmation.get('status') == 'passed'
                     and confirmation.get('measurement', {}).get('quality_passed') is True)
        status = ('confirmed' if confirmed else 'confirmation_failed') if nominee else 'no_qualified_candidate'
        if confirmed and confirmation['measurement']['candidate_over_baseline'] >= 1:
            status = 'no_gain'
        workloads = selected.get('workloads', {})
        for identity, row in workloads.items():
            measured = confirmation.get('workloads', {}).get(identity, {})
            row['profile'] = {'status': 'passed' if nominee else 'missing'}
            row['confirmation'] = {'status': 'passed' if confirmed and measured else confirmation['status']}
        result = {'assignment_id': self.identity,
                  'execution': {'status': status, 'author_exit': author_exit, 'reason': failure},
                  'correctness': selected.get('correctness', {'status': 'missing'}),
                  'performance': {'status': 'measured' if selected.get('measurement', {}).get('quality_passed') is True
                                  else 'coverage_missing',
                                  'search': selected.get('measurement'),
                                  'confirmation': confirmation.get('measurement')},
                  'confirmation': confirmation, 'profile': {'status': 'passed' if nominee else 'missing'},
                  'workloads': workloads}
        publish_new(self.folder / 'terminal.json', result)
        return result

    def binding_with_commit(self):
        from .binding import git
        commit = git(self.root, 'rev-parse', 'HEAD')
        if self.claim.get('prepared_commit') != commit:
            raise ValueError('claim is not bound to the prepared source')
        return commit


def supervise(root: Path, state: Path):
    return AssignmentSupervisor(root, state).run()
