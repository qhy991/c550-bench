"""CPU lifecycle regression for a sealed confirmation timeout; no provider/device calls."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import development.launch as launch


class ConfirmationTimeoutTerminal(unittest.TestCase):
    def exercise(self, *, live=(), late_receipt=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'campaign/logs').mkdir(parents=True)
            (root/'campaign/candidates').mkdir()
            (root/'campaign/candidates/candidate.py').write_text('# own candidate fixture\n')
            (root/'campaign/TASK.md').write_text('Fixture task')
            skills={'evolve':'skills/evolve/SKILL.md',
                    'metax-c550-hardware-exploration':'skills/metax-c550-hardware-exploration/SKILL.md'}
            for relative in skills.values():
                p=root/'.deps/cake-ir'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
            binding={'task':'fixture','compiler':'fixture-compiler','skills':skills,
                     'routes':{'default':'cake'},'search_seconds':9720,
                     'wall_time_seconds':10800,'confirmation_seconds':1080}
            (root/'campaign/development-binding.json').write_text(json.dumps(binding))
            clock=[1000]
            nominee=(1,'search',{'candidate_path':'campaign/candidates/candidate.py',
                                'route':'cake','source_sha256':'fixture-source',
                                'candidate_sha256':'fixture-candidate'},'profile')
            def author(*args):
                clock[0]=10720
                return 124
            def timeout(*args):
                self.assertEqual(args[3],1080)
                clock[0]=11800
                if late_receipt:
                    p=root/'campaign/evaluations/final-confirmation/outcome.json'
                    p.parent.mkdir(parents=True)
                    p.write_text(json.dumps({'status':'accepted','phase':'confirmation',
                        'source_sha256':'fixture-source','candidate_sha256':'fixture-candidate'}))
                raise subprocess.TimeoutExpired(['confirmation-fixture'],1080)
            with ExitStack() as stack:
                for target,value in (
                    ('development.launch.ROOT',root),
                    ('development.launch.provider_environment',lambda _: (Path('/fixture/hmz'),{})),
                    ('development.launch.subprocess.check_output',lambda *a,**k: '' if 'status' in a[0] else 'fixture-commit'),
                    ('development.launch.subprocess.run',lambda *a,**k: None),
                    ('development.controller_entry.controller_command',lambda *a: ['fixture-prepare']),
                    ('development.launch.run_process',author),
                    ('development.launch.select_nominee',lambda: nominee),
                    ('development.launch.live_task_processes',lambda: list(live)),
                    ('development.launch.time.time',lambda: clock[0]),
                    ('development.submission.execute',timeout)):
                    stack.enter_context(patch(target,value))
                fake_digest=stack.enter_context(patch('development.launch.sha256'))
                fake_digest.return_value.hexdigest.return_value='fixture-source'
                with self.assertRaises(subprocess.TimeoutExpired):
                    launch.main([])
            self.assertTrue((root/'campaign/DONE.json').exists(),'timeout must retain a failed terminal')
            done=json.loads((root/'campaign/DONE.json').read_text())
            status=json.loads((root/'campaign/controller-status.json').read_text())
            self.assertTrue(done['terminal'])
            self.assertEqual(done['status'],'no_qualified_result')
            self.assertEqual(done['model_exit'],124)
            self.assertEqual(done['confirmation']['status'],'budget_exceeded')
            self.assertEqual(done['live_task_processes'],list(live))
            self.assertEqual(done['device_release_verified'],not live)
            self.assertTrue(status['done_exists'])
            self.assertIn('TimeoutExpired',status['error'])
            if late_receipt:
                self.assertEqual(json.loads((root/'campaign/evaluations/final-confirmation/outcome.json').read_text())['status'],'accepted')

    def test_timeout_retains_failure_terminal_and_original_exception(self):
        self.exercise()

    def test_timeout_does_not_infer_release_when_own_process_survives(self):
        self.exercise(live=(12345,))

    def test_late_accepted_receipt_cannot_replace_timeout(self):
        self.exercise(late_receipt=True)
