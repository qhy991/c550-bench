"""Whole-task custody and failure semantics without provider or GPU side effects."""
import contextlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
import time
import types
import unittest
from unittest.mock import patch

from campaign.assay import PAIRS, isolated_measurement_coverage, paired_summary
from campaign.custody import Request, nominate, snapshot
from campaign.live import admit_live
from campaign.owner import claim_confirmation, publish_new
from campaign.supervisor import AssignmentSupervisor
from campaign.processes import ProcessIdentity, run_bounded


class CustodyTests(unittest.TestCase):
    def test_complete_source_bundle_is_snapshotted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidate = root / 'candidates/a'
            candidate.mkdir(parents=True)
            (candidate / 'candidate.py').write_text('from helper import run\n')
            (candidate / 'helper.py').write_text('def run(x): return x + 1\n')
            saved = snapshot(root / 'candidates', 'a', root / 'sealed/a')
            (candidate / 'helper.py').write_text('def run(x): return x - 1\n')
            self.assertEqual((saved / 'helper.py').read_text(), 'def run(x): return x + 1\n')
            self.assertEqual((saved / 'helper.py').stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                snapshot(root / 'candidates', 'a', root / 'sealed/a')

    def test_symlink_and_hardlink_cannot_import_external_material(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidate = root / 'candidates/a'
            candidate.mkdir(parents=True)
            foreign = root / 'foreign.py'
            foreign.write_text('secret = 1\n')
            (candidate / 'candidate.py').symlink_to(foreign)
            with self.assertRaises(ValueError):
                snapshot(root / 'candidates', 'a', root / 'sealed/a')
            (candidate / 'candidate.py').unlink()
            os.link(foreign, candidate / 'candidate.py')
            with self.assertRaises(ValueError):
                snapshot(root / 'candidates', 'a', root / 'sealed/b')

    def test_author_cannot_claim_confirmation_or_choose_external_candidate(self):
        for phase, candidate in [('confirmation', 'a'), ('search', '../foreign'), ('search', '/tmp/a')]:
            with self.subTest(phase=phase, candidate=candidate), self.assertRaises(ValueError):
                Request.parse({'id': 'one', 'phase': phase, 'candidate': candidate}, 'one')

    def test_profile_must_match_the_search_snapshot_and_original_clock(self):
        search = {'id': 'search-a', 'phase': 'search', 'candidate': 'a', 'status': 'passed',
                  'completed_at_epoch': 10, 'measurement': {'quality_passed': True, 'candidate_over_baseline': 0.8}}
        profile = {'id': 'profile-b', 'phase': 'profile', 'candidate': 'b', 'status': 'passed', 'completed_at_epoch': 12}
        self.assertIsNone(nominate([search, profile], search_stop=15))
        profile['candidate'] = 'a'
        self.assertEqual(nominate([search, profile], search_stop=15)['id'], 'a')
        profile['completed_at_epoch'] = 16
        self.assertIsNone(nominate([search, profile], search_stop=15))

    def test_failed_confirmation_never_allows_replacement_or_a_new_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            publish_new(state / 'claims/a.json', {'stop_at_epoch': 20, 'search_stop_at_epoch': 15})
            first = {'id': 'one', 'status': 'eligible', 'completed_at_epoch': 13}
            claim = claim_confirmation(state, 'a', first, now=17)
            self.assertEqual(claim['stop_at_epoch'], 20)
            with self.assertRaises(FileExistsError):
                claim_confirmation(state, 'a', {**first, 'id': 'two'}, now=18)


class RequestTests(unittest.TestCase):
    def supervisor(self, directory):
        root = Path(directory).resolve() / 'run'
        state = Path(directory).resolve() / 'state'
        publish_new(root / 'campaign/binding.json', {'assignment_id': 'one', 'measurement_protocol': 'campaign/protocol.json'})
        publish_new(root / 'campaign/protocol.json', {})
        publish_new(state / 'claims/one.json', {'search_stop_at_epoch': 100, 'stop_at_epoch': 120})
        (root / 'campaign/requests').mkdir()
        return AssignmentSupervisor(root, state)

    def test_unknown_request_is_consumed_before_execution_and_closes_search(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = self.supervisor(directory)
            candidate = owner.root / 'campaign/candidates/a'
            candidate.mkdir(parents=True)
            (candidate / 'candidate.py').write_text('def run(x): return x\n')
            publish_new(owner.root / 'campaign/requests/one.json', {'id': 'one', 'phase': 'search', 'candidate': 'a'})
            calls = []
            def failed(identity, phase, candidate):
                self.assertTrue((owner.folder / 'requests/one.json').is_file())
                calls.append(identity)
                raise TimeoutError('device outcome unknown')
            with patch('campaign.supervisor.time.time', return_value=10), patch.object(owner, 'execute', failed):
                with self.assertRaises(RuntimeError):
                    owner.service()
                owner.service()
            self.assertEqual(calls, ['one'])
            response = json.loads((owner.root / 'campaign/responses/one.json').read_text())
            self.assertEqual(response['status'], 'unknown')
            self.assertTrue(response['no_retry'])
            self.assertTrue(owner.search_closed)

    def test_profile_uses_controller_snapshot_after_author_rewrites_source(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = self.supervisor(directory)
            candidate = owner.root / 'campaign/candidates/a'
            candidate.mkdir(parents=True)
            (candidate / 'candidate.py').write_text('def run(x): return x\n')
            snapshot(owner.root / 'campaign/candidates', 'a', owner.folder / 'snapshots/a')
            (candidate / 'candidate.py').write_text('raise RuntimeError("changed")\n')
            owner.results = [{'id': 'search', 'phase': 'search', 'candidate': 'a', 'status': 'passed'}]
            publish_new(owner.root / 'campaign/requests/profile.json', {'id': 'profile', 'phase': 'profile', 'candidate': 'a'})
            def observe(identity, phase, candidate):
                self.assertEqual((owner.folder / 'snapshots' / candidate / 'candidate.py').read_text(),
                                 'def run(x): return x\n')
                return {'status': 'passed'}
            with patch('campaign.supervisor.time.time', return_value=10), patch.object(owner, 'execute', observe):
                owner.service()
            self.assertFalse(owner.search_closed)


class DeadlineTests(unittest.TestCase):
    def test_real_process_group_exits_when_original_deadline_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('campaign.processes.parent_death_signal'), (patch.object(
                    ProcessIdentity, 'observe', side_effect=lambda pid: ProcessIdentity(pid, 'cpu-test'))
                    if sys.platform != 'linux' else contextlib.nullcontext()):
                with self.assertRaises(TimeoutError):
                    run_bounded([sys.executable, '-c', 'import time; time.sleep(60)'],
                                cwd=root, env=dict(os.environ), stop=time.time() + 0.2, hard_stop=time.time() + 2,
                                log=root / 'log', process_record=root / 'process.json')
            process = json.loads((root / 'process.json').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(process['pid'], 0)



class AssayTests(unittest.TestCase):
    def test_child_event_or_profile_reports_cannot_establish_measurement(self):
        result = isolated_measurement_coverage()
        self.assertEqual(result['status'], 'coverage_missing')
        self.assertFalse(result['quality_passed'])
        self.assertNotIn('candidate_over_baseline', result)

    def test_measurement_rejects_missing_samples_and_nonfinite_values(self):
        for samples in ([{'baseline': 1, 'candidate': 1}] * 9,
                        [{'baseline': 1, 'candidate': math.nan}] * PAIRS):
            with self.assertRaises(ValueError):
                paired_summary([{'uuid': 'one', 'pairs': samples}])
        samples = [{'baseline': 2, 'candidate': 1}] * PAIRS
        result = paired_summary([{'uuid': 'one', 'pairs': samples}])
        self.assertTrue(result['quality_passed'])
        self.assertEqual(result['candidate_over_baseline'], 0.5)

    def test_source_gate_refuses_before_reading_allocation_or_provider(self):
        with self.assertRaisesRegex(ValueError, 'remain unqualified'):
            admit_live(Path('/absent/allocation.json'), Path('/absent/state'))


if __name__ == '__main__':
    unittest.main()
