"""Observable correctness gates and full/partial result boundaries on CPU."""
import json
from pathlib import Path
import tempfile
import unittest

import c550bench
c550bench.upstream()
import torch
from sol_execbench.core.data import Definition, ToleranceSpec


def definition(dtype="float32"):
    return Definition.model_validate({
        "name": "synthetic_identity", "axes": {"n": {"type": "var"}},
        "inputs": {"x": {"shape": ["n"], "dtype": dtype}},
        "outputs": {"y": {"shape": ["n"], "dtype": dtype}},
        "reference": "import torch\ndef run(x):\n    return x.clone()\n",
    })


class CorrectnessTests(unittest.TestCase):
    def compare(self, actual, expected, **kwargs):
        return c550bench.compare_outputs(actual, expected, definition(str(expected.dtype).split(".")[-1]),
                                       ToleranceSpec(**kwargs), {"n": expected.numel()})

    def test_float_pass_and_reject_bad_candidate(self):
        expected = torch.tensor([1., -2., 3.])
        self.assertTrue(self.compare(expected.clone(), expected)["passed"])
        self.assertFalse(self.compare(torch.zeros_like(expected), expected)["passed"])

    def test_discrete_outputs_are_exact_before_float_conversion(self):
        expected = torch.tensor([2**30, 2**30+1], dtype=torch.int64)
        wrong = torch.tensor([2**30+1, 2**30+1], dtype=torch.int64)
        self.assertFalse(self.compare(wrong, expected)["passed"])
        self.assertTrue(self.compare(expected, expected)["passed"])

    def test_nan_and_positive_infinity_always_fail(self):
        for value in [float("nan"), float("inf")]:
            x = torch.tensor([value])
            self.assertFalse(self.compare(x, x)["passed"])

    def test_negative_infinity_needs_upstream_opt_in(self):
        x = torch.tensor([float("-inf"), 1.])
        self.assertFalse(self.compare(x, x)["passed"])
        self.assertTrue(self.compare(x, x, allow_negative_inf=True)["passed"])

    def test_error_cap_rejects_rare_outlier(self):
        expected = torch.ones(100)
        wrong = expected.clone(); wrong[0] = 1000
        self.assertFalse(self.compare(wrong, expected, required_matched_ratio=0.98, max_error_cap=1)["passed"])

    def test_shape_dtype_and_reference_schema(self):
        x = torch.ones(4)
        self.assertFalse(self.compare(x.half(), x)["passed"])
        self.assertFalse(self.compare(x.reshape(2,2), x)["passed"])
        report = c550bench.compare_outputs(x, x, definition(), ToleranceSpec(), {"n": 8})
        self.assertEqual(report["outputs"]["y"]["reason"], "invalid_reference_contract")

    def test_output_dictionary_requires_all_and_only_declared_names(self):
        x = torch.ones(2)
        for actual in [{}, {"y": x, "extra": x}]:
            self.assertFalse(self.compare(actual, x)["passed"])

    def test_input_clone_prevents_reference_mutation_of_candidate_input(self):
        x = torch.ones(3)
        copy = c550bench.cloned_inputs([x, 0.5])
        copy[0].zero_()
        self.assertTrue(torch.equal(x, torch.ones(3)))
        self.assertEqual(copy[1], 0.5)

    def test_raw_tolerance_typo_retains_upstream_effective_default(self):
        effective = ToleranceSpec.model_validate({"required_match_ratio": 0.98})
        self.assertEqual(effective.required_matched_ratio, 0.99)


class RuntimeIdentityTests(unittest.TestCase):
    def test_guard_refuses_nvidia_or_generic_cuda_compatibility(self):
        with self.assertRaises(RuntimeError):
            c550bench.validate_c550_identity("NVIDIA H100", None, "MetaX C550")
        with self.assertRaises(RuntimeError):
            c550bench.validate_c550_identity("MetaX C550", None, "MetaX C550")
        c550bench.validate_c550_identity("MetaX C550", "3.8.0.4.c600u", "MetaX C550")


class RunnerTests(unittest.TestCase):
    def test_fresh_rounds_and_partial_result_never_qualify_device(self):
        original_root = c550bench.ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c550bench.ROOT = root
            try:
                (root/'sources.lock.json').write_text(json.dumps({"synthetic": True}))
                (root/'suite.json').write_text(json.dumps({"seed":200,"target_arch":"metax_c550","target_device_name":"MetaX C550","correctness_rounds":10}))
                task = {"id":"L1/synthetic_identity","smoke_workload_uuid":"fixture"}
                problem = root/'.data/benchmark'/task['id'];problem.mkdir(parents=True)
                d = definition().model_dump(mode="json")
                (problem/'definition.json').write_text(json.dumps(d))
                (problem/'reference.py').write_text(d['reference'])
                (problem/'workload.jsonl').write_text(json.dumps({"uuid":"fixture","axes":{"n":16},"inputs":{"x":{"type":"random"}}})+'\n')
                candidate = root/'candidate.py'
                candidate.write_text('previous = None\ndef run(x):\n    global previous\n    if previous is not None and x.equal(previous):\n        raise RuntimeError("inputs reused")\n    previous = x.clone()\n    return x.clone()\n')
                result = c550bench.check_problem(task,device='cpu',candidate_path=candidate,symbol='run',reference_selfcheck=False,
                    workload_scope='all',rounds=3,output=root/'pass.json',seed=200,threads=1)
                self.assertEqual(result['status'],'passed')
                self.assertEqual(len(result['cases']),3)
                self.assertFalse(result['full_device_correctness'])
                candidate.write_text('calls = 0\ndef run(x):\n    global calls\n    calls += 1\n    return x.clone() if calls == 1 else x * 0\n')
                result = c550bench.check_problem(task,device='cpu',candidate_path=candidate,symbol='run',reference_selfcheck=False,
                    workload_scope='all',rounds=3,output=root/'fail.json',seed=200,threads=1)
                self.assertEqual(result['status'],'failed')
                self.assertEqual(len(result['cases']),2)
            finally:
                c550bench.ROOT = original_root


if __name__ == '__main__':
    unittest.main()
