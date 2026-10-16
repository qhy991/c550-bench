from pathlib import Path
import unittest

import torch

import c550bench


class LibraryRadixReferenceTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'examples/l1_058_torch_stable_boundaries.py'
        self.candidate = c550bench.load_module(path, 'c550bench_test_library_radix_reference')

    def assert_reference(self, topk_idx):
        original = topk_idx.clone()
        values = topk_idx.reshape(-1).tolist()
        expected_positions = sorted(range(len(values)), key=values.__getitem__)
        expected_offsets = [sum(value < expert for value in values) for expert in range(257)]

        result = self.candidate.run(topk_idx)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        positions, offsets = result
        for output in result:
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.dtype, torch.int32)
            self.assertEqual(output.device, topk_idx.device)
        self.assertEqual(positions.shape, (len(values),))
        self.assertEqual(offsets.shape, (257,))
        self.assertEqual(positions.tolist(), expected_positions)
        self.assertEqual(offsets.tolist(), expected_offsets)
        self.assertTrue(torch.equal(topk_idx, original))

    def test_stable_ties_and_empty_buckets(self):
        topk_idx = torch.tensor([2, 0, 2, 255, 0, 3, 3, 0], dtype=torch.int32).reshape(1, 1, 8)
        self.assert_reference(topk_idx)

    def test_all_zero_and_all_255_endpoints(self):
        for expert in (0, 255):
            with self.subTest(expert=expert):
                self.assert_reference(torch.full((1, 3, 8), expert, dtype=torch.int32))

    def test_original_shaped_noncontiguous_tail_preserves_input(self):
        values = [(index * 37 + index // 11) % 256 for index in range(8 * 288 * 16)]
        storage = torch.tensor(values, dtype=torch.int32).reshape(8, 288, 16)
        topk_idx = storage[:, :, ::2]
        self.assertEqual(topk_idx.shape, (8, 288, 8))
        self.assertFalse(topk_idx.is_contiguous())
        self.assert_reference(topk_idx)
