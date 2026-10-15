import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

import c550bench


class ModuleLoadingTests(unittest.TestCase):
    def setUp(self):
        self.name = 'c550bench_test_module_loading'
        previous = {self.name: sys.modules[self.name]} if self.name in sys.modules else {}
        def restore():
            sys.modules.pop(self.name, None)
            sys.modules.update(previous)
        self.addCleanup(restore)
        sys.modules.pop(self.name, None)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def source(self, filename, text):
        path = self.root / filename
        path.write_text(text)
        return path

    def test_importable_during_execution_and_after_return(self):
        path = self.source('identity.py', 'import importlib\nidentity = importlib.import_module(__name__)\n')
        module = c550bench.load_module(path, self.name)
        self.assertIs(module.identity, module)
        self.assertIs(importlib.import_module(self.name), module)

    def test_successful_same_name_load_executes_fresh_source_and_replaces_identity(self):
        first = c550bench.load_module(self.source('first.py', 'value = 1\n'), self.name)
        second = c550bench.load_module(self.source('second.py', 'value = 2\n'), self.name)
        self.assertIsNot(first, second)
        self.assertEqual(first.value, 1)
        self.assertEqual(second.value, 2)
        self.assertIs(importlib.import_module(self.name), second)

    def test_failed_execution_removes_new_identity(self):
        for exception in (RuntimeError, KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception):
                path = self.source('failure.py', f'raise {exception.__name__}("original failure")\n')
                with self.assertRaisesRegex(exception, 'original failure'):
                    c550bench.load_module(path, self.name)
                self.assertNotIn(self.name, sys.modules)

    def test_failed_execution_restores_exact_previous_value(self):
        for previous in (types.ModuleType(self.name), None):
            for exception in (RuntimeError, KeyboardInterrupt, SystemExit):
                with self.subTest(previous=previous, exception=exception):
                    sys.modules[self.name] = previous
                    path = self.source('failure.py', f'raise {exception.__name__}("original failure")\n')
                    with self.assertRaisesRegex(exception, 'original failure'):
                        c550bench.load_module(path, self.name)
                    self.assertIn(self.name, sys.modules)
                    self.assertIs(sys.modules[self.name], previous)
