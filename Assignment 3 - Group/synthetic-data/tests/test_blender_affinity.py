"""CPU affinity tests: python tests/test_blender_affinity.py."""
import importlib.util
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('blenderrun', Path(__file__).parents[1] / 'src/sam3dvid/blenderrun.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestAffinity(unittest.TestCase):
    def test_default_command_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(module.limit_blender_cpus(['blender']), ['blender'])

    def test_inherited_affinity_slice_and_stable_worker(self):
        with patch.dict(os.environ, {'SAM3DVID_BLENDER_CPUS': '2'}), \
             patch.object(module.shutil, 'which', return_value='/usr/bin/taskset'), \
             patch.object(module.os, 'sched_getaffinity', return_value={4, 5, 6, 7}, create=True), \
             patch.object(module, '_CPU_SLOT', threading.local()):
            first = module.limit_blender_cpus(['blender'])
            second = module.limit_blender_cpus(['blender'])
            self.assertEqual(first, second)
            self.assertEqual(first[:2], ['/usr/bin/taskset', '-c'])
            self.assertEqual(len(first[2].split(',')), 2)
            self.assertTrue(set(map(int, first[2].split(','))) <= {4, 5, 6, 7})

    def test_invalid_limit(self):
        with patch.dict(os.environ, {'SAM3DVID_BLENDER_CPUS': '0'}):
            with self.assertRaises(ValueError):
                module.limit_blender_cpus(['blender'])


if __name__ == '__main__':
    unittest.main()
