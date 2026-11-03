import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hardware_probe import gpu_numa


class HardwareProbeTest(unittest.TestCase):
    def topology_case(self, affinity):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            device = root / 'device'
            device.mkdir()
            (device / 'numa_node').write_text('-1')
            nodes = root / 'nodes'
            for node, cpus in [(0, '0-11,24-35'), (1, '12-23,36-47')]:
                folder = nodes / f'node{node}'
                folder.mkdir(parents=True)
                (folder / 'cpulist').write_text(cpus)

            def path(value):
                return nodes if str(value) == '/sys/devices/system/node' else device

            with patch('hardware_probe.Path', side_effect=path), patch(
                'hardware_probe.subprocess.check_output', side_effect=[
                    '0, test-uuid\n', f'GPU0 X {affinity} 0 N/A\n']):
                return gpu_numa('0000:01:00.0', 'test-uuid')

    def test_topology_fallback(self):
        self.assertEqual(self.topology_case('0-11,24-35'), 0)

    def test_reject_cross_numa_affinity(self):
        with self.assertRaises(RuntimeError):
            self.topology_case('0-47')


if __name__ == '__main__':
    unittest.main()
