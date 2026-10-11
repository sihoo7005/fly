import math
import tempfile
import unittest
from pathlib import Path

from model import Network, Parameters, Simulator


class ModelTests(unittest.TestCase):
    def test_region_counts_merge_and_weights_are_proportional(self):
        net = Network.from_rows([1, 2, 3], [(1, 2, 2), (1, 2, 3), (1, 3, 10)])
        self.assertEqual(list(net.synapses), [5, 10])
        sim = Simulator(net, Parameters(weight=0.03, threshold=100))
        sim.voltage[0] = 100
        self.assertEqual(sim.step({1: 0}), [1])
        self.assertEqual(sim.step(), [])
        self.assertAlmostEqual(sim.activity(2), 0.15)
        self.assertAlmostEqual(sim.activity(3), 0.30)

    def test_one_step_delay_and_synchronous_updates(self):
        net = Network.from_rows([3, 2, 1], [(1, 2, 100), (2, 3, 100)])
        sim = Simulator(net)
        self.assertEqual(sim.step({1: 1}), [1])
        self.assertEqual(sim.step(), [2])
        self.assertEqual(sim.step(), [3])
        self.assertEqual(sim.step(), [])

    def test_leak_is_applied_during_idle_steps(self):
        sim = Simulator(Network.from_rows([1], []), Parameters(tau_ms=10))
        sim.step({1: 0.5})
        sim.step()
        self.assertAlmostEqual(sim.activity(1), 0.5 * math.exp(-0.1))
        self.assertEqual(sim.step({1: 0.5}), [])
        self.assertAlmostEqual(sim.activity(1), 0.5 * math.exp(-0.2) + 0.5)

    def test_inhibition_cancels_external_input(self):
        sim = Simulator(Network.from_rows([1, 2], [(1, 2, 100)]), inhibitory=[1])
        self.assertEqual(sim.step({1: 1}), [1])
        self.assertEqual(sim.step({2: 1}), [])
        self.assertEqual(sim.activity(2), 0)

    def test_refractory_interval(self):
        sim = Simulator(Network.from_rows([1], []), Parameters(refractory_ms=2))
        self.assertEqual(sim.step({1: 1}), [1])
        self.assertEqual(sim.step({1: 1}), [])
        self.assertEqual(sim.step({1: 1}), [1])

    def test_cache_preserves_large_ids_and_counts(self):
        roots = [720575940621014292, 720575940621939071]
        net = Network.from_rows(roots, [(roots[0], roots[1], 123)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "network.bin"
            net.save(path)
            restored = Network.load(path)
            self.assertEqual(list(restored.ids), roots)
            self.assertEqual(list(restored.synapses), [123])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "size"):
                Network.load(path)

    def test_rejects_unknown_endpoints_and_bad_parameters(self):
        with self.assertRaises(ValueError):
            Network.from_rows([1], [(1, 2, 1)])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                Parameters(weight=value)
        sim = Simulator(Network.from_rows([1], []))
        with self.assertRaises(ValueError):
            sim.step({2: 1})
        self.assertEqual(sim.time_step, 0)


if __name__ == "__main__":
    unittest.main()
