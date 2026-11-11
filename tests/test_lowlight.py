import unittest
import numpy as np
from scout.lowlight import LowLightPolicy


class LowLightPolicyTest(unittest.TestCase):
    def test_dark_fallback_and_hysteretic_recovery(self):
        policy = LowLightPolicy(dark_seconds=5, probe_interval=30, probe_seconds=5)
        dark = np.zeros((90, 160, 3), np.uint8)
        bright = np.full((90, 160, 3), 80, np.uint8)
        self.assertIsNone(policy.update(dark, 0))
        self.assertIsNone(policy.update(dark, 4))
        self.assertEqual(policy.update(dark, 5), 'thermal')
        policy.switched('thermal', 5)
        self.assertIsNone(policy.update(bright, 34))
        self.assertEqual(policy.update(bright, 35), 'probe')
        policy.switched('probe', 35)
        self.assertIsNone(policy.update(bright, 39))
        self.assertEqual(policy.update(bright, 40), 'visible')

    def test_brief_darkness_does_not_trigger_switch(self):
        policy = LowLightPolicy()
        dark = np.zeros((90, 160, 3), np.uint8)
        light = np.full((90, 160, 3), 100, np.uint8)
        self.assertIsNone(policy.update(dark, 0))
        self.assertIsNone(policy.update(light, 4))
        self.assertIsNone(policy.update(dark, 6))
        self.assertIsNone(policy.update(dark, 10))

    def test_dark_probe_returns_to_thermal(self):
        policy = LowLightPolicy()
        policy.switched('probe', 10)
        dark = np.zeros((90, 160, 3), np.uint8)
        self.assertEqual(policy.update(dark, 15), 'thermal')


if __name__ == '__main__':
    unittest.main()
