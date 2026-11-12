import unittest
from unittest.mock import patch
from scout.gimbal_telemetry import GimbalTelemetry


class Message:
    roll, pitch, yaw, yaw_absolute = 1.0, 179.3, 1.8, 120.0
    pointing_a, pointing_b, pointing_c = 0, 2, 1
    def __init__(self, kind, system=1, component=154):
        self.kind, self.system, self.component = kind, system, component
    def get_srcSystem(self): return self.system
    def get_srcComponent(self): return self.component
    def get_type(self): return self.kind


class GimbalTelemetryTest(unittest.TestCase):
    @patch('scout.gimbal_telemetry.time.monotonic', return_value=10.5)
    def test_reports_fresh_attitude_and_raw_encoder(self, clock):
        sensor = GimbalTelemetry()
        sensor._accept(Message('MOUNT_ORIENTATION'), 10)
        sensor._accept(Message('MOUNT_STATUS'), 10.1)
        result = sensor.snapshot()
        self.assertEqual(result['status'], 'live')
        self.assertEqual(result['attitude']['pitch'], 179.3)
        self.assertEqual(result['encoder']['yaw'], 1)
        self.assertEqual(result['age_ms'], 500)

    @patch('scout.gimbal_telemetry.time.monotonic', return_value=14)
    def test_rejects_wrong_source_and_drops_stale_data(self, clock):
        sensor = GimbalTelemetry()
        sensor._accept(Message('MOUNT_ORIENTATION', component=1), 13)
        self.assertEqual(sensor.snapshot()['status'], 'unavailable')
        sensor._accept(Message('MOUNT_ORIENTATION'), 10)
        self.assertIsNone(sensor.snapshot()['attitude'])
        self.assertEqual(sensor.snapshot()['status'], 'unavailable')


if __name__ == '__main__': unittest.main()
