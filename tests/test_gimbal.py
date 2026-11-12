import unittest
import threading
from unittest.mock import patch
from scout.gimbal import GimbalControl, GimbalUnavailable
from scout.mavlink_transport import lock as mavlink_lock


class Message:
    failure_flags = 0
    flags = 32
    def get_srcSystem(self): return 1
    def get_srcComponent(self): return 154


class Link:
    def __init__(self):
        self.mav = self
        self.commands = []
        self.closed = False
    def heartbeat_send(self, *args): pass
    def recv_match(self, **kwargs): return Message()
    def gimbal_device_set_attitude_send(self, *args): self.commands.append(args)
    def close(self): self.closed = True


class GimbalTest(unittest.TestCase):
    def test_disabled_by_default(self):
        control = GimbalControl()
        self.assertFalse(control.status()['enabled'])
        with self.assertRaises(GimbalUnavailable): control.move('up', '')

    def test_token_and_direction_required(self):
        control = GimbalControl('token', True, True, Link)
        with self.assertRaises(PermissionError): control.move('up', 'wrong')
        with self.assertRaises(ValueError): control.move('invalid', 'token')

    @patch('scout.gimbal.time.sleep', return_value=None)
    def test_bounded_pulse_stops_and_closes(self, sleep):
        link = Link()
        control = GimbalControl('token', True, True, lambda: link)
        control.move('up', 'token')
        self.assertEqual(len(link.commands), 2)
        self.assertEqual(link.commands[0][2], 32)
        self.assertEqual(link.commands[1][2], 32)
        self.assertLess(link.commands[0][5], 0)
        self.assertEqual(link.commands[1][4:7], (0.0, 0.0, 0.0))
        self.assertTrue(link.closed)

    @patch('scout.gimbal.time.sleep', side_effect=RuntimeError('test interruption'))
    def test_stop_sent_even_when_pulse_interrupted(self, sleep):
        link = Link()
        control = GimbalControl('token', True, True, lambda: link)
        with self.assertRaises(RuntimeError): control.move('right', 'token')
        self.assertEqual(link.commands[-1][4:7], (0.0, 0.0, 0.0))
        self.assertTrue(link.closed)

    def test_watchdog_gate(self):
        control = GimbalControl('token', True, False, Link)
        self.assertFalse(control.status()['enabled'])
        with self.assertRaises(GimbalUnavailable): control.move('up', 'token')

    def test_operator_authorization_does_not_claim_watchdog_verified(self):
        control = GimbalControl('token', operator_authorized=True, connection_factory=Link)
        self.assertTrue(control.status()['enabled'])
        self.assertFalse(control.status()['watchdog_verified'])

    @patch('scout.gimbal.time.sleep', return_value=None)
    def test_diagonal_joystick_is_bounded_and_stops(self, sleep):
        link = Link()
        control = GimbalControl('token', True, True, lambda: link)
        control.move(None, 'token', .6, -.8)
        self.assertLess(link.commands[0][5], 0)
        self.assertGreater(link.commands[0][6], 0)
        self.assertAlmostEqual(link.commands[0][5], -0.8 * 2 * 3.141592653589793 / 180)
        self.assertEqual(link.commands[-1][4:7], (0.0, 0.0, 0.0))

    def test_invalid_joystick_vectors_are_rejected(self):
        control = GimbalControl('token', True, True, Link)
        for x, y in ((float('nan'), 0), (0, 0), (1, 1), (1.1, 0)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                control.move(None, 'token', x, y)

    def test_busy_telemetry_does_not_queue_a_movement(self):
        acquired, release = threading.Event(), threading.Event()
        def hold():
            with mavlink_lock:
                acquired.set()
                release.wait(2)
        thread = threading.Thread(target=hold)
        thread.start()
        self.assertTrue(acquired.wait(1))
        link = Link()
        try:
            control = GimbalControl('token', True, True, lambda: link)
            with self.assertRaises(GimbalUnavailable):
                control.move('up', 'token')
            self.assertEqual(link.commands, [])
        finally:
            release.set()
            thread.join(1)

    @patch('scout.gimbal.time.sleep', return_value=None)
    @patch('pymavlink.mavutil.mavlink_connection')
    def test_gremsy_sdk_command_identity(self, connect, sleep):
        connect.return_value = Link()
        control = GimbalControl('token', True, True)
        control.move('up', 'token')
        self.assertEqual(connect.call_args.kwargs['source_system'], 1)
        self.assertEqual(connect.call_args.kwargs['source_component'], 193)


if __name__ == '__main__': unittest.main()
