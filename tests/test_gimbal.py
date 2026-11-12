import unittest
from unittest.mock import patch
from scout.gimbal import GimbalControl, GimbalUnavailable


class Message:
    failure_flags = 0
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


if __name__ == '__main__': unittest.main()
