"""Bounded, explicitly commissioned Lynx gimbal controls."""
import hmac
import math
import os
import threading
import time


class GimbalUnavailable(RuntimeError):
    pass


class GimbalControl:
    """Use short rate pulses only after a verified device watchdog is configured."""
    def __init__(self, token=None, commissioned=False, watchdog_verified=False, connection_factory=None, operator_authorized=False):
        self.token = token or ''
        self.watchdog_verified = watchdog_verified
        self.ready = bool(self.token and ((commissioned and watchdog_verified) or operator_authorized))
        self.lock = threading.Lock()
        self.last_command = 0.0
        self.connection_factory = connection_factory

    @classmethod
    def from_environment(cls):
        return cls(os.getenv('SCOUT_GIMBAL_TOKEN'),
                   os.getenv('SCOUT_GIMBAL_COMMISSIONED') == '1',
                   os.getenv('SCOUT_GIMBAL_WATCHDOG_VERIFIED') == '1',
                   operator_authorized=os.getenv('SCOUT_GIMBAL_OPERATOR_AUTHORIZED') == '1')

    def status(self):
        return {'enabled': self.ready,
                'watchdog_verified': self.watchdog_verified,
                'reason': None if self.ready else 'Gimbal control requires commissioning and a verified device watchdog.'}

    def move(self, direction, token, x=None, y=None):
        if not self.ready:
            raise GimbalUnavailable('Gimbal controls are not commissioned')
        if not hmac.compare_digest(self.token.encode(), (token or '').encode()):
            raise PermissionError('Invalid operator token')
        rates = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        if x is None and y is None and direction in rates:
            pitch, yaw = rates[direction]
        elif direction is None and x is not None and y is not None:
            if not math.isfinite(x) or not math.isfinite(y) or abs(x) > 1 or abs(y) > 1:
                raise ValueError('Joystick axes must be finite values between -1 and 1')
            magnitude = math.hypot(x, y)
            if magnitude < .15 or magnitude > 1.001:
                raise ValueError('Joystick vector must be outside the dead zone and within the unit circle')
            pitch, yaw = y, x
        else:
            raise ValueError('Supply a direction or both joystick axes')
        if not self.lock.acquire(blocking=False):
            raise GimbalUnavailable('Gimbal is busy')
        try:
            now = time.monotonic()
            if now - self.last_command < .5:
                raise GimbalUnavailable('Wait before the next movement')
            self.last_command = now
            if self.connection_factory:
                link = self.connection_factory()
            else:
                os.environ.setdefault('MAVLINK20', '1')
                from pymavlink import mavutil
                link = mavutil.mavlink_connection('udpout:192.168.144.240:14566', source_system=246, source_component=190)
            try:
                link.mav.heartbeat_send(6, 8, 0, 0, 0)
                msg = link.recv_match(type='GIMBAL_DEVICE_ATTITUDE_STATUS', blocking=True, timeout=2)
                if msg is None or msg.get_srcSystem() != 1 or msg.get_srcComponent() != 154 or msg.failure_flags:
                    raise GimbalUnavailable('No healthy Lynx gimbal feedback')
                try:
                    self._send_rate(link, math.radians(2 * pitch), math.radians(2 * yaw))
                    time.sleep(.15)
                finally:
                    self._send_rate(link, 0.0, 0.0)
            finally:
                link.close()
        finally:
            self.lock.release()

    @staticmethod
    def _send_rate(link, pitch, yaw):
        link.mav.gimbal_device_set_attitude_send(1, 154, 0, [float('nan')] * 4, 0.0, pitch, yaw)
