"""Read-only Lynx gimbal attitude and raw encoder telemetry."""
import math
import os
import threading
import time

from .mavlink_transport import lock as mavlink_lock


class GimbalTelemetry:
    def __init__(self, connection_factory=None):
        self.connection_factory = connection_factory
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.attitude = None
        self.encoder = None
        self.attitude_at = None
        self.encoder_at = None
        self.encoder_mode = None
        self.encoder_mode_at = None
        self.error = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name='gimbal-telemetry')
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            age = None if self.attitude_at is None else now - self.attitude_at
            encoder_age = None if self.encoder_at is None else now - self.encoder_at
            mode_age = None if self.encoder_mode_at is None else now - self.encoder_mode_at
            return {'status': 'live' if age is not None and age < 3 else 'unavailable',
                    'attitude': dict(self.attitude) if age is not None and age < 3 else None,
                    'encoder': dict(self.encoder) if encoder_age is not None and encoder_age < 3 else None,
                    'encoder_units': ('centidegrees' if self.encoder_mode == 0 else 'raw') if mode_age is not None and mode_age < 10 else None,
                    'age_ms': round(age * 1000) if age is not None else None,
                    'error': self.error}

    def _accept(self, msg, now):
        if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 154:
            return
        if msg.get_type() == 'MOUNT_ORIENTATION':
            vals = (msg.roll, msg.pitch, msg.yaw, msg.yaw_absolute)
            if not all(math.isfinite(v) for v in vals):
                return
            with self.lock:
                self.attitude = dict(roll=vals[0], pitch=vals[1], yaw=vals[2], yaw_absolute=vals[3])
                self.attitude_at, self.error = now, None
        elif msg.get_type() == 'MOUNT_STATUS':
            with self.lock:
                self.encoder = dict(pitch=msg.pointing_a, roll=msg.pointing_b, yaw=msg.pointing_c)
                self.encoder_at = now
        elif msg.get_type() == 'PARAM_VALUE' and msg.param_id == 'MAV_TS_ENCNT':
            if math.isfinite(msg.param_value) and msg.param_value in (0, 1):
                with self.lock:
                    self.encoder_mode, self.encoder_mode_at = int(msg.param_value), now

    def _run(self):
        link = None
        try:
            while not self.stop_event.is_set():
                try:
                    with mavlink_lock:
                        if link is None:
                            if self.connection_factory:
                                link = self.connection_factory()
                            else:
                                os.environ.setdefault('MAVLINK20', '1')
                                from pymavlink import mavutil
                                link = mavutil.mavlink_connection('udpout:192.168.144.240:14566', source_system=1, source_component=193)
                        # Discard buffered messages so a disconnected camera cannot look live.
                        for _ in range(256):
                            if link.recv_match(blocking=False) is None:
                                break
                        link.mav.heartbeat_send(6, 8, 0, 0, 0)
                        link.mav.param_request_read_send(1, 154, b'MAV_TS_ENCNT', -1)
                        link.mav.command_long_send(1, 154, 512, 0, 158, 0, 0, 0, 0, 0, 0)
                        end = time.monotonic() + .65
                        seen = set()
                        while not self.stop_event.is_set() and time.monotonic() < end and len(seen) < 3:
                            msg = link.recv_match(type=['MOUNT_ORIENTATION', 'MOUNT_STATUS', 'PARAM_VALUE'], blocking=True, timeout=.2)
                            if msg and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 154:
                                self._accept(msg, time.monotonic())
                                if msg.get_type() != 'PARAM_VALUE' or msg.param_id == 'MAV_TS_ENCNT':
                                    seen.add(msg.get_type())
                except Exception as exc:
                    with self.lock:
                        self.error = str(exc)
                    if link is not None:
                        link.close()
                        link = None
                self.stop_event.wait(1)
        finally:
            if link is not None:
                link.close()
