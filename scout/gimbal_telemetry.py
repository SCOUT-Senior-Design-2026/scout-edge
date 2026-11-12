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
            return {'status': 'live' if age is not None and age < 3 else 'unavailable',
                    'attitude': dict(self.attitude) if age is not None and age < 3 else None,
                    'encoder': dict(self.encoder) if encoder_age is not None and encoder_age < 3 else None,
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

    def _run(self):
        while not self.stop_event.is_set():
            link = None
            try:
                with mavlink_lock:
                    if self.connection_factory:
                        link = self.connection_factory()
                    else:
                        os.environ.setdefault('MAVLINK20', '1')
                        from pymavlink import mavutil
                        link = mavutil.mavlink_connection('udpout:192.168.144.240:14566', source_system=244, source_component=190)
                    link.mav.heartbeat_send(6, 8, 0, 0, 0)
                    end = time.monotonic() + .3
                    seen = set()
                    while not self.stop_event.is_set() and time.monotonic() < end and len(seen) < 2:
                        msg = link.recv_match(type=['MOUNT_ORIENTATION', 'MOUNT_STATUS'], blocking=True, timeout=.2)
                        if msg and msg.get_srcSystem() == 1 and msg.get_srcComponent() == 154:
                            self._accept(msg, time.monotonic())
                            seen.add(msg.get_type())
                    link.close()
                    link = None
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            finally:
                if link is not None:
                    link.close()
            self.stop_event.wait(1)
