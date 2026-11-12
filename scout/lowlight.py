"""Low-light source policy and Gremsy camera view selection."""
import os
import time

import cv2
import numpy as np

from .mavlink_transport import serialized


class LowLightPolicy:
    """Use a timed visible probe to recover when only one output is available."""
    def __init__(self, dark=22, bright=35, dark_seconds=5, probe_interval=30,
                 probe_seconds=5):
        if not 0 <= dark < bright <= 255:
            raise ValueError('low-light thresholds must satisfy 0 <= dark < bright <= 255')
        self.dark, self.bright = dark, bright
        self.dark_seconds, self.probe_interval, self.probe_seconds = dark_seconds, probe_interval, probe_seconds
        self.mode = 'visible'
        self.dark_since = None
        self.changed_at = None
        self.next_sample = 0
        self.luminance = None

    def update(self, frame, now):
        if self.mode == 'thermal':
            return 'probe' if now - self.changed_at >= self.probe_interval else None
        if now < self.next_sample:
            return None
        self.next_sample = now + 1
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        self.luminance = float(np.median(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)))
        if self.mode == 'probe':
            if now - self.changed_at < self.probe_seconds:
                return None
            return 'visible' if self.luminance >= self.bright else 'thermal'
        if self.luminance < self.dark:
            if self.dark_since is None:
                self.dark_since = now
            if now - self.dark_since >= self.dark_seconds:
                return 'thermal'
        else:
            self.dark_since = None
        return None

    def switched(self, mode, now):
        self.mode, self.changed_at = mode, now
        self.dark_since = None
        self.next_sample = now + 1


class GremsyView:
    """Change the main stream view via acknowledged MAVLink camera parameters."""
    @serialized
    def __init__(self, address='192.168.144.240:14566'):
        os.environ.setdefault('MAVLINK20', '1')
        from pymavlink import mavutil
        self.mavutil = mavutil
        self.link = mavutil.mavlink_connection('udpout:' + address, source_system=246, source_component=190, dialect='common')
        self.link.mav.heartbeat_send(6, 8, 0, 0, 0)
        self.link.mav.param_ext_request_read_send(1, 101, b'C_SOURCE', -1)
        end = time.monotonic() + 3
        self.original = None
        while time.monotonic() < end:
            msg = self.link.recv_match(type='PARAM_EXT_VALUE', blocking=True, timeout=.3)
            if msg and msg.param_id == 'C_SOURCE':
                self.original = int.from_bytes(msg.get_payload()[24:28], 'little')
                break
        if self.original not in (0, 1):
            self.link.close()
            raise RuntimeError('Expected a visible Lynx source before enabling automatic switching')

    @serialized
    def set_mode(self, thermal):
        self.link.mav.heartbeat_send(6, 8, 0, 0, 0)
        value = 2 if thermal else self.original
        self.link.mav.param_ext_set_send(1, 101, b'C_SOURCE', bytes([value]) + bytes(127), 5)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            msg = self.link.recv_match(type='PARAM_EXT_ACK', blocking=True, timeout=.3)
            if msg and msg.param_id == 'C_SOURCE':
                if msg.param_result != 0:
                    raise RuntimeError('Camera rejected view switch')
                return
        raise RuntimeError('Camera view switch was not acknowledged')

    def close(self):
        self.link.close()
