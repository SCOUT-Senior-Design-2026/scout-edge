"""Serialize request/reply exchanges sharing the Lynx UDP return route."""
from functools import wraps
from threading import RLock

lock = RLock()


def serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with lock:
            return function(*args, **kwargs)
    return wrapped
