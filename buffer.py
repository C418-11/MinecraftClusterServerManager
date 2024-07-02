# -*- coding: utf-8 -*-
# cython: language_level = 3

__author__ = "C418____11 <553515788@qq.com>"
__version__ = "0.0.1Dev"

import threading
from collections import deque
from typing import Callable


class StringBuffer:
    def __init__(self, max_line: int = 1024):
        self._buffer: deque[str] = deque(maxlen=max_line)

        self._write_lock = threading.Lock()
        self._callback_lock = threading.RLock()
        self._callbacks: list[Callable[[str], None]] = []

    @property
    def buffer(self) -> deque[str]:
        return self._buffer.copy()

    def read(self) -> str:
        with self._write_lock:
            return "".join(self._buffer)

    def write(self, value: str) -> None:
        def _process_line(txt: str) -> None:
            if len(self._buffer) < 1:
                self._buffer.append(txt)
                return
            if self._buffer[-1].endswith("\n"):
                self._buffer.append(txt)
                return
            self._buffer[-1] += txt

        for line in value.splitlines(True):
            with self._write_lock:
                _process_line(line)

        for callback in self._callbacks:
            with self._callback_lock:
                callback(value)

    def register(self, callback: Callable[[str], None]):
        with self._callback_lock:
            self._callbacks.append(callback)
            callback(self.read())

    def unregister(self, callback: Callable[[str], None]):
        with self._callback_lock:
            self._callbacks.remove(callback)

    def isRegistered(self, callback: Callable[[str], None]) -> bool:
        with self._callback_lock:
            return callback in self._callbacks

    def clear(self) -> None:
        with self._write_lock:
            self._buffer.clear()


__all__ = ("StringBuffer",)
