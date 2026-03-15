"""NumPy-backed circular buffer for streaming float64 values.

O(1) append, O(1) chronological array view.  Used by FeatureBuilder
to maintain rolling windows over BarEvent fields (close, volume, etc.).
"""

import numpy as np


class RingBuffer:
    """Pre-allocated circular buffer for a single float64 scalar series."""

    __slots__ = ("_buf", "_capacity", "_head", "_count", "name")

    def __init__(self, capacity: int, name: str = "") -> None:
        self._buf = np.zeros(capacity, dtype=np.float64)
        self._capacity = capacity
        self._head = 0
        self._count = 0
        self.name = name

    def append(self, value: float) -> None:
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._capacity
        if self._count < self._capacity:
            self._count += 1

    def get_array(self) -> np.ndarray:
        """Return chronological view of buffered values."""
        if self._count == 0:
            return np.empty(0, dtype=np.float64)
        if self._count < self._capacity:
            return self._buf[: self._count].copy()
        # Full buffer — stitch tail..end + start..head
        return np.concatenate(
            (self._buf[self._head :], self._buf[: self._head])
        )

    def get_window(self, n: int) -> np.ndarray:
        """Return the last *n* values in chronological order.

        Alias kept for backward compat with ``last()``.
        """
        if n <= 0 or self._count == 0:
            return np.empty(0, dtype=np.float64)
        n = min(n, self._count)
        arr = self.get_array()
        return arr[-n:]

    def last(self, n: int = 1) -> np.ndarray:
        """Return the last *n* values in chronological order."""
        return self.get_window(n)

    def is_full(self) -> bool:
        return self._count >= self._capacity

    def __len__(self) -> int:
        return self._count

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._head = 0
        self._count = 0
