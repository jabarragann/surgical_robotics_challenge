from __future__ import annotations
from typing import Union
from typing import Type
from numpy.linalg import norm, svd, det
import numpy as np
import logging

log = logging.getLogger(__name__)


class Frame:
    def __init__(self, r: np.ndarray, p: np.ndarray) -> None:
        """Create a frame with rotation `r` and translation `p`.
        Args:
            r (np.ndarray): Rotation.
            p (np.ndarray): translation.
        """
        self.r = np.array(r)
        self.p = np.array(p).reshape((3, 1))

    def __array__(self):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = self.r
        out[:3, 3] = self.p.squeeze()
        return out

    def __str__(self):
        return np.array_str(np.array(self), precision=4, suppress_small=True)

    def inv(self) -> Frame:
        return Frame(self.r.T, -(self.r.T @ self.p))

    def __matmul__(self, other: Union[np.ndarray, Frame]) -> Frame:
        """[summary]
        Args:
            other (Union[np.ndarray, Frame]): [description]
        Returns:
            Frame: [description]
        """

        if isinstance(other, np.ndarray):
            if len(other.shape) == 1:
                assert other.shape == (3,), "Dimension error, points array should have a shape (3,)"
                other = other.reshape(3, 1)
            elif len(other) > 2:
                assert (
                    other.shape[0] == 3
                ), "Dimension error, points array should have a shape (3,N), where `N` is the number points."

            return (self.r @ other) + self.p
        elif isinstance(other, Frame):
            return Frame(self.r @ other.r, self.r @ other.p + self.p)
        else:
            raise TypeError

    @classmethod
    def identity(cls: Type[Frame]) -> Type[Frame]:
        return Frame(np.identity(3), np.zeros(3))
