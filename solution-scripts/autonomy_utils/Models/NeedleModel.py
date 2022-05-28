import numpy as np


class NeedleModel:

    radius = 0.1018

    def __init__(self) -> None:
        self.radius = 0.1018

    def get_tip_tail_pose(self) -> np.ndarray:
        """Get 3D positions of the tip and the tail w.r.t the needle local frame.
        The needle is parametrized by a circle of radius `radius`. The tip of the needle is
        located at the angle pi/3 and the tail at  pi.

        Returns:
            np.ndarray: [description]
        """
        # Get 3D position of the tip and tail
        theta = np.array([np.pi / 3 - 2.5 * np.pi / 180, np.pi + np.pi / 180]).reshape((2, 1))
        needle_salient = self.radius * np.hstack(
            (np.cos(theta), np.sin(theta), theta * 0, np.ones((2, 1)) / self.radius)
        )
        return needle_salient

    def sample_3d_pts(self, N: int) -> np.ndarray:
        """Sample `N` 3D points of the needle on the needle local coordinate frame

        Args:
            N (int): [description]

        Returns:
            np.ndarray: Needle 3D points
        """

        theta = np.linspace(np.pi / 3, np.pi, num=N).reshape((-1, 1))
        needle_salient = self.radius * np.hstack((np.cos(theta), np.sin(theta), theta * 0))
        return needle_salient
