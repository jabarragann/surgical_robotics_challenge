import numpy as np


def find_closest_rotation(matrix: np.ndarray) -> np.ndarray:
    """Find closest rotation to the input matrix
    Algorithm from https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix/23083722

    Args:
        matrix (np.ndarray): (3x3) rotation matrix

    Returns:
        np.ndarray: [description]
    """

    # Method 1
    # def normalize(x):
    #     return x/np.sqrt(x.dot(x))

    # x = matrix[:3,0]
    # y = matrix[:3,1]
    # z = matrix[:3,2]

    # error = x.dot(z)
    # new_x = x-(error/2)*z
    # new_z = z-(error/2)*x
    # new_y = np.cross(new_z,new_x)

    # new_matrix = np.zeros_like(matrix)
    # new_matrix[:3,0] = normalize(new_x)
    # new_matrix[:3,1] = normalize(new_y)
    # new_matrix[:3,2] = normalize(new_z)

    # Method 2
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    new_matrix = u @ vh

    assert np.isclose(np.linalg.det(new_matrix), 1.0), "Normalization procedure failed"

    return new_matrix