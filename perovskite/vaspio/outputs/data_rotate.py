from functools import lru_cache
from math import cos, sin

import scipy

from scipy.ndimage import affine_transform
import numpy as np


@lru_cache(maxsize=10)
def get_matrix(angles=(90, 90, 90), inverse=False):
    """
    Axis of rotation Get matrix by angle.
    (shear+compress)
    Examples: z = 120
    
    ############################################################
    
    ----------------------          --------------------------------
    -oooooooooooooooooooo-          --------------------------------
    -oooooooooooooooooooo-          -oooooooooooooooooooo-----------
    -oooooooooooooooooooo-          ---oooooooooooooooooooo---------
    -oooooooooooooooooooo-    >>>   -----oooooooooooooooooooo-------
    -oooooooooooooooooooo-          -------oooooooooooooooooooo-----
    -oooooooooooooooooooo-          ---------oooooooooooooooooooo---
    -oooooooooooooooooooo-          -----------oooooooooooooooooooo-
    ----------------------          --------------------------------
    
    ############################################################
    
    1.The ``matrix`` is the transform matrix to rotate the data with angle. Always in Cartesian coordinates.

    2.The ``inverse matrix`` is the interpolation matrix for get true data matrix(Cartesian coordinates)
     from relative data matrix (Non-Cartesian coordinates).
     The

    Parameters
    ----------
    angles: tuple
        3 angle of x, y, z
        z angle is the intersection angle of x,y,
        y angle is the intersection angle of x,z,
        x angle is the intersection angle of y,z.
    inverse:
        Compute the (multiplicative) inverse of a matrix.

    """
    theta1, theta2, theta3 = [np.pi / 180 * angle for angle in angles]

    matrix1 = np.array([[1, cos(theta3), 0],
                        [0, sin(theta3), 0],
                        [0, 0, 1]])

    matrix2 = np.array([[1, 0, 0],
                        [0, 1, cos(theta1)],
                        [0, 0, sin(theta1)]])

    matrix3 = np.array([[1, 0, cos(theta2)],
                        [0, 1, 0],
                        [0, 0, sin(theta2)]])
    matrix = np.dot(matrix1, matrix2).dot(matrix3)

    if inverse:
        matrix = np.linalg.inv(matrix)

    return matrix


def rotation_axis_by_angle(data, angles=(90, 90, 90), times=(2, 2, 2)):
    """
    Get true data matrix(Cartesian coordinates) from relative data matrix (Non-Cartesian coordinates).

    Parameters
    ----------
    data: np.ndarray
        data with shape (nx,ny,nz).
    angles:tuple
        3 angle of x, y, z
        z angle is the intersection angle of x,y,
        y angle is the intersection angle of x,z,
        x angle is the intersection angle of y,z.
    times: tuple
        expand the multiple of the matrix.

    """
    matrix = get_matrix(angles=angles, inverse=True)
    return rotation_axis_by_matrix(data, matrix, times=times)


def rotation_axis_by_matrix(data, matrix, times=(2, 2, 2)):
    """
    Get true data matrix(Cartesian coordinates) from relative data matrix (Non-Cartesian coordinates).

    Parameters
    ----------
    data: np.ndarray
        data with shape (nx,ny,nz).
    matrix:tuple
        See Also ``get_matrix``
    times: tuple
        expand the multiple of the matrix.

    """
    dims_old = data.shape

    dims = tuple([int(i * j) for i, j in zip(dims_old, times)])
    n_data = np.zeros(dims)
    d0s = int((dims[0] - dims_old[0]) / 2)
    d1s = int((dims[1] - dims_old[1]) / 2)
    d2s = int((dims[2] - dims_old[2]) / 2)
    n_data[d0s:d0s + dims_old[0], d1s:d1s + dims_old[1], d2s:d2s + dims_old[2]] = data
    coords = np.meshgrid(range(dims[0]), range(dims[1]), range(dims[2]), indexing="ij")
    xy_coords = np.vstack([coords[0].reshape(-1), coords[1].reshape(-1), coords[2].reshape(-1)])
    # apply the transformation matrix
    # please note: the coordinates are not homogeneous.
    # for the 3D case, I've added code for homogeneous coordinates, you might want to look at that
    # please also note: rotation is always around the origin:
    # since I want the origin to be in the image center, I had to substract dim/2, rotate, then add it again
    dims2 = np.array([i / 2 for i in dims])
    dims2 = dims2.reshape(-1, 1)
    xy_coords = np.dot(matrix, xy_coords - dims2) + dims2
    #
    # # undo the stacking and reshaping
    x = xy_coords[0, :]
    y = xy_coords[1, :]
    z = xy_coords[2, :]
    x = x.reshape(dims, order="A")
    y = y.reshape(dims, order="A")
    z = z.reshape(dims, order="A")

    new_coords = [x, y, z]

    # use map_coordinates to sample values for the new image
    new_img = scipy.ndimage.map_coordinates(n_data, new_coords, order=2)
    return new_img


def _coords(points, angles=(90, 90, 90), times=(2, 2, 2)):
    """

    Parameters
    ----------
    points: np.darray
        percent of shape.
        key points with shape(n_sample,3)

    angles:tuple
        3 angle of x, y, z
        z angle is the intersection angle of x,y,
        y angle is the intersection angle of x,z,
        x angle is the intersection angle of y,z.
    times: tuple
        expand the multiple of the matrix.

    """
    dims_old = [1, 1, 1]
    matrix = get_matrix(angles=angles)

    times = np.array(list(times))
    times = times.reshape((-1, 1))
    dims_old = np.array(dims_old)
    dims_old = dims_old.reshape(-1, 1)
    dims2 = dims_old / 2

    points = points.T * dims_old

    xy_coords = np.dot(matrix, points - dims2) + dims2

    xy_coords = xy_coords + (times / 2 - 0.5)

    return xy_coords


def rote_index(points, data, angles=(90, 90, 90), times=(2, 2, 2), data_init=True, return_type="float"):
    """

    Parameters
    ----------
    points: np.darray
        key points with shape(n_sample,3)
        percent of shape.
    data: np.ndarray or tuple
        data or data.shape
    data_init:bool
        The data is the init data (relative location) or Cartesian coordinates.(rotation_axis_by_angle)
    angles:tuple
        3 angle of x, y, z
        z angle is the intersection angle of x,y,
        y angle is the intersection angle of x,z,
        x angle is the intersection angle of y,z.
    times: tuple
        expand the multiple of the matrix.
    return_type:str
        "float", "int", "percent"
        for "float", "int" return the new index
        for "percent" return the new percent.

    """

    data_shape = data.shape if isinstance(data, np.ndarray) else data
    if data_init:
        times_np = np.array([1,1,1])
    else:
        times_np = np.array(times)

    dims = data_shape
    dims = np.array(dims).reshape((-1, 1))

    xy_coords = _coords(points, angles=angles, times=times)
    if return_type == "percent":
        return xy_coords
    if return_type == "float":
        return (dims * xy_coords/times_np).T
    else:
        return np.round((dims * xy_coords/times_np).T).astype(int) # for rounding off: .4 -, .5 +


def rote_value(points, data, angles=(90, 90, 90), times=(2, 2, 2), method="in", data_type="td"):
    """

    Parameters
    ----------
    points: np.darray
        key points with shape(n_sample,3)
        percent of shape.
    data: np.ndarray
        data
    angles:tuple
        3 angle of x, y, z
        z angle is the intersection angle of x,y,
        y angle is the intersection angle of x,z,
        x angle is the intersection angle of y,z.
    times: tuple
        expand the multiple of the matrix.
    data_type:str
        if  "init"  the data accept init data (elfcar, chgcar). see rotation_axis_by_angle.
        if  "td"  the data accept true matrix data . see rotation_axis_by_angle.

    method:str
        if  "near" , return nearest site's value.
        if  "inter" , return the interpolation value.
    """

    if data_type == "td":
        new_data = data
    else:
        new_data = rotation_axis_by_angle(data, angles=angles, times=times)

    if method == "near":
        ind = rote_index(points, data, angles=angles, times=times, return_type="int")
        new_value = np.array([new_data[tuple(i)] for i in ind.T])
        return new_value
    else:
        ind = rote_index(points, data, angles=angles, times=times, return_type="float")
        new_value = scipy.ndimage.map_coordinates(new_data, ind, order=2)
        return new_value
