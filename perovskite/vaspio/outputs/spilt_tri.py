import itertools
from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from sympy import Matrix, symarray, solve
from sympy.core.numbers import One

one = One()


def spilt_tri_prism_z(data, six_index_xy, z_range=(0, 1), index_percent=True):
    """
    Spilt tri_prism in block data with six point, and the tri_prism is parallel to to Z axis.

    Thus use Three points of (x,y) and z_range to site the six point.


    Parameters
    ----------
    data: np.ndarray, tuple
        The matrix data self or the shape of matrix data.
    six_index_xy: np.ndarray
        Three x,y site of the six point of Tri-Prism.
        Examples:
            np.array([(0, 0.5), (0, 0), (0.7, 0.5)]) #if not index_percent
            np.array([(0, 5), (0, 0), (7, 5)]) #if not index_percent
    z_range:tuple
        z axis range.
        Examples:
            z_range=(0, 1) #if index_percent
            z_range=(0, 120) #if not index_percent
    index_percent:bool
        Treat the six_index_xy and z_range as percent of aixs or true index.

    Returns
    -------
    np.ndarray

    """
    if index_percent is False:
        if z_range[1] <= 1:
            raise NotImplemented("The z_range must be the shape of z axis without index_percent, such as (0,120)")
    six_index = []
    for i in six_index_xy:
        k = list(i)
        k.append(z_range[0])
        six_index.append(k)
        k = list(i)
        k.append(z_range[1])
        six_index.append(k)
    six_index = np.array(six_index)
    return spilt_tri_prism(data, six_index, index_percent)


def spilt_tri_prism(data, six_index, index_percent=True):
    """
    Spilt tri_prism in block data with six point.

    Parameters
    ----------
    data: np.ndarray, tuple
        The matrix data self or the shape of matrix data.
    six_index:np.ndarray
        Three x,y site of the six point of Tri-Prism.
        Examples:
            np.array([(0, 0.5), (0, 0), (0.7, 0.5)]) #if not index_percent
            np.array([(0, 5), (0, 0), (7, 5)]) #if not index_percent

    index_percent:bool
        Treat the six_index_xy and z_range as percent of aixs or true index.

    Returns
    -------
    np.ndarray
        Bool type with data' shape.
    """

    data_shape = data.shape if isinstance(data, np.ndarray) else data

    if not index_percent:
        six_index = six_index / data_shape

    assert np.max(six_index) <= 1
    assert np.min(six_index) >= 0

    # get the center of tri_prism
    center = np.mean(six_index, axis=0)
    center1 = np.append(center, 1)

    # get the five face of tri_prism
    si = tuple([tuple(i) for i in list(six_index)])
    face = get_face(si)

    # check the inner direction (to face)
    face_bool = np.sum(face * center1, axis=1)

    # check the index's direction  (to face)
    dims = data_shape
    coords = np.meshgrid(np.linspace(0, 1, dims[0]), np.linspace(0, 1, dims[1]), np.linspace(0, 1, dims[2]),
                         indexing="ij")
    xy_coords = np.vstack([coords[0].ravel(), coords[1].ravel(), coords[2].ravel(), np.ones_like(coords[2]).ravel()])

    corrds_bool = np.dot(face, xy_coords)

    # get the index's is in inner (to face)
    inface_index = corrds_bool * face_bool.reshape(-1, 1)
    inface_bool = inface_index >= 0

    # reback the index's to block
    inface_index_allTrue = np.all(inface_bool, axis=0)
    mark = inface_index_allTrue.reshape(dims, order="A")
    mark = mark.astype(int) # bool to 01
    return mark


@lru_cache(maxsize=300)
def get_face(six_index, tol=1e-4):
    """Get the five face of tri_prism."""
    six_index = np.array(six_index)
    all_com = []
    x = symarray('x', 4)

    # get the face fo all 3 point.
    for i in itertools.combinations(six_index, 3):
        a = np.array(i)
        a = np.concatenate((a, np.array([[1], [1], [1]])), axis=1)
        a = Matrix(a)

        z = a * x
        z = [sum(zi) for zi in z]

        re = solve(z, *x)

        if isinstance(re, dict):
            dd = {}
            for i in x:
                if i in re.keys():
                    dd[i] = re[i]
                else:
                    dd[i] = one
            all_com.append(dd)

    def sub_s(dic):
        """The intercept is reduced by 1, and the coefs are multiple to intercept"""
        coef = []
        for v in dic.values():
            if hasattr(v, 'free_symbols') and isinstance(v.free_symbols, Iterable):
                for free in v.free_symbols:
                    df = dic[free]
                    v = v.xreplace({free: df})
                coef.append(v)
            else:
                coef.append(v)
        return tuple(coef)

    coefs = []
    [coefs.append(sub_s(i)) for i in all_com]
    coefs = list(set(coefs))
    coefs.sort()
    six_index1 = np.concatenate((six_index, np.array([1, 1, 1, 1, 1, 1]).reshape(-1, 1)), axis=1)

    # delete the inner face, by judge the six points in only one side of face.
    sea = []
    for i in coefs:
        sums = np.sum(six_index1 * np.array(i), axis=1)
        th1 = np.all(sums >= -tol)
        th2 = np.all(sums <= tol)
        if th1:
            sea.append(i)
        elif th2:
            sea.append(i)
        else:
            pass

    sea = np.array(sea).astype(float) #sympy to float
    if sea.shape != (5, 4):
        print("The log")
        for n, i in enumerate(coefs):
            print("Face", i)
            sums = np.sum(six_index1 * np.array(i), axis=1)
            print("distinguish", sums)
            th1 = np.all(sums >= -tol)
            th2 = np.all(sums <= tol)
            if th1 or th2:
                print("number", n, "True ")
            else:
                print("number", n, "False ")
        raise TypeError("please check the log")

    return sea


def relative_location(points):
    """

    Parameters
    ----------
    points:
        np.ndarray
        shape(n_sample,n_dim)

    Returns
    -------
    new_points:
        np.ndarray
        shape(n_sample,n_dim)
    """
    maxs = np.max(points, axis=0)
    mins = np.min(points, axis=0)
    points = (points - mins) / (maxs - mins)
    return points


if __name__ == "__main__":
    # ss = spilt_tri_prism_z(np.ones([11, 12, 13]), np.array([(0, 5), (0, 0), (7, 5)]), z_range=(0, 8),
    #                        index_percent=False)
    #
    # ss2 = spilt_tri_prism_z(np.ones([11, 11, 13]), np.array([(0, 5), (0, 0), (7, 5)]), z_range=(0, 8),
    #                         index_percent=False)
    #
    # ss2 = spilt_tri_prism_z(np.ones([11, 12, 13]), np.array([(0, 5), (0, 0), (7, 5)]), z_range=(0, 8),
    #                         index_percent=False)

    a = np.array([5, 2.0001, 2, 4.0004, 4, 1])
    b = np.sort(a)
    c = np.diff(b)
    c=np.append(c,np.inf)
    d = np.where(abs(c)>1e-3)
    res=b[d]