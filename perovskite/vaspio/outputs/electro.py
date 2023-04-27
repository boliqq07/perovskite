import copy
import logging

import numpy as np
from pymatgen.io.vasp import Chgcar, VolumetricData

# whether pyplot installed
from .data_rotate import rotation_axis_by_angle, rote_index
from .spilt_tri import relative_location, spilt_tri_prism_z

try:
    # import matplotlib
    # matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    plt_installed = True
except ImportError:
    print('Warning: Module matplotlib.pyplot is not installed')
    plt_installed = False

# whether mayavi installed
try:
    from mayavi import mlab

    mayavi_installed = True
except ImportError:
    mayavi_installed = False


class ElePlot:

    def __init__(self, data):
        """

        Parameters
        ----------
        data: np.ndarray
        """
        self.elf_data = data
        self.grid = self.elf_data.shape
        self.width = (1, 1, 1)

        self.__logger = logging.getLogger("vaspy.log")

    @staticmethod
    def expand_data(data, grid, widths):
        """

        Expand the data n times by widths.

        Parameters
        ----------
        data: np.ndarray
            elf or chg data.
        grid: tuple
            numpy.shape of data.
        widths: tuple of int, 3D
            number of replication on x, y, z axis
            According to Widths, the three-dimensional matrix was extended along the X, Y and Z axes.
            Examples:
                (2,1,3)
        """

        # expand grid
        widths = np.array(widths)
        expanded_grid = np.array(grid) * widths  # expanded grid
        # expand eld_data matrix
        expanded_data = copy.deepcopy(data)
        nx, ny, nz = widths
        # x axis
        added_data = copy.deepcopy(expanded_data)
        for i in range(nx - 1):
            expanded_data = np.append(expanded_data, added_data, axis=0)
        # y axis
        added_data = copy.deepcopy(expanded_data)
        for i in range(ny - 1):
            expanded_data = np.append(expanded_data, added_data, axis=1)
        # z axis
        added_data = copy.deepcopy(expanded_data)
        for i in range(nz - 1):
            expanded_data = np.append(expanded_data, added_data, axis=2)

        return expanded_data, expanded_grid

    def get_width_by_pbc_max(self, pbc_directions):
        """the max pbc +1"""
        if isinstance(pbc_directions, list):
            pbc_directions = np.concatenate(pbc_directions, axis=0)
        width = np.max(pbc_directions, axis=0) + 1
        width = width.astype(int)
        width = tuple(width.tolist())
        self.width = width
        return width

    @staticmethod
    def get_pbc_index(pbc_direction):

        tpbc_d = np.array(pbc_direction)

        reformed_tpd = [tpbc_d[:, i] + 1 if np.any(tpbc_d[:, i] < 0)
                        else tpbc_d[:, i] for i in range(tpbc_d.shape[1])]
        reformed_tpd = np.array(reformed_tpd).T
        return reformed_tpd


class ChgCar(Chgcar, ElePlot):

    def __init__(self, poscar, data, data_aug=None):
        Chgcar.__init__(self, poscar, data, data_aug)
        self.elf_data = self.data["total"]
        ElePlot.__init__(self, data=self.elf_data)

    @classmethod
    def from_file(cls, filename):
        """
        Reads a CHGCAR file.

        :param filename: Filename
        :return: Chgcar
        """
        (poscar, data, data_aug) = VolumetricData.parse_file(filename)
        return cls(poscar, data, data_aug=data_aug)

    def get_cartesian_data(self, data=None, times=(2, 2, 2), pbc_directions=None):

        angles = self.structure.lattice.angles
        if data is None:
            if isinstance(pbc_directions, tuple):
                elf_data, grid = self.expand_data(self.elf_data, self.grid, pbc_directions)
                self.width = pbc_directions
                self.cartesian_data = rotation_axis_by_angle(elf_data, angles=angles, times=times)
            elif pbc_directions is not None:
                widths = self.get_width_by_pbc_max(pbc_directions)
                elf_data, grid = self.expand_data(self.elf_data, self.grid, widths)
                self.cartesian_data = rotation_axis_by_angle(elf_data, angles=angles, times=times)
            else:
                self.cartesian_data = rotation_axis_by_angle(self.elf_data, angles=angles, times=times)

        else:
            self.cartesian_data = rotation_axis_by_angle(data, angles=angles, times=times)

        return self.cartesian_data

    def trans(self, point_indexes, pbc_direction):
        frac_coords = self.structure.frac_coords
        point = frac_coords[point_indexes, :]
        if pbc_direction is None:
            pass
        else:
            point = point + self.get_pbc_index(pbc_direction)
        return point

    def get_tri_data_z(self, point_indexes, pbc_direction=None, z_range=None, z_absolute=True):
        assert hasattr(self, 'cartesian_data'), "please '.get_cartesian_data' first."
        point = self.trans(point_indexes, pbc_direction)
        point = np.array(point) / np.array(self.width)

        percent = rote_index(point, self.cartesian_data.shape, data_init=False, angles=self.structure.lattice.angles,
                             return_type="int")
        maxs = np.max(percent.astype(int), axis=0)
        mins = np.min(percent.astype(int), axis=0)
        if z_range is None:
            data_target = self.cartesian_data[mins[0]:maxs[0], mins[1]:maxs[1], :]
        elif z_range == "zero_to_half":
            data_target = self.cartesian_data[mins[0]:maxs[0], mins[1]:maxs[1], :int(self.cartesian_data.shape[2] / 2)]
        elif z_range == "half_to_all":
            data_target = self.cartesian_data[mins[0]:maxs[0], mins[1]:maxs[1], int(self.cartesian_data.shape[2] / 2):]
        elif isinstance(z_range, tuple) and z_absolute:
            data_target = self.cartesian_data[mins[0]:maxs[0], mins[1]:maxs[1], z_range[0]:z_range[1]]
        elif isinstance(z_range, tuple) and not z_absolute:
            z_r = (int(z_range[0] * self.cartesian_data.shape[2]), int(z_range[1] * self.cartesian_data.shape[2]))
            data_target = self.cartesian_data[mins[0]:maxs[0], mins[1]:maxs[1], z_r[0]:z_r[1]]
        else:
            raise TypeError("The z_range must be None(all),'zero_to_half','half_to_all' or tuple of with int 2")
        relative = relative_location(percent[:, (0, 1)])
        site = relative * np.array(data_target.shape[:2])
        data_target_tri = spilt_tri_prism_z(data_target.shape, site, z_range=(0, data_target.shape[2]),
                                            index_percent=False)
        data_result = data_target_tri * data_target
        return data_result

    def get_cubic_data(self, point_indexes, pbc_direction=None):
        assert hasattr(self, 'cartesian_data'), "please '.get_cartesian_data' first."

        point = self.trans(point_indexes, pbc_direction)

        percent = rote_index(point, self.cartesian_data.shape, data_init=False, angles=self.structure.lattice.angles,
                             return_type="int")
        maxs = np.max(percent.astype(int), axis=0)
        mins = np.min(percent.astype(int), axis=0)

        data_result = self.cartesian_data[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

        return data_result

    def plot_field(self, show_mode="show", data=None, **kwargs):
        """

        use mayavi.mlab to plot 3d field.

        Parameter
        ---------
        kwargs: {
            'vmin'   : ,min ,
            'vmax'   : ,max,
            'axis_cut': ,cut size,
            'nct'     : int, number of contours,
            'opacity' : float, opacity of contour,
            'widths'   : tuple of int
                        number of replication on x, y, z axis,
        }

        """

        if not mayavi_installed:
            self.__logger.warning("Mayavi is not installed on your device.")
            return
        # set parameters
        vmin = kwargs['vmin'] if 'vmin' in kwargs else 0.0
        vmax = kwargs['vmax'] if 'vmax' in kwargs else 1.0
        axis_cut = kwargs['axis_cut'] if 'axis_cut' in kwargs else 'z'
        nct = kwargs['nct'] if 'nct' in kwargs else 3
        widths = kwargs['widths'] if 'widths' in kwargs else (1, 1, 1)
        times = kwargs['times'] if 'times' in kwargs else (2, 2, 2)
        if data is None:
            elf_data, grid = self.expand_data(self.elf_data, self.grid, widths)
            elf_data = self.get_cartesian_data(data=elf_data, times=times)
        else:
            elf_data = data
        # create pipeline
        field = mlab.pipeline.scalar_field(elf_data)  # data source
        mlab.pipeline.volume(field, vmin=vmin, vmax=vmax)  # put data into volumn to visualize
        # cut plane
        if axis_cut in ['Z', 'z']:
            plane_orientation = 'z_axes'
        elif axis_cut in ['Y', 'y']:
            plane_orientation = 'y_axes'
        elif axis_cut in ['X', 'x']:
            plane_orientation = 'x_axes'
        cut = mlab.pipeline.scalar_cut_plane(
            field.children[0], plane_orientation=plane_orientation)
        cut.enable_contours = True  # 开启等值线显示
        cut.contour.number_of_contours = nct
        mlab.show()
        # mlab.savefig('field.png', size=(2000, 2000))
        if show_mode == 'show':
            mlab.show()
        elif show_mode == 'save':
            mlab.savefig('mlab_contour3d.png')
        else:
            raise ValueError('Unrecognized show mode parameter : ' +
                             show_mode)

        return None

# class ElfCar(Elfcar, ElePlot):
#     def __init__(self, poscar, data, data_aug=None):
#         ElfCar.__init__(self, poscar, data, data_aug)
#         self.elf_data = self.data["total"]
#         ElePlot.__init__(self, data=self.elf_data)
#
#     @classmethod
#     def from_file(cls, filename):
#         """
#         Reads a CHGCAR file.
#
#         :param filename: Filename
#         :return: Chgcar
#         """
#         (poscar, data, data_aug) = VolumetricData.parse_file(filename)
#         return cls(poscar, data, data_aug=data_aug)
