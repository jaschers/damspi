import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the damspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import numpy as np
import pandas as pd
import eagleSqlTools as sql
from astropy.cosmology import Planck13 as cosmo
from astropy import units as u
import h5py
from damspi.utils import (
    convert_float_to_three_digit, 
    convert_float_to_six_digit, 
    redshift,
    nfw_profile,
    nfw_integral,
    cored_profile,
    cored_integral,
    M_bh_2,
    rescaled_distance
    )
from scipy.spatial.transform import Rotation as R
from astropy.coordinates import cartesian_to_spherical
from scipy.odr import Model, RealData, ODR, Data
import damspi.plot as damplot
import yaml

username = os.getenv("VIRGODB_USERNAME")
password = os.getenv("VIRGODB_PASSWORD")

with open("config/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class DataCollector:
    """
    Collect the data of the black holes and their host galaxies.

    The data of the black holes and their host galaxies is collected from the EAGLE database. The data is stored in pandas DataFrames.

    Parameters
    ----------
    sim_name: str
        The name of the EAGLE simulation.
    number_files: int
        The number of hdf5 files.

    Attributes
    ----------
    sim_name: str
        The name of the EAGLE simulation.
    number_files: int
        The number of hdf5 files.
    dict_redshift: dict
        The dictionary of the redshifts of the snapshots.
    hubble_constant: float
        The Hubble constant.
    minimal_galaxy_mass: float
        The minimal galaxy mass to form a black hole in the EAGLE simulation.
    bh_mass_formation: astropy.units.quantity.Quantity
        The black hole mass at which the black hole forms in the EAGLE simulation.
    stellar_mass_range: list
        The stellar mass range of the host galaxies.
    halo_mass_range: list
        The halo mass range of the host galaxies.
    bh_mass_limit: astropy.units.quantity.Quantity
        The black hole mass limit.

    Methods
    -------
    black_hole_data(nsnap)
        Get the black hole data for snapshot nsnap from the EAGLE simulations.
    find_next_smallest(dictionary, values)
        Find the next smallest value in a dictionary.
    galaxy_data(nsnap)
        Get the galaxy data for snapshot nsnap from the EAGLE simulations.
    read_dataset(itype, att, nsnap)
        Read a selected dataset, itype is the PartType and att is the attribute name.

    Notes
    -----
    The black hole data is read from the hdf5 files stored in the 'data/' folder. The galaxy data is requested from the EAGLE database using the SQL request.

    Examples
    --------
    >>> DataCollector(sim_name = "RefL0100N1504", number_files = 256)
    """

    def __init__(self, sim_name, number_files):
        self.sim_name = sim_name
        self.number_files = number_files
        nsnap_list = np.linspace(0, 28, 29)
        redshift_list = [20., 15.132, 9.993, 8.988, 8.075, 7.050, 5.971, 5.487, 5.037, 4.485, 3.984, 3.528, 3.017, 2.478, 2.237, 2.012, 1.737, 1.487, 1.259, 1.004, 0.865, 0.736, 0.615, 0.503, 0.366, 0.271, 0.183, 0.101, 0.]
        self.dict_redshift = dict(zip(nsnap_list, redshift_list))
        self.hubble_constant = cosmo.H0.value / 100
        self.minimal_galaxy_mass = 10**10 * u.Msun / self.hubble_constant
        self.bh_mass_formation = 10**5 * u.Msun / self.hubble_constant
        self.stellar_mass_range = [eval(config["Milky_way"]["stellar_mass_range"][0]), eval(config["Milky_way"]["stellar_mass_range"][1])] * u.Msun
        self.halo_mass_range = [eval(config["Milky_way"]["halo_mass_range"][0]), eval(config["Milky_way"]["halo_mass_range"][1])] * u.Msun
        self.bh_mass_limit = 10**6 * u.Msun

    def black_hole_data(self, nsnap):
        """
        Get the black hole data for snapshot nsnap from the EAGLE simulations.

        Parameters
        ----------
        nsnap: int
            The snapshot number.

        Returns
        -------
        table_bh: pandas.DataFrame
            The black hole data.

        Notes
        -----
        The black hole data is read from the hdf5 files stored in the 'data/' folder. The data is stored in a pandas DataFrame.

        Examples
        --------
        >>> black_hole_data(nsnap = 28)
        """

        # extract BH values at z = 0
        bh_subgrid_mass = (self.read_dataset(itype = 5, att = "BH_Mass", nsnap = nsnap) * u.g).to(u.M_sun).value
        bh_n_merger = self.read_dataset(itype = 5, att = "BH_CumlNumSeeds", nsnap = nsnap) 
        bh_coordinates = (self.read_dataset(itype = 5, att = "Coordinates", nsnap = nsnap) * u.cm).to(u.kpc).value
        bh_group_number = self.read_dataset(itype = 5, att = "GroupNumber", nsnap = nsnap)
        bh_subgroup_number = self.read_dataset(itype = 5, att = "SubGroupNumber", nsnap = nsnap)
        bh_id = self.read_dataset(itype = 5, att = "ParticleIDs", nsnap = nsnap)
        bh_formation_redshift = redshift(self.read_dataset(itype = 5, att = "BH_FormationTime", nsnap = nsnap))

        nsnap_closest, z_closest = self.find_next_smallest(self.dict_redshift, bh_formation_redshift)

        # stack data 
        data_bh = np.dstack((bh_id, bh_group_number, bh_subgroup_number, bh_subgrid_mass, bh_coordinates[:,0], bh_coordinates[:,1], bh_coordinates[:,2], bh_formation_redshift, z_closest, nsnap_closest, bh_n_merger))[0]

        # put data into pandas DataFrame
        table_bh = pd.DataFrame(data_bh, columns = ["bh_id", "group_number", "subgroup_number", "m", "coord_x", "coord_y", "coord_z", "z_f", "z_c", "nsnap_c", "n_merger"]).reset_index(drop = True)

        # select only BHs that did not merge
        table_bh = table_bh[table_bh["n_merger"] == 1].reset_index(drop = True)

        # select only BHs that are in the IMBH mass range, i.e. < 10^6 M_solar
        table_bh = table_bh[table_bh["m"] < self.bh_mass_limit.to(u.Msun).value].reset_index(drop = True)

        return(table_bh)

    @staticmethod
    def find_next_smallest(dictionary, values):
        """
        Find the next smallest value in a dictionary.
        
        Parameters
        ----------
        dictionary: dict
            The dictionary.
        values: list
            The values.
            
        Returns
        -------
        smallest_keys: list
            The keys of the smallest values.
        smallest_values: list
            The smallest values.
        
        Notes
        -----
        The smallest values are found by looping over the dictionary and comparing the values with the values in the values list.
        
        Examples
        --------
        >>> find_next_smallest(dictionary = {"a": 1, "b": 2, "c": 3}, values = [0.5, 1.5, 2.5])
        """

        smallest_values = []
        smallest_keys = []
        for value in values:
            smallest_value = None
            for dict_key, dict_value in zip(dictionary.keys(), dictionary.values()):
                if dict_value < value:
                    if smallest_value is None or dict_value > smallest_value:
                        smallest_value = dict_value
                        smallest_key = dict_key
            smallest_values.append(smallest_value)
            smallest_keys.append(smallest_key)
        return(smallest_keys, smallest_values)

    def galaxy_data(self, nsnap):
        """
        Get the galaxy data for snapshot nsnap from the EAGLE simulations.

        Parameters
        ----------
        nsnap: int
            The snapshot number.

        Returns
        -------
        table_galaxy: pandas.DataFrame
            The galaxy data.

        Notes
        -----
        The galaxy data is requested from the EAGLE database using the SQL request. The data is stored in a pandas DataFrame.

        Examples
        --------
        >>> galaxy_data(nsnap = 28)
        """

        query = f'SELECT \
                    SH.GalaxyID as galaxy_id, \
                    SH.GroupID as group_id, \
                    SH.GroupNumber as group_number, \
                    SH.SubGroupNumber as subgroup_number, \
                    SH.Redshift as z, \
                    SH.Snapnum as nsnap, \
                    SH.Mass as m, \
                    SH.MassType_Star as m_star, \
                    FOF.Group_M_Crit200 as m200, \
                    SH.CentreOfPotential_x as cop_x, \
                    SH.CentreOfPotential_y as cop_y, \
                    SH.CentreOfPotential_z as cop_z, \
                    SH.Stars_Spin_x as spin_x, \
                    SH.Stars_Spin_y as spin_y, \
                    SH.Stars_Spin_z as spin_z, \
                    SH.Image_face as img_face, \
                    SH.Image_edge as img_edge, \
                    SH.Image_box as img_box \
                FROM \
                    {self.sim_name}_SubHalo as MH, \
                    {self.sim_name}_SubHalo as SH, \
                    {self.sim_name}_Aperture as AP, \
                    {self.sim_name}_FOF as FOF \
                WHERE \
                    MH.Snapnum = {nsnap} \
                    and MH.SubGroupNumber = 0 \
                    and FOF.Group_M_Crit200 between {self.halo_mass_range[0].value} and {self.halo_mass_range[1].value} \
                    and sqrt(square(MH.CentreOfMass_x - MH.CentreOfPotential_x) + square(MH.CentreOfMass_y - MH.CentreOfPotential_y) + square(MH.CentreOfMass_z - MH.CentreOfPotential_z)) <= 0.07*FOF.Group_R_Crit200 * 1e-3 \
                    and AP.ApertureSize = 30 \
                    and AP.Mass_Star between {self.stellar_mass_range[0].value} and {self.stellar_mass_range[1].value} \
                    and FOF.GroupID = MH.GroupID \
                    and MH.Snapnum = SH.Snapnum \
                    and MH.GroupID = SH.GroupID \
                    and AP.GalaxyID = MH.GalaxyID \
                    and SH.Spurious = 0 \
                    and MH.Spurious = 0'
        
        # query = f'SELECT \
        #             SH.GalaxyID as galaxy_id, \
        #             SH.GroupID as group_id, \
        #             SH.GroupNumber as group_number, \
        #             SH.SubGroupNumber as subgroup_number, \
        #             SH.Redshift as z, \
        #             SH.Snapnum as nsnap, \
        #             SH.Mass as m, \
        #             SH.MassType_Star as m_star, \
        #             FOF.Group_M_Crit200 as m200, \
        #             SH.CentreOfPotential_x as cop_x, \
        #             SH.CentreOfPotential_y as cop_y, \
        #             SH.CentreOfPotential_z as cop_z, \
        #             SH.Stars_Spin_x as spin_x, \
        #             SH.Stars_Spin_y as spin_y, \
        #             SH.Stars_Spin_z as spin_z, \
        #             SH.Image_face as img_face, \
        #             SH.Image_edge as img_edge, \
        #             SH.Image_box as img_box \
        #         FROM \
        #             {self.sim_name}_SubHalo as MH, \
        #             {self.sim_name}_SubHalo as SH, \
        #             {self.sim_name}_FOF as FOF \
        #         WHERE \
        #             MH.Snapnum = {nsnap} \
        #             and MH.SubGroupNumber = 0 \
        #             and FOF.Group_M_Crit200 between {self.halo_mass_range[0].value} and {self.halo_mass_range[1].value} \
        #             and sqrt(square(MH.CentreOfMass_x - MH.CentreOfPotential_x) + square(MH.CentreOfMass_y - MH.CentreOfPotential_y) + square(MH.CentreOfMass_z - MH.CentreOfPotential_z)) <= 0.07*FOF.Group_R_Crit200 * 1e-3 \
        #             and FOF.GroupID = MH.GroupID \
        #             and MH.Snapnum = SH.Snapnum \
        #             and MH.GroupID = SH.GroupID \
        #             and SH.Spurious = 0 \
        #             and MH.Spurious = 0'

        # Execute query.
        con = sql.connect(username, password=password) # username, password
        # load galaxy data
        data_galaxy = sql.execute_query(con, query)
        # put data into pandas DataFrame
        table_galaxy = pd.DataFrame(data_galaxy)
        
        # Create a mask for host galaxies (subgroup_number == 0)
        host_galaxies = table_galaxy[table_galaxy['subgroup_number'] == 0]

        # create a mask for satellite galaxies (subgroup_number != 0)
        satellite_galaxies = table_galaxy[table_galaxy['subgroup_number'] != 0]

        # for each satellite galaxy, find the host galaxy and calculate the distance to the host galaxy
        for index, row in satellite_galaxies.iterrows():
            # find the host galaxy
            host_galaxy = host_galaxies[host_galaxies['group_number'] == row['group_number']]
            # calculate the distance to the host galaxy
            r = np.sqrt((host_galaxy['cop_x'].values[0] - row['cop_x'])**2 + (host_galaxy['cop_y'].values[0] - row['cop_y'])**2 + (host_galaxy['cop_z'].values[0] - row['cop_z'])**2) * u.Mpc
            # rescale the distance to the host galaxy based on its mass
            r_rescaled = rescaled_distance(r, host_galaxy['m200'].values[0] * u.Msun)
            # add the distance to the satellite galaxy
            satellite_galaxies.loc[index, 'r_rescaled'] = r_rescaled.value * 1e3 # in kpc

        # select only satellite galaxies that are within the distance limit
        r_min, r_max = config["Milky_way"]["satellite_rescaled_distance_range"] # kpc
        satellite_galaxies = satellite_galaxies[(satellite_galaxies['r_rescaled'] > r_min) & (satellite_galaxies['r_rescaled'] < r_max)].reset_index(drop = True)

        # keep only the satellite galaxies that have at least one star particle, i.e. m_star > 0
        satellite_galaxies = satellite_galaxies[satellite_galaxies["m_star"] > 0].reset_index(drop = True)

        # merge satellite galaxies with host galaxies into table_galaxy
        table_galaxy = pd.concat([host_galaxies, satellite_galaxies]).reset_index(drop = True)

        # Group by 'group_number' and calculate the total mass of host galaxies and their satellite galaxies
        grouped = table_galaxy.groupby('group_number')['m'].sum().reset_index()

        # Merge the host galaxies DataFrame with the grouped data
        host_galaxies = host_galaxies.merge(grouped, on='group_number', suffixes=('', '_total'))

        # Calculate the total mass of satellite galaxies for each host galaxy
        host_galaxies['m_sat'] = host_galaxies['m_total'] - host_galaxies['m']

        # Check if satellite mass is less than 10% of the total mass
        host_galaxies['m_sat < 0.1 m_total'] = host_galaxies['m_sat'] < 0.1 * host_galaxies['m_total']

        host_galaxies = host_galaxies[host_galaxies['m_sat < 0.1 m_total']]

        columns_to_drop = ['m_total', 'm_sat', 'm_sat < 0.1 m_total']

        host_galaxies = host_galaxies.drop(columns=columns_to_drop).reset_index(drop = True)

        # get the satellite galaxies that are part of the host galaxies
        satellite_galaxies = satellite_galaxies[satellite_galaxies['group_number'].isin(host_galaxies['group_number'])].reset_index(drop = True)

        # determine how many satellite galaxies each host galaxy has
        grouped_satellites = satellite_galaxies.groupby('group_number')['subgroup_number'].count().reset_index()
        
        # add the number of satellite galaxies to the host galaxies
        host_galaxies = host_galaxies.merge(grouped_satellites, on='group_number', suffixes=('', '_count'))

        # rename the column
        host_galaxies = host_galaxies.rename(columns = {'subgroup_number_count': 'n_satellites'})

        return(host_galaxies)

    def read_dataset(self, itype, att, nsnap):
        """
        Read a selected dataset, itype is the PartType and att is the attribute name.

        Parameters
        ----------
        itype: int
            The PartType.
        att: str
            The attribute name.
        nsnap: int
            The snapshot number.

        Returns
        -------
        data: numpy.ndarray
            The data.

        Notes
        -----
        The data is read from the hdf5 files stored in the 'data/' folder.

        Examples
        --------
        >>> read_dataset(itype = 5, att = "BH_Mass", nsnap = 28)
        """

        # Output array.
        data = []

        redshift = self.dict_redshift[int(nsnap)]
        nsnap = convert_float_to_three_digit(nsnap)
        redshift_1, redshift_2 = convert_float_to_six_digit(redshift)

        # Loop over each file and extract the PARTICLE data.
        for i in range(self.number_files):
            # f = h5py.File("./data/snap_028_z000p000.%i.hdf5"%i, "r")
            f = h5py.File(f"data/{self.sim_name}/snapshot_{nsnap}_z{redshift_1}p{redshift_2}/snap_{nsnap}_z{redshift_1}p{redshift_2}.%i.hdf5"%i, "r")

            # extract data of particle type itype but only if it is available
            try:
                tmp = f["PartType%i/%s"%(itype, att)][...]
            except:
                continue
            data.append(tmp)

            # Get conversion factors.
            cgs     = f["PartType%i/%s"%(itype, att)].attrs.get("CGSConversionFactor")
            aexp    = f["PartType%i/%s"%(itype, att)].attrs.get("aexp-scale-exponent")
            hexp    = f["PartType%i/%s"%(itype, att)].attrs.get("h-scale-exponent")

            # Get expansion factor and Hubble parameter from the header.
            a       = f["Header"].attrs.get("Time")
            h       = f["Header"].attrs.get("HubbleParam")

            f.close()

        # Combine to a single array.
        if len(tmp.shape) > 1:
            data = np.vstack(data)
        else:
            data = np.concatenate(data)

        # Convert to physical.
        if data.dtype != np.int32 and data.dtype != np.int64:
            data = np.multiply(data, cgs * a**aexp * h**hexp, dtype="f8")

        return data


class CoordinateTransformer:
    """
    The coordinates of the black holes are transformed to the galactic coordinate system.

    Parameters
    ----------
    table_galaxy: pandas.DataFrame
        The galaxy data of a single galaxy. Required columns are: "group_number", "subgroup_number", "z_f", "z_c", "nsnap_c", "cop_x", "cop_y", "cop_z", "spin_x", "spin_y", "spin_z".
    table_bh: pandas.DataFrame
        The black hole data of a single black hole. Required columns are: "group_number", "subgroup_number", "z_f", "z_c", "nsnap_c", "coord_x", "coord_y", "coord_z".
    box_size: astropy.units.quantity.Quantity
        The size of the EAGLE box.

    Attributes
    ----------
    distance_sun: float
        The distance between the Sun and the galactic centre in kpc.
    box_size: float
        The size of the EAGLE box in kpc.
    table_galaxy: pandas.DataFrame
        The galaxy data.
    galaxy_id: int
        The galaxy ID.
    galaxy_centre: numpy.ndarray
        The cartesian coordinates of the galaxy centre.
    galaxy_spin: numpy.ndarray
        The spin vector of the galaxy.
    galaxy_spin_rot: numpy.ndarray
        The rotated spin vector of the galaxy.
    rot_matrix_x: numpy.ndarray
        The rotation matrix around the x-axis.
    rot_matrix_y: numpy.ndarray
        The rotation matrix around the y-axis.
    table_bh: pandas.DataFrame
        The black hole data.
    bh_coord: numpy.ndarray
        The cartesian coordinates of the black holes in the EAGLE box coordinate system.
    bh_coord_gc: numpy.ndarray
        The cartesian coordinates of the black holes with respect to the galaxy centre.
    distance_gc: numpy.ndarray
        The distance between the black holes and the galaxy centre.
    bh_spherical_coord_gc: numpy.ndarray
        The spherical coordinates of the black holes with its origin of coordinates at the position of the galactic centre.
    
    Methods
    -------
    rot_angle_x(vec)
        Get rotation angle around x-axis.
    rot_angle_y(vec)
        Get rotation angle around y-axis.
    rot_matrices_x_y(vec)
        Get rotation matrices around x- and y-axis.
    rotate_x_y(vec, rot_matrix_x, rot_matrix_y)
        Rotate vector around x- and y-axis.
    plot_3d_maps(sim_name, path, save_animation = False)
        Plot 3d maps of black holes.

    Notes
    -----
    The coordinate system is rotated so that the spin vector of the galaxy is aligned with the z-axis.  

    Examples
    --------
    >>> CoordinateTransformer(table_galaxy = table_galaxy, table_bh = table_bh, box_size = 100 * u.Mpc)
    """

    def __init__(self, table_galaxy, table_bh, box_size):
        # shift coorindates system to position of the sun (8.33 kpc), https://iopscience.iop.org/article/10.1088/1475-7516/2011/03/051/pdf
        self.distance_sun = config["Milky_way"]["distance_sun"] # 8.33 # kpc
        self.box_size = box_size.to(u.kpc).value

        self.table_galaxy = table_galaxy
        self.galaxy_id = self.table_galaxy["galaxy_id"].values[0]
        self.galaxy_centre = self.table_galaxy[["cop_x", "cop_y", "cop_z"]].values[0] * 1e3 # convert to kpc
        self.galaxy_spin = self.table_galaxy[["spin_x", "spin_y", "spin_z"]].values[0] 
        self.galaxy_spin = self.galaxy_spin / np.linalg.norm(self.galaxy_spin)
        self.rot_matrix_x, self.rot_matrix_y = self.rot_matrices_x_y(self.galaxy_spin)
        self.galaxy_spin_rot = self.rotate_x_y(self.galaxy_spin, self.rot_matrix_x, self.rot_matrix_y)

        self.table_bh = table_bh
        self.bh_coord = self.table_bh[["coord_x", "coord_y", "coord_z"]].values
    
    @property
    def bh_coord_gc(self):
        """
        The cartesian coordinates of the black holes with respect to the galaxy centre.

        Returns
        -------
        bh_coord_gc: numpy.ndarray
            The cartesian coordinates of the black holes with respect to the galaxy centre.
        """

        bh_coord_gc = self.bh_coord - self.galaxy_centre
        # Consider periodic boundary conditions
        bh_coord_gc = bh_coord_gc - self.box_size * np.round(bh_coord_gc / self.box_size)
        return(bh_coord_gc)

    @property
    def distance_gc(self):
        """
        The distance between the black holes and the galaxy centre.

        Returns
        -------
        distance_gc: numpy.ndarray
            The distance between the black holes and the galaxy centre.
        """
        distance_gc = np.linalg.norm(self.bh_coord_gc, axis = 1)
        return(distance_gc)

    @property
    def bh_spherical_coord_gc(self):
        """
        The spherical coordinates of the black holes with its origin of coordinates at the position of the galactic centre.

        Returns
        -------
        r: numpy.ndarray
            The radial distance.
        lat: numpy.ndarray
            The latitude.
        long: numpy.ndarray
            The longitude.
        """
        r, lat, long = cartesian_to_spherical(
            self.bh_coord_gc_rot[:,0], 
            self.bh_coord_gc_rot[:,1], 
            self.bh_coord_gc_rot[:,2]
            )
        r, lat, long = r.value, lat.value, long.value
        return(r, lat, long)

    @staticmethod
    def rot_angle_x(vec):
        """
        Get rotation angle around x-axis.

        Parameters
        ----------
        vec: numpy.ndarray or list
            The vector.

        Returns
        -------
        theta: float
            The rotation angle.

        Notes
        -----
        The rotation angle is calculated using the arctan function.

        Examples
        --------
        >>> rot_angle_x([1, 2, 3])
        """
        theta = np.arctan(vec[1] / vec[2])
        return(theta)

    @staticmethod
    def rot_angle_y(vec):
        """
        Get rotation angle around y-axis.

        Parameters
        ----------
        vec: numpy.ndarray or list
            The vector.
            
        Returns
        -------
        theta: float
            The rotation angle.
        
        Notes
        -----
        The rotation angle is calculated using the arctan function.

        Examples
        --------
        >>> rot_angle_y([1, 2, 3])
        """
        if vec[2] < 0:
            theta = np.arctan(- vec[0] / vec[2]) + np.pi
        else:
            theta = np.arctan(- vec[0] / vec[2])
        return(theta)

    def rot_matrices_x_y(self, vec):
        """
        Get rotation matrices around x- and y-axis.

        Parameters
        ----------
        vec: numpy.ndarray or list
            The vector.

        Returns
        -------
        rot_matrix_x: numpy.ndarray
            The rotation matrix around the x-axis.
        rot_matrix_y: numpy.ndarray
            The rotation matrix around the y-axis.
        
        Notes
        -----
        The rotation matrices are calculated using the scipy.spatial.transform.Rotation class.

        Examples
        --------
        >>> rot_matrices_x_y([1, 2, 3])
        """
        theta_x = self.rot_angle_x(vec)
        rot_matrix_x = R.from_euler('x', theta_x)

        vec_rot_xz = rot_matrix_x.apply(vec)

        theta_y = self.rot_angle_y(vec_rot_xz)
        rot_matrix_y = R.from_euler('y', theta_y)

        return(rot_matrix_x, rot_matrix_y)

    @staticmethod
    def rotate_x_y(vec, rot_matrix_x, rot_matrix_y):
        """
        Rotate vector around x- and y-axis.

        Parameters
        ----------
        vec: numpy.ndarray or list
            The vector.
        rot_matrix_x: numpy.ndarray
            The rotation matrix around the x-axis.
        rot_matrix_y: numpy.ndarray
            The rotation matrix around the y-axis.

        Returns
        -------
        vec_rot_xy: numpy.ndarray
            The rotated vector.

        Notes
        -----
        The vector is rotated using the scipy.spatial.transform.Rotation class.

        Examples
        --------
        >>> rotate_x_y([1, 2, 3], rot_matrix_x, rot_matrix_y)
        """
        vec_rot_x = rot_matrix_x.apply(vec)
        vec_rot_xy = rot_matrix_y.apply(vec_rot_x)

        return(vec_rot_xy)

    @property
    def bh_coord_gc_rot(self):
        """
        Rotate coordinate system so that spin vector of galaxy is aligned with z-axis. The coordinate system is rotated so that the spin vector of the galaxy is aligned with the z-axis.

        Returns
        -------
        bh_coord_gc_rot: numpy.ndarray
            The cartesian coordinates of the black holes with respect to the galaxy centre.
        """
        bh_coord_gc_rot = self.rotate_x_y(self.bh_coord_gc, self.rot_matrix_x, self.rot_matrix_y)
        return(bh_coord_gc_rot)

    @property
    def bh_coord_sun(self):
        """
        Get black hole coordinates with respect to the Sun. The coordinate system is rotated so that the spin vector of the galaxy is aligned with the z-axis.

        Returns
        -------
        bh_coord_sun: numpy.ndarray
            The cartesian coordinates of the black holes with respect to the Sun.
        """
        bh_coord_sun = self.bh_coord_gc_rot
        bh_coord_sun[:,0] += self.distance_sun
        return(bh_coord_sun)

    @property
    def bh_galactic_coord(self):
        """
        Get galactic coordinates of black holes.

        Returns
        -------
        r: numpy.ndarray
            The radial distance.
        lat: numpy.ndarray
            The latitude.
        long: numpy.ndarray
            The longitude.
        """
        r, lat, long = cartesian_to_spherical(
            self.bh_coord_sun[:,0], 
            self.bh_coord_sun[:,1], 
            self.bh_coord_sun[:,2]
            )
        r, lat, long = r.value, lat.value, long.value
        return(r, lat, long)

    def plot_3d_maps(self, sim_name, path, save_animation = False):
        """
        Plot 3d maps of black holes.

        Parameters
        ----------
        sim_name: str
            The name of the EAGLE simulation.
        path: str
            The path to the folder where the plots are saved.
        save_animation: bool
            If True, the 3d maps are saved as animation.

        Notes
        -----
        The 3d maps are plotted using the damspi.plot.GalaxyPlotter class.

        Examples
        --------
        >>> plot_3d_maps(sim_name = "RefL0050N0752", path = "./plots/", save_animation = False)
        """
        galaxy_plotter = damplot.GalaxyPlotter(sim_name = sim_name, table_galaxy = self.table_galaxy, table_bh = self.table_bh)

        galaxy_plotter.plot_3d_map(
            self.bh_coord_gc, 
            self.galaxy_spin, 
            self.galaxy_id, 
            path = path + "3d_map", 
            save_animation = save_animation
            )
        galaxy_plotter.plot_3d_map(
            self.bh_coord_gc_rot, 
            self.galaxy_spin_rot, 
            self.galaxy_id, 
            path = path + "3d_map_rot", 
            save_animation = save_animation
            )
        galaxy_plotter.plot_3d_map(
            self.bh_coord_sun, 
            self.galaxy_spin_rot, 
            self.galaxy_id, 
            path = path + "3d_map_rot_shifted", 
            shifted = True,
            save_animation = save_animation
            )


class DMMiniSpikesCalculator:
    """
    Calculate the DM mini-spikes around the black holes.

    Parameters
    ----------
    sim_name: str
        The name of the EAGLE simulation.
    box_size: astropy.units.quantity.Quantity
        The size of the EAGLE box.
    dm_profile: str
        The dark matter profile.
    core_index: float
        The core index of the dark matter profile.
    table_bh: pandas.DataFrame
        The black hole data of a single black hole. Required columns are: "group_number", "subgroup_number", "z_f", "z_c", "nsnap_c", "coord_x", "coord_y", "coord_z", "m".

    Attributes
    ----------
    sim_name: str
        The name of the EAGLE simulation.
    box_size: float
        The size of the EAGLE box in kpc.
    dm_profile: str
        The dark matter profile.
    core_index: float
        The core index of the dark matter profile.
    table_bh: pandas.DataFrame
        The black hole data.
    bh_id: int
        The EAGLE black hole ID.
    group_number: int
        The EAGLE group number.
    subgroup_number: int
        The EAGLE subgroup number.
    z_formation: float
        The formation redshift of the black hole.
    z_closest: float
        The redshift of the closest snapshot.
    nsnap_closest: int
        The snapshot number of the closest snapshot.
    bh_coord: numpy.ndarray
        The cartesian coordinates of the black holes in the EAGLE box coordinate system.
    bh_mass: astropy.units.quantity.Quantity
        The black hole mass.
    no_host: bool
        If True, the black hole does not have a host galaxy.
    hubble_constant: float
        The Hubble constant.
    minimal_galaxy_mass: astropy.units.quantity.Quantity
        The minimal galaxy mass to form a black hole in the EAGLE simulation.
    bh_mass_formation: astropy.units.quantity.Quantity
        The black hole mass at which the black hole forms in the EAGLE simulation.
    rho_0: astropy.units.quantity.Quantity
        The dark matter density at the scale radius.
    r_s: astropy.units.quantity.Quantity
        The scale radius.
    r_c: astropy.units.quantity.Quantity
        The core radius.
    gamma_c: float
        The core index.
    r_h: astropy.units.quantity.Quantity
        The radius of gravitational influence.
    r_sp: astropy.units.quantity.Quantity
        The spike radius.
    rho_at_r_sp: astropy.units.quantity.Quantity
        The dark matter density at the spike radius.
    spike_index: float
        The spike index.

    
    Methods
    -------
    redshift_to_scale_factor(redshift)
        Convert redshift to scale factor using Astropy.
    distance_two_points(coord_1, coord_2)
        Calculate the distance between two points.
    query_galaxy_zf(subgroup_number_avail = True)
        Query the galaxy data of the host galaxy at the formation redshift of the black hole.
    get_table_galaxy_zf()
        Get the galaxy data of the host galaxy at the formation redshift of the black hole.
    density_within_aperature(r)
        Calculate the dark matter density within a given aperture.
    nfw_profile_log(params, r)
        Used to fit the NFW profile to the DM density profile of the host galaxy.
    cored_profile_log(params, r)
        Used to fit the cored profile to the DM density profile of the host galaxy.
    nfw_cost_function(params, r):
        The cost function for the NFW profile.
    cored_cost_function(params, r):
        The cost function for the cored profile.
    cored_cost_function_fixed_gamma_c(params, r):
        The cost function for the cored profile with fixed core index.
    nfw_fit()
        Fit the NFW profile to the dark matter density profile.
    cored_fit()
        Fit the cored profile to the dark matter density profile.
    cored_fit_fixed_gamma_c()
        Fit the cored profile to the dark matter density profile with fixed core index.
    radius_gravitational_influence_equation(r, rho_0, r_s, M_bh, r_c = None, gamma_c = None):
        The equation for the radius of gravitational influence.
    radius_gravitational_influence(rho_0, r_s, M_bh, r_c = None, gamma_c = None):
        Calculate the radius of gravitational influence.
    plot_nfw(path):
        Plot the NFW profile.
    plot_cored(path):
        Plot the cored profile.
    plot_radius_gravitational_influence(path):
        Plot the radius of the gravitational influence of the black hole fit.
    """

    def __init__(self, sim_name, box_size, dm_profile, core_index, table_bh):
        self.sim_name = sim_name
        self.box_size = box_size.to(u.kpc).value
        self.dm_profile = dm_profile
        self.core_index = core_index
        self.table_bh = table_bh
        self.bh_id = self.table_bh["bh_id"].values[0]
        self.group_number = self.table_bh["group_number"].values[0]
        self.subgroup_number = self.table_bh["subgroup_number"].values[0]
        self.z_formation = table_bh["z_f"].values[0]
        self.z_closest = table_bh["z_c"].values[0]
        self.nsnap_closest = table_bh["nsnap_c"].values[0]
        self.bh_coord = table_bh[["coord_x", "coord_y", "coord_z"]].values[0]
        self.bh_mass = table_bh["m"].values[0] * u.Msun
        self.no_host = (self.group_number == 2**30) or (self.subgroup_number == 2**30)
        self.hubble_constant = cosmo.H0.value / 100
        self.minimal_galaxy_mass = 10**10 * u.Msun / self.hubble_constant
        self.bh_mass_formation = 10**5 * u.Msun / self.hubble_constant
        if self.dm_profile == "nfw":
            self.rho_0, self.r_s = self.nfw_fit()
        elif self.dm_profile == "cored":
            if self.core_index == None:
                self.rho_0, self.r_s, self.r_c, self.gamma_c = self.cored_fit()
            else:
                self.rho_0, self.r_s, self.r_c, self.gamma_c = self.cored_fit_fixed_gamma_c()


    @staticmethod
    def redshift_to_scale_factor(redshift):
        """
        Convert redshift to scale factor using Astropy.

        Parameters
        ----------
        redshift: float
            The redshift.

        Returns
        -------
        a: float
            The scale factor.

        Notes
        -----
        The scale factor is calculated using the astropy.cosmology.PlanckXX class

        Examples
        --------
        >>> redshift_to_scale_factor(6.0)
        """
        a = cosmo.scale_factor(redshift)
        return a

    def distance_two_points(self, coord_1, coord_2):
        """
        Calculate the distance between two points.

        Parameters
        ----------
        coord_1: numpy.ndarray
            The cartesian coordinates of the first point.
        coord_2: numpy.ndarray
            The cartesian coordinates of the second point.

        Returns
        -------
        distance: float
            The distance between the two points.
        
        Notes
        -----
        The distance is calculated using the numpy.linalg.norm function.

        Examples
        --------
        >>> distance_two_points(np.array([1, 2, 3]), np.array([4, 5, 6]))
        """
        diff = coord_1 - coord_2
        diff = diff - self.box_size * np.round(diff / self.box_size)
        diff = np.linalg.norm(diff, axis = 1)
        return(diff)

    def query_galaxy_zf(self, subgroup_number_avail = True):
        """
        Query the galaxy data of the host galaxy at the formation redshift of the black hole.

        Parameters
        ----------
        subgroup_number_avail: bool
            If True, the subgroup number is available.
        
        Returns
        -------
        query: str
            The SQL query.
        
        Notes
        -----
        The galaxy data is requested from the EAGLE database using the SQL request. The data is stored in a pandas DataFrame.
        """
        scale_factor = self.redshift_to_scale_factor(self.z_closest)
        scale_factor = scale_factor * 1e3 # convert to kpc
        if (self.no_host == False) or (subgroup_number_avail == True):
            query = f"SELECT \
                        DES.GroupID as group_id, \
                        DES.GroupNumber as group_number, \
                        DES.SubGroupNumber as subgroup_number, \
                        DES.Redshift as z, \
                        DES.Snapnum as nsnap, \
                        DES.Mass as m, \
                        DES.CentreOfPotential_x as cop_x, \
                        DES.CentreOfPotential_y as cop_y, \
                        DES.CentreOfPotential_z as cop_z, \
                        DES.Image_face as img_face, \
                        DES.Image_edge as img_edge, \
                        DES.Image_box as img_box, \
                        DES.Spurious as spurious, \
                        AP.ApertureSize as aperture_size, \
                        AP.Mass_DM as m_dm_ap \
                    FROM \
                        {self.sim_name}_SubHalo as DES, \
                        {self.sim_name}_Aperture as AP \
                    WHERE \
                        DES.Snapnum = {self.nsnap_closest} \
                        and DES.GroupNumber = {self.group_number} \
                        and DES.SubGroupNumber = {self.subgroup_number} \
                        and AP.GalaxyID = DES.GalaxyID"

        else:
            query = f"SELECT \
                        DES.GroupID as group_id, \
                        DES.GroupNumber as group_number, \
                        DES.SubGroupNumber as subgroup_number, \
                        DES.Redshift as z, \
                        DES.Snapnum as nsnap, \
                        DES.Mass as m, \
                        DES.CentreOfPotential_x as cop_x, \
                        DES.CentreOfPotential_y as cop_y, \
                        DES.CentreOfPotential_z as cop_z, \
                        DES.Image_face as img_face, \
                        DES.Image_edge as img_edge, \
                        DES.Image_box as img_box, \
                        DES.Spurious as spurious \
                    FROM \
                        {self.sim_name}_SubHalo as DES \
                    WHERE \
                        DES.Snapnum = {self.nsnap_closest} \
                        and DES.Mass >= {self.minimal_galaxy_mass.to(u.Msun).value} \
                        and sqrt(square((DES.CentreOfPotential_x * {scale_factor}) - {self.bh_coord[0]}) + square((DES.CentreOfPotential_y * {scale_factor}) - {self.bh_coord[1]}) + square((DES.CentreOfPotential_z * {scale_factor}) - {self.bh_coord[2]})) <= 2e3 \
                        and DES.Spurious = 0"
        return(query)

    def get_table_galaxy_zf(self):
        """
        Get the galaxy data of the host galaxy at the formation redshift of the black hole.

        Returns
        -------
        table_galaxy_zf: pandas.DataFrame
            The galaxy data of the host galaxy at the formation redshift of the black hole.

        Notes
        -----
        The galaxy data is requested from the EAGLE database using the SQL request. The data is stored in a pandas DataFrame.
        """
        # get query
        if self.no_host:
            # if BH has no host galaxy, request galaxy data without aperture data (saves a lot of time)
            query = self.query_galaxy_zf(subgroup_number_avail = False)
        else:
            # if BH has a host galaxy, request galaxy data with aperture data
            query = self.query_galaxy_zf(subgroup_number_avail = True)

        # Execute query.
        con = sql.connect(username, password=password)
        # load galaxy data
        data_galaxy = sql.execute_query(con, query)
        # put data into pandas DataFrame
        table_galaxy_zf = pd.DataFrame(data_galaxy)

        # if no host galaxy is found, select the closest galaxy with mass > minimal_galaxy_mass
        if self.no_host:
            table_galaxy_zf = table_galaxy_zf[table_galaxy_zf["m"] >= self.minimal_galaxy_mass.value].reset_index(drop = True)

            # extract the coordinates of the galaxies at the redshift closest to BH formation redshift
            scale_factor = self.redshift_to_scale_factor(self.z_closest)
            galaxy_coord = table_galaxy_zf[["cop_x", "cop_y", "cop_z"]].to_numpy() * scale_factor * 1e3 # convert to kpc

            # calculate the distance of the galaxies to the BH
            distance_galaxies_bh = self.distance_two_points(galaxy_coord, self.bh_coord)

            # add distances to table
            table_galaxy_zf["distance BH [kpc]"] = distance_galaxies_bh

            # select the galaxy closest to the BH
            table_galaxy_zf = table_galaxy_zf[table_galaxy_zf["distance BH [kpc]"] == np.min(distance_galaxies_bh)].reset_index(drop = True)

            self.group_number = table_galaxy_zf["group_number"].values[0]
            self.subgroup_number = table_galaxy_zf["subgroup_number"].values[0]

            # request galaxy data with aperture data this time
            query = self.query_galaxy_zf(subgroup_number_avail = True)

            # Execute query.
            con = sql.connect(username, password=password)
            # load galaxy data
            data_galaxy = sql.execute_query(con, query)
            # put data into pandas DataFrame
            table_galaxy_zf = pd.DataFrame(data_galaxy)

        # check if BH host halo at formation redshift is spurious by definition or by having no proper aperture data
        spurios_galaxy = (table_galaxy_zf["spurious"] == 1).all()
        spurios_aperature = (table_galaxy_zf["m_dm_ap"] == 0).any()
        self.spurios = True if (spurios_galaxy or spurios_aperature) else False

        self.table_galaxy_zf = table_galaxy_zf
    
        return(self.table_galaxy_zf)

    @staticmethod
    def density_within_aperature(aperture, mass):
        """
        Calculate the DM density within the aperture.

        Parameters
        ----------
        aperture: astropy.units.quantity.Quantity
            The aperture size.
        mass: astropy.units.quantity.Quantity
            The DM mass within the aperture.

        Returns
        -------
        rho: astropy.units.quantity.Quantity
            The DM density within the aperture.

        Notes
        -----
        The DM density within the aperture is calculated using the DM mass within the aperture and the aperture size.

        Examples
        --------
        >>> density_within_aperature(10 * u.kpc, 10**10 * u.Msun)
        """
        volume = 4/3 * np.pi * aperture**3
        rho = (mass / volume).to(u.Msun/u.kpc**3)
        return(rho)

    @staticmethod
    def nfw_profile_log(params, r):
        """
        Used to fit the NFW profile to the DM density profile of the host galaxy.

        Parameters
        ----------
        params: tuple
            The NFW parameters.
        r: numpy.ndarray or astropy.units.quantity.Quantity
            The radius.

        Returns
        -------
        rho_log: numpy.ndarray or float
            The DM density profile of the host galaxy (log).
        
        Notes
        -----
        The NFW profile (log) is calculated using the NFW parameters and the radius.

        Examples
        --------
        >>> nfw_profile_log((5e6 * u.Msun / u.kpc**3, 40 * u.kpc), 10 * u.kpc)
        """
        rho_0, r_s = params
        if hasattr(r, 'unit'):
            r = r.to(u.kpc).value
        if hasattr(rho_0, 'unit'):
            rho_0 = rho_0.to(u.Msun/u.kpc**3).value
        if hasattr(r_s, 'unit'):
            r_s = r_s.to(u.kpc).value
        rho_log = np.log(rho_0) - np.log(r/r_s) - 2*np.log(1+r/r_s) 
        return(rho_log)

    def cored_profile_log(self, params, r):
        """
        Used to fit the cored profile to the DM density profile of the host galaxy.

        Parameters
        ----------
        params: tuple
            The cored parameters.
        r: numpy.ndarray or astropy.units.quantity.Quantity
            The radius.

        Returns
        -------
        rho_log: numpy.ndarray or float
            The DM density profile of the host galaxy (log).

        Notes
        -----
        The cored profile (log) is calculated using the cored parameters and the radius.

        Examples
        --------
        >>> cored_profile_log((5e6 * u.Msun / u.kpc**3, 40 * u.kpc, 5 * u.kpc, 0.2), 10 * u.kpc)
        """
        rho_0, r_s, r_c, gamma_c = params
        if hasattr(r, 'unit'):
            r = r.to(u.kpc).value
        if hasattr(rho_0, 'unit'):
            rho_0 = rho_0.to(u.Msun/u.kpc**3).value
        if hasattr(r_s, 'unit'):
            r_s = r_s.to(u.kpc).value
        if hasattr(r_c, 'unit'):
            r_c = r_c.to(u.kpc).value
        rho_nfw_log = np.log(rho_0) - np.log(r/r_s) - 2*np.log(1 + r/r_s)
        rho_nfw_log_rc = np.log(rho_0) - np.log(r_c/r_s) - 2*np.log(1 + r_c/r_s)
        mask = r < r_c
        log_rho_cored = np.where(mask, rho_nfw_log_rc - gamma_c * np.log(r/r_c), rho_nfw_log)

        return log_rho_cored

    def nfw_cost_function(self, params, r):
        """
        Used to fit the NFW profile to the DM density profile of the host galaxy.

        Parameters
        ----------
        params: tuple
            The NFW parameters.
        r: numpy.ndarray
            The radius.

        Returns
        -------
        predicted: numpy.ndarray
            The predicted DM density profile of the host galaxy (log).

        Notes
        -----
        The NFW profile (log) is calculated using the NFW parameters and the radius.

        Examples
        --------
        >>> nfw_cost_function((5e6 * u.Msun / u.kpc**3, 40 * u.kpc), 10 * u.kpc)
        """
        rho_0, r_s = params
        if rho_0 <= 0 or r_s <= 0:
            return np.inf
        output = self.nfw_profile_log((rho_0, r_s), r)
        return output

    def cored_cost_function(self, params, r):
        """
        Used to fit the cored profile to the DM density profile of the host galaxy.

        Parameters
        ----------
        params: tuple
            The cored parameters.
        r: numpy.ndarray
            The radius.

        Returns
        -------
        predicted: numpy.ndarray
            The predicted DM density profile of the host galaxy (log).

        Notes
        -----
        The cored profile (log) is calculated using the cored parameters and the radius.

        Examples
        --------
        >>> cored_cost_function((5e6 * u.Msun / u.kpc**3, 40 * u.kpc, 0.2), 10 * u.kpc)
        """
        rho_0, r_s, r_c, gamma_c = params
        # set conditions for parameters due to physical constraints
        if rho_0 <= 0 or r_s <= 0 or r_c <= 0 or gamma_c < 0 or gamma_c >= 1:
            return np.inf
        output = self.cored_profile_log((rho_0, r_s, r_c, gamma_c), r)
        return output

    def cored_cost_function_fixed_gamma_c(self, params, r):
        """
        Used to fit the cored profile to the DM density profile of the host galaxy.

        Parameters
        ----------
        params: tuple
            The cored parameters.
        r: numpy.ndarray
            The radius.

        Returns
        -------
        predicted: numpy.ndarray
            The predicted DM density profile of the host galaxy (log).

        Notes
        -----
        The cored profile (log) is calculated using the cored parameters and the radius.

        Examples
        --------
        >>> cored_cost_function((5e6 * u.Msun / u.kpc**3, 40 * u.kpc, 0.2), 10 * u.kpc)
        """
        rho_0, r_s, r_c = params
        # set conditions for parameters due to physical constraints
        if rho_0 <= 0 or r_s <= 0 or r_c <= 0:
            return np.inf
        output = self.cored_profile_log((rho_0, r_s, r_c, self.core_index), r)
        return output

    def nfw_fit(self):
        """
        Fit the NFW profile to the DM density profile of the host galaxy.

        Returns
        -------
        rho_0: astropy.units.quantity.Quantity
            The NFW normalisation parameter.
        r_s: astropy.units.quantity.Quantity
            The NFW scale radius.

        Notes
        -----
        The NFW profile is fitted using the scipy.optimize.minimize function.
        """
        table_galaxy_zf = self.get_table_galaxy_zf()

        # return NaN if BH host halo at formation redshift is spurious
        if self.spurios:
            return(np.nan, np.nan)

        # extract DM halo profile of halo in which BH formed at formation redshift
        self.ap_size = table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        ap_mass = table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun

        # compute DM density profile of halo in which BH formed at formation redshift
        self.dm_rho_log = np.log(self.density_within_aperature(self.ap_size, ap_mass).value) * u.Msun / u.kpc**3

        if hasattr(self.ap_size, 'unit'):
            self.ap_size = self.ap_size.to(u.kpc).value
        if hasattr(self.dm_rho_log, 'unit'):
            self.dm_rho_log = self.dm_rho_log.to(u.Msun/u.kpc**3).value
        model = Model(self.nfw_cost_function)
        data = RealData(self.ap_size, self.dm_rho_log)
        odr = ODR(data, model, beta0=[5e6, 40]) 

        odr.set_job(fit_type=2)
        output = odr.run()
        popt = output.beta
        rho_0, r_s = popt

        self.rho_0 = rho_0 * u.Msun / u.kpc**3
        self.r_s = r_s * u.kpc

        return(self.rho_0, self.r_s)

    def cored_fit(self):
        """
        Fit the cored profile to the DM density profile of the host galaxy.

        Returns
        -------
        rho_0: astropy.units.quantity.Quantity
            The cored normalisation parameter.
        r_s: astropy.units.quantity.Quantity
            The cored scale radius.
        r_c: astropy.units.quantity.Quantity
            The core radius.
        gamma_c: float
            The cored core index.

        Notes
        -----
        The cored profile is fitted using the scipy.optimize.minimize function.
        """
        table_galaxy_zf = self.get_table_galaxy_zf()

        # return NaN if BH host halo at formation redshift is spurious
        if self.spurios:
            return(np.nan, np.nan, np.nan, np.nan)

        # extract DM halo profile of halo in which BH formed at formation redshift
        self.ap_size = table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        ap_mass = table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun

        # compute DM density profile of halo in which BH formed at formation redshift
        self.dm_rho_log = np.log(self.density_within_aperature(self.ap_size, ap_mass).value) * u.Msun / u.kpc**3

        if hasattr(self.ap_size, 'unit'):
            self.ap_size = self.ap_size.to(u.kpc).value
        if hasattr(self.dm_rho_log, 'unit'):
            self.dm_rho_log = self.dm_rho_log.to(u.Msun/u.kpc**3).value
        model = Model(self.cored_cost_function)
        # model = Model(self.cored_cost_function_fixed_gamma_c)
        data = RealData(self.ap_size, self.dm_rho_log)
        odr = ODR(data, model, beta0=[5e6, 10, 3, 0.4]) #[5e6, 40, 5, 0.4]
        # odr = ODR(data, model, beta0=[5e6, 10, 3]) 

        odr.set_job(fit_type=2)
        output = odr.run()
        popt = output.beta
        rho_0, r_s, r_c, gamma_c = popt
        # rho_0, r_s, r_c = popt
        # gamma_c = 0

        self.rho_0 = rho_0 * u.Msun / u.kpc**3
        self.r_s = r_s * u.kpc
        self.r_c = r_c * u.kpc
        self.gamma_c = gamma_c

        return(self.rho_0, self.r_s, self.r_c, self.gamma_c)

    def cored_fit_fixed_gamma_c(self):
        """
        Fit the cored profile with fixed core index to the DM density profile of the host galaxy.

        Returns
        -------
        rho_0: astropy.units.quantity.Quantity
            The cored normalisation parameter.
        r_s: astropy.units.quantity.Quantity
            The cored scale radius.
        r_c: astropy.units.quantity.Quantity
            The core radius.
        gamma_c: float
            The cored core index.

        Notes
        -----
        The cored profile is fitted using the scipy.optimize.minimize function.
        """
        table_galaxy_zf = self.get_table_galaxy_zf()

        # return NaN if BH host halo at formation redshift is spurious
        if self.spurios:
            return(np.nan, np.nan, np.nan, np.nan)

        # extract DM halo profile of halo in which BH formed at formation redshift
        ap_size = table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        ap_mass = table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun

        # compute DM density profile of halo in which BH formed at formation redshift
        dm_rho_log = np.log(self.density_within_aperature(ap_size, ap_mass).value) * u.Msun / u.kpc**3

        if hasattr(ap_size, 'unit'):
            ap_size = ap_size.to(u.kpc).value
        if hasattr(dm_rho_log, 'unit'):
            dm_rho_log = dm_rho_log.to(u.Msun/u.kpc**3).value
        model = Model(self.cored_cost_function_fixed_gamma_c)
        data = RealData(ap_size, dm_rho_log)
        odr = ODR(data, model, beta0=[5e6, 10, 3]) 

        odr.set_job(fit_type=2)
        output = odr.run()
        popt = output.beta
        rho_0, r_s, r_c = popt

        self.rho_0 = rho_0 * u.Msun / u.kpc**3
        self.r_s = r_s * u.kpc
        self.r_c = r_c * u.kpc
        self.gamma_c = self.core_index

        return(self.rho_0, self.r_s, self.r_c, self.gamma_c)

    def radius_gravitational_influence_equation(self, r, rho_0, r_s, M_bh, r_c = None, gamma_c = None):
        """
        Equation used to calculate the radius of the gravitational influence of the black hole.

        Parameters
        ----------
        r: astropy.units.quantity.Quantity
            The radius.
        rho_0: astropy.units.quantity.Quantity
            The NFW normalisation parameter.
        r_s: astropy.units.quantity.Quantity
            The NFW scale radius.
        M_bh: astropy.units.quantity.Quantity
            The black hole mass.
        r_c: astropy.units.quantity.Quantity
            The core radius.
        gamma_c: float
            The core index.

        Returns
        -------
        y: numpy.ndarray
            The difference between the left and right hand side of the equation.

        Notes
        -----
        The equation is calculated using the astropy.units.quantity.Quantity class.

        Examples
        --------
        r = np.array([10, 20, 30]) * u.kpc
        rho_0 = 5e6 * u.Msun / u.kpc**3
        r_s = 40 * u.kpc
        M_bh = 10**5 * u.Msun
        >>> radius_gravitational_influence_equation(r, rho_0, r_s, M_bh)
        """
        if self.dm_profile == "nfw":
            y1 = np.log(nfw_integral(r, rho_0, r_s).value)
        elif self.dm_profile == "cored":
            y1 = np.log(cored_integral(r, rho_0, r_s, r_c, gamma_c).value) 

        y2 = np.log(M_bh_2(M_bh).value)
        y = (y1 - y2)
        y = y
        return(y)
    
    def radius_gravitational_influence(self, rho_0, r_s, M_bh, r_c = None, gamma_c = None):
        """
        Calculate the radius of the gravitational influence of the black hole.

        Parameters
        ----------
        rho_0: astropy.units.quantity.Quantity
            The NFW normalisation parameter.
        r_s: astropy.units.quantity.Quantity
            The NFW scale radius.
        M_bh: astropy.units.quantity.Quantity
            The black hole mass.
        r_c: astropy.units.quantity.Quantity
            The core radius.
        gamma_c: float
            The core index.

        Returns
        -------
        r_h: astropy.units.quantity.Quantity
            The radius of the gravitational influence of the black hole.

        Notes
        -----
        The radius of the gravitational influence of the black hole is calculated using the scipy.optimize.fsolve function.

        Examples
        --------
        rho_0 = 5e6 * u.Msun / u.kpc**3
        r_s = 40 * u.kpc
        M_bh = 10**5 * u.Msun
        >>> radius_gravitational_influence(rho_0, r_s, M_bh)
        """
        # Generate a range of r values
        r_min, r_max = 0.1e-3 * u.kpc, 200e-3 * u.kpc #kpc
        r = np.linspace(r_min, r_max, 1000)
        r_log = np.log(r.value)

        # Compute the function values
        if self.dm_profile == "nfw":
            y_log = self.radius_gravitational_influence_equation(r, rho_0, r_s, M_bh)
        elif self.dm_profile == "cored":
            y_log = self.radius_gravitational_influence_equation(r, rho_0, r_s, M_bh, r_c, gamma_c)

        # Fit a linear function to the logarithmic data
        coefficients = np.polyfit(r_log, y_log, 1)

        # Obtain the optimal r value from the linear fit
        r_h = -coefficients[1] / coefficients[0]
        r_h = np.exp(r_h) * u.kpc

        return(r_h)

    @property
    def r_h(self):
        """
        The radius of the gravitational influence of the black hole.

        Returns
        -------
        r_h: astropy.units.quantity.Quantity
            The radius of the gravitational influence of the black hole.
        """
        # r_h = self.radius_gravitational_influence(self.rho_0, self.r_s, self.bh_mass_formation) # use BH mass at its formation
        if self.dm_profile == "nfw":
            r_h = self.radius_gravitational_influence(self.rho_0, self.r_s, self.bh_mass) 
        elif self.dm_profile == "cored":
            r_h = self.radius_gravitational_influence(self.rho_0, self.r_s, self.bh_mass, r_c = self.r_c, gamma_c = self.gamma_c) 
        return(r_h)

    @property
    def r_sp(self):
        """
        The dark matter mini-spike radius.

        Returns
        -------
        r_sp: astropy.units.quantity.Quantity
            The dark matter mini-spike radius.
        """
        r_sp = 0.2 * self.r_h
        return(r_sp)

    @property
    def rho_at_r_sp(self):
        """
        The dark matter mini-spike density.

        Returns
        -------
        rho_sp: astropy.units.quantity.Quantity
            The dark matter mini-spike density.
        """
        if self.dm_profile == "nfw":
            rho_at_r_sp = nfw_profile((self.rho_0, self.r_s), self.r_sp)
        elif self.dm_profile == "cored":
            rho_at_r_sp = cored_profile((self.rho_0, self.r_s, self.r_c, self.gamma_c), self.r_sp)
        return(rho_at_r_sp)

    @property
    def spike_index(self):
        """
        The dark matter mini-spike index.

        Returns
        -------
        gamma_sp: float
            The dark matter mini-spike index.
        """
        if self.dm_profile == "nfw":
            gamma = 1
        elif self.dm_profile == "cored":
            gamma = self.gamma_c
        gamma_sp = (9 - 2 * gamma) / (4 - gamma)
        return(gamma_sp)

    def plot_nfw(self, path):
        """
        Plot the NFW fit.

        Parameters
        ----------
        path: str
            The path to the folder where the plots are saved.

        Notes
        -----
        The NFW fit is plotted using the damspi.plot.BlackHolePlotter class.
        """
        os.makedirs(path, exist_ok = True)
        galaxy_zf_mass = self.table_galaxy_zf["m"].values[0] * u.Msun
        galaxy_zf_z = self.table_galaxy_zf["z"].values[0]
        ap_size = self.table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        m_dm_ap = self.table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun
        rho = self.density_within_aperature(ap_size, m_dm_ap)

        bh_plotter = damplot.BlackHolePlotter(sim_name = self.sim_name, table_bh = self.table_bh)

        bh_plotter.plot_nfw(ap_size, rho, self.rho_0, self.r_s, galaxy_zf_mass, galaxy_zf_z, path)

    def plot_cored(self, path):
        """
        Plot the cored profile fit.

        Parameters
        ----------
        path: str
            The path to the folder where the plots are saved.

        Notes
        -----
        The cored profile fit is plotted using the damspi.plot.BlackHolePlotter class.
        """
        os.makedirs(path, exist_ok = True)
        galaxy_zf_mass = self.table_galaxy_zf["m"].values[0] * u.Msun
        galaxy_zf_z = self.table_galaxy_zf["z"].values[0]
        ap_size = self.table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        m_dm_ap = self.table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun
        rho = self.density_within_aperature(ap_size, m_dm_ap)

        bh_plotter = damplot.BlackHolePlotter(sim_name = self.sim_name, table_bh = self.table_bh)

        bh_plotter.plot_cored(ap_size, rho, self.rho_0, self.r_s, self.r_c, self.gamma_c, galaxy_zf_mass, galaxy_zf_z, path)

    def plot_radius_gravitational_influence(self, path):
        """
        Plot the radius of the gravitational influence of the black hole fit.

        Parameters
        ----------
        path: str
            The path to the folder where the plots are saved.

        Notes
        -----
        The radius of the gravitational influence of the black hole fit is plotted using the damspi.plot.BlackHolePlotter class.
        """
        galaxy_zf_mass = self.table_galaxy_zf["m"].values[0] * u.Msun
        galaxy_zf_z = self.table_galaxy_zf["z"].values[0]

        bh_plotter = damplot.BlackHolePlotter(sim_name = self.sim_name, table_bh = self.table_bh)

        if self.dm_profile == "nfw":
            bh_plotter.plot_radius_gravitational_influence_nfw(
                r_h = self.r_h, 
                rho_0 = self.rho_0, 
                r_s = self.r_s, 
                M_bh = self.bh_mass, 
                galaxy_mass = galaxy_zf_mass, 
                z = galaxy_zf_z, 
                path = path
                )
        elif self.dm_profile == "cored":
            bh_plotter.plot_radius_gravitational_influence_cored(
                r_h = self.r_h, 
                rho_0 = self.rho_0, 
                r_s = self.r_s, 
                r_c = self.r_c,
                gamma_c = self.gamma_c,
                M_bh = self.bh_mass, 
                galaxy_mass = galaxy_zf_mass, 
                z = galaxy_zf_z, 
                path = path
                )
