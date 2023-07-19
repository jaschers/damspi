import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import numpy as np
import pandas as pd
import eagleSqlTools as sql
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import h5py
from dammspi.utils import (
    convert_float_to_three_digit, 
    convert_float_to_six_digit, 
    redshift,
    nfw_profile,
    nfw_integral,
    M_bh_2
    )
from scipy.spatial.transform import Rotation as R
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from scipy.odr import Model, RealData, ODR
import dammspi.plot as dammplot

# ignore division by zero
np.seterr(divide='ignore')
# don't print pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

class DataCollector:
    def __init__(self, sim_name, number_files):
        self.sim_name = sim_name
        self.number_files = number_files
        self.nsnap_list = np.linspace(0, 28, 29)
        self.redshift_list = [20., 15.132, 9.993, 8.988, 8.075, 7.050, 5.971, 5.487, 5.037, 4.485, 3.984, 3.528, 3.017, 2.478, 2.237, 2.012, 1.737, 1.487, 1.259, 1.004, 0.865, 0.736, 0.615, 0.503, 0.366, 0.271, 0.183, 0.101, 0.]
        self.dict_redshift = dict(zip(self.nsnap_list, self.redshift_list))
        self.hubble_constant = cosmo.H0.value / 100
        self.minimal_galaxy_mass = 10**10 * u.Msun / self.hubble_constant
        self.bh_mass_formation = 10**5 * u.Msun / self.hubble_constant

    def galaxy_data(self, nsnap):
        """
        Get the galaxy data for snapshot nsnap from the EAGLE simulations
        """
        query = f"SELECT \
                    DES.GalaxyID as galaxy_id, \
                    DES.GroupID as group_id, \
                    DES.GroupNumber as group_number, \
                    DES.SubGroupNumber as subgroup_number, \
                    DES.Redshift as z, \
                    DES.Snapnum as nsnap, \
                    DES.Mass as m, \
                    DES.CentreOfMass_x as com_x, \
                    DES.CentreOfMass_y as com_y, \
                    DES.CentreOfMass_z as com_z, \
                    DES.Stars_Spin_x as spin_x, \
                    DES.Stars_Spin_y as spin_y, \
                    DES.Stars_Spin_z as spin_z, \
                    DES.Image_face as img_face, \
                    DES.Image_edge as img_edge, \
                    DES.Image_box as img_box \
                FROM \
                    {self.sim_name}_SubHalo as DES \
                WHERE \
                    DES.Snapnum = {nsnap} \
                    and DES.Mass between 1.3e12 and 2.3e12 \
                    and DES.Spurious = 0"

        # Execute query.
        con = sql.connect("dvd351", password="zqfARI55") # username, password
        # load galaxy data
        data_galaxy = sql.execute_query(con, query)
        # put data into pandas table
        self.table_galaxy = pd.DataFrame(data_galaxy)
        
        if nsnap == 28:
            # load galaxy outlier
            table_outlier = pd.read_csv(f"config/{self.sim_name}.csv")
            outlier = table_outlier["galaxy_id"].values
            # remove outlier from table
            self.table_galaxy = self.table_galaxy[~self.table_galaxy["galaxy_id"].isin(outlier)].reset_index(drop = True)

        return(self.table_galaxy)

    @staticmethod
    def find_next_smallest(dictionary, values):
        """
        Find the next smallest value in a dictionary.
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

    def read_dataset(self, itype, att, nsnap):
        """ Read a selected dataset, itype is the PartType and att is the attribute name. """

        # Output array.
        data = []

        redshift = self.dict_redshift[int(nsnap)]
        nsnap = convert_float_to_three_digit(nsnap)
        redshift_1, redshift_2 = convert_float_to_six_digit(redshift)

        # Loop over each file and extract the PARTICLE data.
        for i in range(self.number_files):
            # f = h5py.File("./data/snap_028_z000p000.%i.hdf5"%i, "r")
            f = h5py.File(f"/lfs/l7/hess/users/jaschers/eagle/data/{self.sim_name}/snapshot_{nsnap}_z{redshift_1}p{redshift_2}/snap_{nsnap}_z{redshift_1}p{redshift_2}.%i.hdf5"%i, "r")

            tmp = f["PartType%i/%s"%(itype, att)][...]
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

    def bh_data(self, nsnap):
        # extract BH values at z = 0
        # bh_mass = (self.read_dataset(itype = 5, att = "Mass", nsnap = 28) * u.g).to(u.M_sun).value
        bh_subgrid_mass = (self.read_dataset(itype = 5, att = "BH_Mass", nsnap = nsnap) * u.g).to(u.M_sun).value
        bh_n_merger = self.read_dataset(itype = 5, att = "BH_CumlNumSeeds", nsnap = nsnap) 
        bh_coordinates = (self.read_dataset(itype = 5, att = "Coordinates", nsnap = nsnap) * u.cm).to(u.kpc).value
        # bh_time_last_merger = redshift(self.read_dataset(itype = 5, att = "BH_TimeLastMerger", nsnap = nsnap)) 
        bh_group_number = self.read_dataset(itype = 5, att = "GroupNumber", nsnap = nsnap)
        bh_subgroup_number = self.read_dataset(itype = 5, att = "SubGroupNumber", nsnap = nsnap)
        bh_id = self.read_dataset(itype = 5, att = "ParticleIDs", nsnap = nsnap)
        bh_formation_redshift = redshift(self.read_dataset(itype = 5, att = "BH_FormationTime", nsnap = nsnap))

        nsnap_closest, z_closest = self.find_next_smallest(self.dict_redshift, bh_formation_redshift)

        # stack data 
        data_bh = np.dstack((bh_id, bh_group_number, bh_subgroup_number, bh_subgrid_mass, bh_coordinates[:,0], bh_coordinates[:,1], bh_coordinates[:,2], bh_formation_redshift, z_closest, nsnap_closest, bh_n_merger))[0]

        # put data into pandas table
        self.table_bh = pd.DataFrame(data_bh, columns = ["bh_id", "group number", "subgroup number", "m [M_solar]", "coord x", "coord y", "coord z", "z_f", "z_c", "nsnap_c", "n merger"]).reset_index(drop = True)

        # select only BHs that did not merge
        self.table_bh = self.table_bh[self.table_bh["n merger"] == 1]

        return(self.table_bh)


class CoordinateTransformer:
    def __init__(self, table_galaxy, table_bh):
        # shift coorindates system to position of the sun (8.33 kpc), https://iopscience.iop.org/article/10.1088/1475-7516/2011/03/051/pdf
        self.distance_sun = 8.33 # kpc

        self.table_galaxy = table_galaxy
        self.galaxy_id = self.table_galaxy["galaxy_id"].values[0]
        self.galaxy_centre = self.table_galaxy[["com_x", "com_y", "com_z"]].values[0] * 1e3 # convert to kpc
        self.galaxy_spin = self.table_galaxy[["spin_x", "spin_y", "spin_z"]].values[0] 
        self.galaxy_spin = self.galaxy_spin / np.linalg.norm(self.galaxy_spin)

        self.table_bh = table_bh
        self.bh_coord = self.table_bh[["coord x", "coord y", "coord z"]].values
    
    @property
    def bh_coord_gc(self):
        """
        get black hole coordinates with respect to the galaxy centre
        """
        bh_coord_gc = self.bh_coord - self.galaxy_centre
        return(bh_coord_gc)

    @property
    def distance_gc(self):
        """
        get distance between black hole and galaxy centre
        """
        distance_gc = np.linalg.norm(self.bh_coord_gc, axis = 1)
        return(distance_gc)

    @staticmethod
    def rot_angle_x(vec):
        """
        get rotation angle around x-axis
        """
        theta = np.arctan(vec[1] / vec[2])
        return(theta)

    @staticmethod
    def rot_angle_y(vec):
        """
        get rotation angle around y-axis
        """
        if vec[2] < 0:
            theta = np.arctan(- vec[0] / vec[2]) + np.pi
        else:
            theta = np.arctan(- vec[0] / vec[2])
        return(theta)

    def rot_matrices_x_y(self, vec):
        """
        get rotation matrices around x- and y-axis
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
        rotate vector around x- and y-axis
        """
        vec_rot_x = rot_matrix_x.apply(vec)
        vec_rot_xy = rot_matrix_y.apply(vec_rot_x)

        return(vec_rot_xy)

    @property
    def bh_coord_gc_rot(self):
        """
        rotate black hole coordinates around x- and y-axis
        """
        rot_matrix_x, rot_matrix_y = self.rot_matrices_x_y(self.galaxy_spin)
        self.galaxy_spin_rot = self.rotate_x_y(self.galaxy_spin, rot_matrix_x, rot_matrix_y)
        bh_coord_gc_rot = self.rotate_x_y(self.bh_coord_gc, rot_matrix_x, rot_matrix_y)
        return(bh_coord_gc_rot)

    @property
    def bh_coord_gc_rot_shifted(self):
        """
        shift black hole coordinates to position of the sun
        """
        bh_coord_gc_rot_shifted = self.bh_coord_gc_rot
        bh_coord_gc_rot_shifted[:,0] += self.distance_sun
        return(bh_coord_gc_rot_shifted)

    @property
    def bh_galactic_coord(self):
        """
        get black hole coordinates in galactic coordinate system
        """
        r, lat, long = cartesian_to_spherical(
            self.bh_coord_gc_rot_shifted[:,0], 
            self.bh_coord_gc_rot_shifted[:,1], 
            self.bh_coord_gc_rot_shifted[:,2]
            )
        r, lat, long = r.value, lat.value, long.value
        return(r, lat, long)

    def plot_3d_maps(self, sim_name, path, save_animation = False):
        galaxy_plotter = dammplot.GalaxyPlotter(sim_name = sim_name, table_galaxy = self.table_galaxy, table_bh = self.table_bh)

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
            self.bh_coord_gc_rot_shifted, 
            self.galaxy_spin_rot, 
            self.galaxy_id, 
            path = path + "3d_map_rot_shifted", 
            shifted = True,
            save_animation = save_animation
            )


class DMMiniSpikesCalculator:
    def __init__(self, sim_name, table_bh):
        self.sim_name = sim_name
        self.table_bh = table_bh
        self.group_number = self.table_bh["group number"].values[0]
        self.subgroup_number = self.table_bh["subgroup number"].values[0]
        self.z_formation = table_bh["z_f"].values[0]
        self.z_closest = table_bh["z_c"].values[0]
        self.nsnap_closest = table_bh["nsnap_c"].values[0]
        self.bh_coord = table_bh[["coord x", "coord y", "coord z"]].values[0]
        self.no_host = (self.group_number == 2**30) or (self.subgroup_number == 2**30)
        self.hubble_constant = cosmo.H0.value / 100
        self.minimal_galaxy_mass = 10**10 * u.Msun / self.hubble_constant
        self.bh_mass_formation = 10**5 * u.Msun / self.hubble_constant

    @staticmethod
    def redshift_to_scale_factor(redshift):
        """
        Convert redshift to scale factor using Astropy.
        
        Parameters:
            redshift (float): The redshift value.
            
        Returns:
            float: The corresponding scale factor.
        """
        return cosmo.scale_factor(redshift)

    @staticmethod
    def distance_two_points(coord_1, coord_2):
        """
        Calculate the distance between two points.
        
        Parameters:
            coord_1 (array): The coordinates of the first point.
            coord_2 (array): The coordinates of the second point.
            
        Returns:
            float: The distance between the two points.
        """
        return np.linalg.norm(coord_1 - coord_2, axis = 1)

    def query_galaxy_zf(self):
        if self.no_host:
            query = f"SELECT \
                        DES.GroupID as group_id, \
                        DES.GroupNumber as group_number, \
                        DES.SubGroupNumber as subgroup_number, \
                        DES.Redshift as z, \
                        DES.Snapnum as nsnap, \
                        DES.Mass as m, \
                        DES.CentreOfMass_x as com_x, \
                        DES.CentreOfMass_y as com_y, \
                        DES.CentreOfMass_z as com_z, \
                        DES.Image_face as img_face, \
                        DES.Image_edge as img_edge, \
                        DES.Image_box as img_box, \
                        AP.ApertureSize as aperture_size, \
                        AP.Mass_DM as m_dm_ap \
                    FROM \
                        {self.sim_name}_SubHalo as DES, \
                        {self.sim_name}_Aperture as AP \
                    WHERE \
                        DES.Snapnum = {self.nsnap_closest} \
                        and AP.GalaxyID = DES.GalaxyID \
                        and DES.Spurious = 0"
        else:
            query = f"SELECT \
                        DES.GroupID as group_id, \
                        DES.GroupNumber as group_number, \
                        DES.SubGroupNumber as subgroup_number, \
                        DES.Redshift as z, \
                        DES.Snapnum as nsnap, \
                        DES.Mass as m, \
                        DES.CentreOfMass_x as com_x, \
                        DES.CentreOfMass_y as com_y, \
                        DES.CentreOfMass_z as com_z, \
                        DES.Image_face as img_face, \
                        DES.Image_edge as img_edge, \
                        DES.Image_box as img_box, \
                        AP.ApertureSize as aperture_size, \
                        AP.Mass_DM as m_dm_ap \
                    FROM \
                        {self.sim_name}_SubHalo as DES, \
                        {self.sim_name}_Aperture as AP \
                    WHERE \
                        DES.Snapnum = {self.nsnap_closest} \
                        and DES.GroupNumber = {self.group_number} \
                        and DES.SubGroupNumber = {self.subgroup_number} \
                        and AP.GalaxyID = DES.GalaxyID \
                        and DES.Spurious = 0"
        return(query)

    @property
    def table_galaxy_zf(self):
        # get query
        query = self.query_galaxy_zf()
        # Execute query.
        con = sql.connect("dvd351", password="zqfARI55") # username, password
        # load galaxy data
        data_galaxy = sql.execute_query(con, query)
        # put data into pandas table
        table_galaxy_zf = pd.DataFrame(data_galaxy)

        # if no host galaxy is found, select the closest galaxy with mass > minimal_galaxy_mass
        if self.no_host:
            table_galaxy_zf = table_galaxy_zf[table_galaxy_zf["m"] >= self.minimal_galaxy_mass.value].reset_index(drop = True)

            # extract the coordinates of the galaxies at the redshift closest to BH formation redshift
            scale_factor = self.redshift_to_scale_factor(self.z_closest)
            galaxy_coord = table_galaxy_zf[["com_x", "com_y", "com_z"]].to_numpy() * scale_factor * 1e3 # convert to kpc

            # calculate the distance of the galaxies to the BH
            distance_galaxies_bh = self.distance_two_points(galaxy_coord, self.bh_coord)

            # add distances to table
            table_galaxy_zf["distance BH [kpc]"] = distance_galaxies_bh

            # select the galaxy closest to the BH
            table_galaxy_zf = table_galaxy_zf[table_galaxy_zf["distance BH [kpc]"] == np.min(distance_galaxies_bh)].reset_index(drop = True)

        # remove entries with m_dm_ap = 0 since it causes problems in the NFW fit
        table_galaxy_zf = table_galaxy_zf[table_galaxy_zf["m_dm_ap"] != 0]
        
        return(table_galaxy_zf)

    @staticmethod
    def density_within_aperature(aperture, mass):
        volume = 4/3 * np.pi * aperture**3
        rho = (mass / volume).to(u.Msun/u.kpc**3)
        return(rho)

    @staticmethod
    def nfw_profile_log(params, r):
        rho_0, r_s = params
        if hasattr(r, 'unit'):
            r = r.to(u.kpc).value
        if hasattr(rho_0, 'unit'):
            rho_0 = rho_0.to(u.Msun/u.kpc**3).value
        if hasattr(r_s, 'unit'):
            r_s = r_s.to(u.kpc).value
        rho_log = np.log(rho_0) - np.log(r/r_s) - 2*np.log(1+r/r_s) 
        return(rho_log)

    def nfw_cost_function(self, params, r):
        rho_0, r_s = params
        if rho_0 <= 0 or r_s <= 0:
            return np.inf
        predicted = self.nfw_profile_log((rho_0, r_s), r)
        return predicted

    def nfw_fit(self, r, rho):
        if hasattr(r, 'unit'):
            r = r.to(u.kpc).value
        if hasattr(rho, 'unit'):
            rho = rho.to(u.Msun/u.kpc**3).value
        model = Model(self.nfw_cost_function)
        data = RealData(r, rho)
        odr = ODR(data, model, beta0=[5e6, 40]) 

        odr.set_job(fit_type=2)
        output = odr.run()
        popt = output.beta
        rho_0, r_s = popt

        rho_0 = rho_0 * u.Msun / u.kpc**3
        r_s = r_s * u.kpc

        return(rho_0, r_s)

    def nfw_fit_para(self):
        table_galaxy_zf = self.table_galaxy_zf

        # extract DM halo profile of halo in which BH formed at formation redshift
        ap_size = table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        ap_mass = table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun

        # compute DM density profile of halo in which BH formed at formation redshift
        dm_rho_log = np.log(self.density_within_aperature(ap_size, ap_mass).value) * u.Msun / u.kpc**3

        self.rho_0, self.r_s = self.nfw_fit(ap_size, dm_rho_log)

        return(self.rho_0, self.r_s)

    def radius_gravitational_influence_equation(self, r, rho_0, r_s, M_bh):
        y1 = np.log(nfw_integral(r, rho_0, r_s).value)
        y2 = np.log(M_bh_2(M_bh).value)
        y = (y1 - y2)
        y = y
        return(y)
    
    def radius_gravitational_influence(self, rho_0, r_s, M_bh):
        # Generate a range of r values
        r_min, r_max = 0.1e-3 * u.kpc, 200e-3 * u.kpc #kpc
        r = np.linspace(r_min, r_max, 1000)
        r_log = np.log(r.value)

        # Compute the function values
        y_log = self.radius_gravitational_influence_equation(r, rho_0, r_s, M_bh)

        # Fit a linear function to the logarithmic data
        coefficients = np.polyfit(r_log, y_log, 1)

        # Obtain the optimal r value from the linear fit
        r_h = -coefficients[1] / coefficients[0]
        r_h = np.exp(r_h) * u.kpc
        return(r_h)

    @property
    def r_h(self):
        rho_0, r_s = self.nfw_fit_para()
        r_h = self.radius_gravitational_influence(rho_0, r_s, self.bh_mass_formation)
        return(r_h)

    @property
    def r_sp(self):
        r_sp = 0.2 * self.r_h
        return(r_sp)

    @property
    def rho_at_r_sp(self):
        rho_0, r_s = self.nfw_fit_para()
        rho_at_r_sp = nfw_profile((rho_0, r_s), self.r_sp)
        return(rho_at_r_sp)

    def plot_nfw(self, path):
        os.makedirs(path, exist_ok = True)
        galaxy_zf_mass = self.table_galaxy_zf["m"].values[0] * u.Msun
        galaxy_zf_z = self.table_galaxy_zf["z"].values[0]
        ap_size = self.table_galaxy_zf["aperture_size"].to_numpy() * u.kpc
        m_dm_ap = self.table_galaxy_zf["m_dm_ap"].to_numpy() * u.Msun
        rho = self.density_within_aperature(ap_size, m_dm_ap)

        bh_plotter = dammplot.BlackHolePlotter(sim_name = self.sim_name, table_bh = self.table_bh)

        bh_plotter.plot_nfw(ap_size, rho, self.rho_0, self.r_s, galaxy_zf_mass, galaxy_zf_z, path)

    def plot_radius_gravitational_influence(self, path):
        galaxy_zf_mass = self.table_galaxy_zf["m"].values[0] * u.Msun
        galaxy_zf_z = self.table_galaxy_zf["z"].values[0]

        bh_plotter = dammplot.BlackHolePlotter(sim_name = self.sim_name, table_bh = self.table_bh)

        bh_plotter.plot_radius_gravitational_influence(
            r_h = self.r_h, 
            rho_0 = self.rho_0, 
            r_s = self.r_s, 
            M_bh = self.bh_mass_formation, 
            galaxy_mass = galaxy_zf_mass, 
            z = galaxy_zf_z, 
            path = path
            )


# TO DO:
# replace data path
# replace username and password
# add cosmological parameters into config file
# add multiprocessing