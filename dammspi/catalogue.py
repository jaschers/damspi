import numpy as np
import pandas as pd
import eagleSqlTools as sql
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import h5py
from dammspi.utils import (
    convert_float_to_three_digit, 
    convert_float_to_six_digit, 
    redshift
    )

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
        query for EAGLE galaxy data
        """
        if nsnap == 28:
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
                        DES.Snapnum = 28 \
                        and DES.Mass between 1.3e12 and 2.3e12 \
                        and DES.Spurious = 0"

        # Execute query.
        con = sql.connect("dvd351", password="zqfARI55") # username, password
        print("Loading galaxy data...")
        # load galaxy data
        data_galaxy = sql.execute_query(con, query)
        print("galaxy data successfully loaded!")
        # put data into pandas table
        table_galaxy = pd.DataFrame(data_galaxy)
        
        if nsnap == 28:
            # load galaxy outlier
            table_outlier = pd.read_csv(f"config/{self.sim_name}.csv")
            outlier = table_outlier["galaxy_id"].values
            # remove outlier from table
            table_galaxy = table_galaxy[~table_galaxy["galaxy_id"].isin(outlier)].reset_index(drop = True)

        return(table_galaxy)

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
        print("Loading black hole data...")
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
        print("Black hole data successfully loaded!")

        # stack data 
        data_bh = np.dstack((bh_id, bh_group_number, bh_subgroup_number, bh_subgrid_mass, bh_coordinates[:,0], bh_coordinates[:,1], bh_coordinates[:,2], bh_formation_redshift, bh_n_merger))[0]

        # put data into pandas table
        table_bh = pd.DataFrame(data_bh, columns = ["BH ID", "group number", "subgroup number", "m [M_solar]", "coord x", "coord y", "coord z", "z_f", "n merger"]).reset_index(drop = True)

        return(table_bh)

class CoordinateTransform:
    def __init__(self, table_galaxy, table_bh):
        self.table_galaxy = table_galaxy
        self.group_number = self.table_galaxy["group_number"].values[0]
        self.table_bh = table_bh[table_bh["group number"] == group_number]

        # add galaxy root id to table
        self.table_bh["galaxy ID"] = np.ones(len(self.table_bh)) * galaxy_id
    
    def centre_of_mass(self):
        com = self.table_galaxy[["com_x", "com_y", "com_z"]].values[0] * 1e3 # convert to kpc
        return(com)
    
    def spin_vector(self):
        # spin vector of galaxy to rotate galaxy according to galactic plane
        spin = self.table_galaxy[["spin_x", "spin_y", "spin_z"]].values[0]
        # norm spin vector
        spin = spin / np.linalg.norm(spin)
        return(spin)

    def bh_coord_gc(self):
        # get galaxy centre
        galaxy_centre = self.centre_of_mass()
        # to be continued
        # extract coordinates of BHs relative to galaxy centre
        bh_coord_z0_gc = np.dstack((table_bh_z0["coord x"] - galaxy_centre[0], table_bh_z0["coord y"] - galaxy_centre[1], table_bh_z0["coord z"] - galaxy_centre[2]))[0]