import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import dammspi.catalogue as dammcat
import dammspi.plot as dammplot
from dammspi.utils import convert_to_bool
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.constants as const
import astropy.units as u


def main():
    ######################################## argparse setup ########################################
    script_descr="""
    Extracts IMBH catalogue from EAGLE data
    """

    # Open argument parser
    parser = argparse.ArgumentParser(description=script_descr)

    # Define expected arguments
    parser.add_argument("-sn", "--sim_name", type = str, required = False, default = "RefL0025N0376", metavar = "-", help = "Name of the EAGLE simulation, default: RefL0025N0376")
    parser.add_argument("-nf", "--number_files", type = int, required = False, default = 16, metavar = "-", help = "Number of files for the particle data, default: 16")
    parser.add_argument("-plt", "--plot", type = str, required = False, default = "n", metavar = "-", help = "Bool if plots for individual galaxies are saved (takes some time) [y, n], default: n")
    parser.add_argument("-sa", "--save_animation", type = str, required = False, default = "n", metavar = "-", help = "Bool if animations for individual galaxies are saved (takes a long time) [y, n], default: n")

    args = parser.parse_args()
    print("####### Setup #######")
    print(vars(args))

    args.plot = convert_to_bool(args.plot)
    args.save_animation = convert_to_bool(args.save_animation)
    ##########################################################################################

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/"
    os.makedirs(path_catalogue, exist_ok = True)

    # extract galaxy data at z = 0
    data_collector = dammcat.DataCollector(sim_name = args.sim_name, number_files = args.number_files)
    table_galaxy_z0_total = data_collector.galaxy_data(nsnap = 28)

    # unique galaxy root ids
    galaxy_id_unique = np.unique(table_galaxy_z0_total["galaxy_id"])
    print("Number of galaxies:", len(galaxy_id_unique))

    # extract bh data at z = 0
    table_bh_z0_total = data_collector.bh_data(nsnap = 28)

    if args.plot:
        # save galaxy images
        galaxy_plotter = dammplot.GalaxyPlotter(sim_name = args.sim_name, table_galaxy = table_galaxy_z0_total, table_bh = table_bh_z0_total)
        galaxy_plotter.save_gri_images()

        # plot galaxy properties distributions, such as 
        galaxy_plotter.plot_galaxy_distributions()

    # table of BH catalogue to be filled up
    bh_catalogue = pd.DataFrame()

    for count, galaxy_id in enumerate(galaxy_id_unique):
        print(f"Processing galaxy {galaxy_id} ({count + 1}/{len(galaxy_id_unique)})")

        table_galaxy_z0 = table_galaxy_z0_total[table_galaxy_z0_total["galaxy_id"] == galaxy_id]
        galaxy_group_number = table_galaxy_z0["group_number"].values[0]

        # add galaxy root id to table
        table_bh_z0 = table_bh_z0_total[table_bh_z0_total["group number"] == galaxy_group_number]

        table_bh_z0["galaxy_id"] = np.ones(len(table_bh_z0)) * galaxy_id

        coordinate_transformer = dammcat.CoordinateTransformer(table_galaxy = table_galaxy_z0, table_bh = table_bh_z0)

        # get distance of BHs to galaxy center
        r_gc = coordinate_transformer.distance_gc

        # get BH coordinates in galactic frame
        r, lat, long = coordinate_transformer.bh_galactic_coord

        # add id, radial distances, latitude and longitude, (sub)groupnumber 2^30, and satellite information to table
        table_bh_z0["galaxy_id"] = galaxy_id
        table_bh_z0["d_GC [kpc]"] = r_gc
        table_bh_z0["d_Sun [kpc]"] = r
        table_bh_z0["lat [rad]"] = lat
        table_bh_z0["long [rad]"] = long
        table_bh_z0["(sub)group number: 2^30"] = False
        table_bh_z0["satellite"] = table_bh_z0["subgroup number"] != 0

        # keep only relevant columns
        table_bh_z0 = table_bh_z0[[
            "galaxy_id", 
            "bh_id", 
            "m [M_solar]", 
            "z_f",
            "z_c",
            "nsnap_c", 
            "d_GC [kpc]", 
            "d_Sun [kpc]", 
            "lat [rad]", 
            "long [rad]", 
            "satellite", 
            "(sub)group number: 2^30"
            ]].reset_index(drop = True)

        # fill in BH catalogue
        bh_catalogue = pd.concat([bh_catalogue, table_bh_z0], ignore_index=True)

        if args.plot:
            # plot 3D maps of each galaxy
            path = f"plots/{args.sim_name}/galaxy_id_{galaxy_id}/black_holes/coordinates/"
            os.makedirs(path, exist_ok = True)
            coordinate_transformer.plot_3d_maps(sim_name = args.sim_name, path = path, save_animation = args.save_animation)

            # TO DO: add 2D maps

            # plot BH distributions for each galaxy, such as mass, formation redshift, distance to galaxy center, etc.
            bh_plotter = dammplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = table_bh_z0)

            path = f"plots/{args.sim_name}/galaxy_id_{galaxy_id}/black_holes/distributions/"
            os.makedirs(path, exist_ok = True)
            bh_plotter.plot_bh_dist_galaxy(path)


    for _, row in tqdm(bh_catalogue.iterrows(), total = len(bh_catalogue)):
        bh_id = row["bh_id"]
        galaxy_id = row["galaxy_id"]
        nsnap_c = row["nsnap_c"]

        table_bh_zf_total = data_collector.bh_data(nsnap = nsnap_c)
        table_bh_zf = table_bh_zf_total[table_bh_zf_total["bh_id"] == bh_id]
        
        dm_mini_spikes = dammcat.DMMiniSpikesCalculator(sim_name = args.sim_name, table_bh = table_bh_zf)

        # extract mini spike parameters
        # spike radius
        r_sp = dm_mini_spikes.r_sp

        # DM densitiy at spike radius
        rho_at_r_sp = dm_mini_spikes.rho_at_r_sp
        rho_at_r_sp = (rho_at_r_sp * const.c**2).to(u.GeV/u.cm**3)

        # add mini spike parameters to table
        bh_catalogue.loc[bh_catalogue["bh_id"] == bh_id, "r_sp [pc]"] = r_sp.to(u.pc).value
        bh_catalogue.loc[bh_catalogue["bh_id"] == bh_id, "rho(r_sp) [GeV/cm3]"] = rho_at_r_sp.value

        if args.plot:
            path = f"plots/{args.sim_name}/galaxy_id_{int(galaxy_id)}/black_holes/mini_spikes/id_{int(bh_id)}/"
            # plot mini spike parameters
            dm_mini_spikes.plot_nfw(path)
            dm_mini_spikes.plot_radius_gravitational_influence(path)

    print(bh_catalogue)
    bh_catalogue.to_csv(path_catalogue + "catalogue.csv", index = False)


if __name__ == "__main__":
    main()
