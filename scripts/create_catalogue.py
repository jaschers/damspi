import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import dammspi.catalogue as dammcat
import dammspi.plot as dammplot
from dammspi.utils import convert_to_bool, parse_args
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import astropy.constants as const
import astropy.units as u
import multiprocessing as mp
from functools import partial


# define function to process each MW-like galaxy within the EAGLE simulation with multiprocessing
def determine_coordinates(table_galaxy_z0_total, table_bh_z0_total, args, lst, galaxy_id):
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
    # table_bh_z0["(sub)group number: 2^30"] = False
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
        "satellite" 
        # "(sub)group number: 2^30"
        ]].reset_index(drop = True)

    # add table to list
    lst.append(table_bh_z0)

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


# define function to process each row of the BH catalogue with multiprocessing
def calculate_spikes(args, lst, row_tuple):
    index, row = row_tuple
    bh_id = row["bh_id"]
    galaxy_id = row["galaxy_id"]
    nsnap_c = row["nsnap_c"]

    # initialize empty table
    table = pd.DataFrame()

    data_collector = dammcat.DataCollector(sim_name = args.sim_name, number_files = args.number_files)
    table_bh_zf_total = data_collector.bh_data(nsnap=nsnap_c)
    table_bh_zf = table_bh_zf_total[table_bh_zf_total["bh_id"] == bh_id]

    dm_mini_spikes = dammcat.DMMiniSpikesCalculator(sim_name = args.sim_name, table_bh = table_bh_zf)

    # extract mini spike parameters
    # spike radius
    r_sp = dm_mini_spikes.r_sp

    # DM density at spike radius
    rho_at_r_sp = dm_mini_spikes.rho_at_r_sp
    rho_at_r_sp = (rho_at_r_sp * const.c ** 2).to(u.GeV / u.cm ** 3)

    # check if BH has a galaxy host at formation redshift
    no_host = dm_mini_spikes.no_host

    # add mini spike parameters to table
    table["bh_id"] = [bh_id]
    table["r_sp [pc]"] = [r_sp.to(u.pc).value]
    table["rho(r_sp) [GeV/cm3]"] = [rho_at_r_sp.value]
    table["no_host"] = [no_host]
    # bh_catalogue.loc[bh_catalogue["bh_id"] == bh_id, "r_sp [pc]"] = r_sp.to(u.pc).value
    # bh_catalogue.loc[bh_catalogue["bh_id"] == bh_id, "rho(r_sp) [GeV/cm3]"] = rho_at_r_sp.value

    lst.append(table)

    if args.plot:
        path = f"plots/{args.sim_name}/galaxy_id_{int(galaxy_id)}/black_holes/mini_spikes/id_{int(bh_id)}/"
        # plot mini spike parameters
        dm_mini_spikes.plot_nfw(path)
        dm_mini_spikes.plot_radius_gravitational_influence(path)


if __name__ == "__main__":
    # initialise user input
    args = parse_args()

    # Set the number of processes to the number of available CPU cores or adjust as needed
    num_processes = mp.cpu_count()
    print("Number of CPU cores available:", num_processes)

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/"
    os.makedirs(path_catalogue, exist_ok = True)

    # extract galaxy data at z = 0
    data_collector = dammcat.DataCollector(sim_name = args.sim_name, number_files = args.number_files)
    table_galaxy_z0_total = data_collector.galaxy_data(nsnap = 28)

    # unique galaxy root ids
    galaxy_id_unique = np.unique(table_galaxy_z0_total["galaxy_id"])

    # extract bh data at z = 0
    table_bh_z0_total = data_collector.bh_data(nsnap = 28)

    if args.plot:
        # save galaxy images
        galaxy_plotter = dammplot.GalaxyPlotter(sim_name = args.sim_name, table_galaxy = table_galaxy_z0_total, table_bh = table_bh_z0_total)
        galaxy_plotter.save_gri_images()

        # plot galaxy properties distributions, such as 
        galaxy_plotter.plot_galaxy_distributions()

    # Create a multiprocessing manager list to share the pandas tables between processes
    coord_list = mp.Manager().list()

    # Initialize the multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # add necesarry arguments to the function except of galaxy_id_unique since this is the variable to loop over
        determine_coordinates_with_args = partial(
            determine_coordinates, 
            table_galaxy_z0_total, 
            table_bh_z0_total, 
            args, 
            coord_list
            )
        # Use tqdm to visualize the progress of the loop
        # loop over all individual galaxies to extract coordinates of BHs
        for _ in tqdm(pool.imap_unordered(determine_coordinates_with_args, galaxy_id_unique), total=len(galaxy_id_unique)):
            pass

    # combine all tables into the BH catalogue
    bh_catalogue = pd.concat(coord_list, ignore_index=True)

    print(f"Number of unmerged black holes in simulation {args.sim_name}: {len(bh_catalogue)}")
    print("Extract black hole dark matter mini spike parameter for each BH...")

    # Create a multiprocessing manager list to share the pandas tables between processes
    spikes_list = mp.Manager().list()

    # calculate dark matter mini spike parameters for each BH
    # Initialize the multiprocessing pool
    # loop over all individual BHs to calculate mini spike parameters
    with mp.Pool(processes=num_processes) as pool:
        # add necesarry arguments to the function except of the table rows since this is the variable to loop over
        calculate_spikes_with_args = partial(calculate_spikes, args, spikes_list)
        # Use tqdm to visualize the progress of the loop
        for _ in tqdm(pool.imap_unordered(calculate_spikes_with_args, bh_catalogue.iterrows()), total = len(bh_catalogue)):
            pass

    # combine all mini spike tables into a single table
    spikes_table = pd.concat(spikes_list, ignore_index=True)

    # merge mini spike table with BH catalogue
    bh_catalogue = bh_catalogue.merge(spikes_table, on = "bh_id", how = "left")

    # save BH catalogue
    bh_catalogue.to_csv(path_catalogue + "catalogue.csv", index=False)
