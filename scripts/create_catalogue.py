import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the damspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import damspi.catalogue as damcat
import damspi.plot as damplot
from damspi.utils import convert_to_bool, parse_args, remove_distant_satellites
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

    # select only BHs of the galaxy
    table_bh_z0 = table_bh_z0_total[table_bh_z0_total["group_number"] == galaxy_group_number].reset_index(drop = True)

    # remove BHs that are in satallitle galaxies not within 40 kpc and 300 kpc to match MW-like galaxies, check for stars and gas and get satellite id if applicable
    table_bh_z0 = remove_distant_satellites(table_bh_z0, nsnap = 28, args = args)

    # add galaxy id to table
    table_bh_z0["main_galaxy_id"] = np.ones(len(table_bh_z0), dtype = "int") * int(galaxy_id)

    coordinate_transformer = damcat.CoordinateTransformer(table_galaxy = table_galaxy_z0, table_bh = table_bh_z0, box_size = args.box_size)

    # get distance of BHs to galaxy center
    r_gc, lat_gc, long_gc = coordinate_transformer.bh_spherical_coord_gc

    # get BH coordinates in galactic frame
    r_sun, lat_sun, long_sun = coordinate_transformer.bh_galactic_coord

    # add id, radial distances, latitude and longitude, (sub)groupnumber 2^30, and satellite information to table
    table_bh_z0["d_GC"] = r_gc
    table_bh_z0["lat_GC"] = lat_gc
    table_bh_z0["long_GC"] = long_gc
    table_bh_z0["d_Sun"] = r_sun
    table_bh_z0["lat_Sun"] = lat_sun
    table_bh_z0["long_Sun"] = long_sun
    table_bh_z0["m_main_galaxy"] = table_galaxy_z0["m"].values[0]
    table_bh_z0["satellite"] = table_bh_z0["main_galaxy_id"] != table_bh_z0["host_galaxy_id"]
    table_bh_z0["n_satellites"] = table_galaxy_z0["n_satellites"].values[0]

    # keep only relevant columns
    table_bh_z0 = table_bh_z0[[
        "main_galaxy_id", 
        "host_galaxy_id",
        "bh_id", 
        "m", 
        "z_f",
        "z_c",
        "nsnap_c", 
        "d_GC",
        "lat_GC", 
        "long_GC",  
        "d_Sun", 
        "lat_Sun", 
        "long_Sun",
        "m_main_galaxy",
        "m_host_galaxy",
        "has_stars",
        "has_gas",
        "satellite",
        "n_satellites",
        ]].reset_index(drop = True)

    # add table to list
    lst.append(table_bh_z0)

    if args.plot:
        # plot 3D maps of each galaxy
        path = f"plots/{args.sim_name}/galaxy_id_{galaxy_id}/black_holes/coordinates/"
        os.makedirs(path, exist_ok = True)
        coordinate_transformer.plot_3d_maps(sim_name = args.sim_name, path = path, save_animation = args.save_animation)

        # plot BH distributions for each galaxy, such as mass, formation redshift, distance to galaxy center, etc.
        bh_plotter = damplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = table_bh_z0)

        path = f"plots/{args.sim_name}/galaxy_id_{galaxy_id}/black_holes/distributions/"
        os.makedirs(path, exist_ok = True)
        bh_plotter.plot_bh_dist_galaxy(path)


# define function to process each row of the BH catalogue with multiprocessing
def calculate_spikes(args, lst, row_tuple):
    index, row = row_tuple
    bh_id = row["bh_id"]
    galaxy_id = row["main_galaxy_id"]
    nsnap_c = row["nsnap_c"]

    # initialize empty table
    table = pd.DataFrame()

    data_collector = damcat.DataCollector(sim_name = args.sim_name, number_files = args.number_files)
    table_bh_zf_total = data_collector.black_hole_data(nsnap=nsnap_c)
    table_bh_zf = table_bh_zf_total[table_bh_zf_total["bh_id"] == bh_id]

    dm_mini_spikes = damcat.DMMiniSpikesCalculator(
        sim_name = args.sim_name, 
        box_size = args.box_size, 
        dm_profile = args.dark_matter_profile, 
        core_index = args.core_index,
        table_bh = table_bh_zf
        )

    # only determine DM mini spikes if BH host halo at formation redshift is not spurious
    if dm_mini_spikes.spurios == False:
        # extract mini spike parameters
        # spike radius
        r_sp = dm_mini_spikes.r_sp

        # DM density at spike radius
        rho_at_r_sp = dm_mini_spikes.rho_at_r_sp
        rho_at_r_sp = (rho_at_r_sp * const.c ** 2).to(u.GeV / u.cm ** 3)

        # get spike index
        gamma_sp = dm_mini_spikes.spike_index

        # check if BH has a galaxy host at formation redshift
        no_host = dm_mini_spikes.no_host

        # add mini spike parameters to table
        table["bh_id"] = [bh_id]
        table["r_sp"] = [r_sp.to(u.pc).value]
        table["rho(r_sp)"] = [rho_at_r_sp.value]
        table["gamma_sp"] = [gamma_sp]
        table["no_host"] = [no_host]

        if args.dark_matter_profile == "cored":
            r_c = dm_mini_spikes.r_c
            table["r_c"] = [r_c.to(u.kpc).value]

        lst.append(table)

        if args.plot:
            path = f"plots/{args.sim_name}/galaxy_id_{int(galaxy_id)}/black_holes/spikes/id_{int(bh_id)}/"
            os.makedirs(path, exist_ok = True)
            # plot mini spike parameters
            if args.dark_matter_profile == "nfw":
                dm_mini_spikes.plot_nfw(path)
            elif args.dark_matter_profile == "cored":
                dm_mini_spikes.plot_cored(path)
            dm_mini_spikes.plot_radius_gravitational_influence(path)

if __name__ == "__main__":
    # initialise user input
    args = parse_args(include_cat = True, include_plot = True)

    # Set the number of processes to the number of available CPU cores or adjust as needed
    num_processes = mp.cpu_count()
    print("Number of CPU cores available:", num_processes)

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/imbh/"
    path_catalogue_temp = f"catalogue/{args.sim_name}/imbh_temp/"
    os.makedirs(path_catalogue, exist_ok = True)
    os.makedirs(path_catalogue_temp, exist_ok = True)

    # load temporary catalogue if requested
    if args.load_temporary_catalogue:
        bh_catalogue = pd.read_hdf(path_catalogue_temp + f"catalogue_temp_{args.name}.h5", key = "table")
        print("Temporary catalogue sucessfully loaded!")
    # else create new catalogue
    else:
        # extract galaxy data at z = 0
        data_collector = damcat.DataCollector(sim_name = args.sim_name, number_files = args.number_files)
        table_galaxy_z0_total = data_collector.galaxy_data(nsnap = 28)

        # unique galaxy root ids
        galaxy_id_unique = np.unique(table_galaxy_z0_total["galaxy_id"])
        # galaxy_id_unique = galaxy_id_unique[:1]

        # extract bh data at z = 0
        table_bh_z0_total = data_collector.black_hole_data(nsnap = 28)

        if args.plot:
            # save galaxy images
            galaxy_plotter = damplot.GalaxyPlotter(sim_name = args.sim_name, table_galaxy = table_galaxy_z0_total, table_bh = table_bh_z0_total)
            galaxy_plotter.save_gri_images()

            # plot galaxy properties distributions, such as 
            galaxy_plotter.plot_galaxy_distributions()

        # Create a multiprocessing manager list to share the pandas tables between processes
        coord_list = mp.Manager().list()

        print("Start extracting black hole coordinates for each MW-like galaxy...")
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

        # save current catalogue as temporary file so that it can be loaded by the same script if needed
        bh_catalogue.to_hdf(path_catalogue_temp + f"catalogue_temp_{args.name}.h5", key = "table")

    print(f"Number of unmerged black holes in MW-like galaxies in simulation {args.sim_name}: {len(bh_catalogue)}")
    print("Start extracting black hole dark matter mini spike parameter for each BH...")

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

    # drop the rows for which no mini spike parameters could be calculated due to spurious BH host halos
    bh_catalogue = bh_catalogue.dropna()

    # save BH catalogue
    bh_catalogue.to_hdf(path_catalogue + f"catalogue_{args.name}.h5", key = "table")
