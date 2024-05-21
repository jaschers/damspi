import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the damspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import damspi.plot as damplot
from damspi.utils import parse_args, gamma_core
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # get user input
    args = parse_args(include_upsampling = True)

    # open the catalogues
    filename_bh = f"catalogue/{args.sim_name}/imbh/catalogue_{args.name}.h5"
    filename_galaxy = f"catalogue/{args.sim_name}/galaxy/mw_galaxies_catalogue_{args.name}.h5"
    path_kde = f"catalogue/{args.sim_name}/imbh/"
    bh_catalogue = pd.read_hdf(filename_bh, key = "table")
    bh_catalogue_main_galaxy = bh_catalogue[bh_catalogue['satellite'] == False].reset_index(drop = True)
    galaxy_catalogue = pd.read_hdf(filename_galaxy, key = "table")

    
    print("bh_catalogue:")
    print(bh_catalogue)
    print(bh_catalogue.columns)

    print("bh_catalogue_main_galaxy:")
    print(bh_catalogue_main_galaxy)
    print(bh_catalogue_main_galaxy.columns)

    print("galaxy_catalogue:")
    print(galaxy_catalogue)
    print(galaxy_catalogue.columns)

    print("Total number of BHs:", len(bh_catalogue))
    print("Number of BHs within the main galaxy:", len(bh_catalogue_main_galaxy))
    print("Number of BHs in satellites:", len(bh_catalogue) - len(bh_catalogue_main_galaxy))

    print("Number of main galaxies:", len(np.unique(galaxy_catalogue[galaxy_catalogue["subgroup_number"] == 0]['galaxy_id'])))
    print("Number of satellite galaxies:", len(np.unique(galaxy_catalogue[galaxy_catalogue["subgroup_number"] != 0]['galaxy_id'])))

    # get the latitude and longitude of the black holes
    d_gc = bh_catalogue['d_GC']
    lat_gc = bh_catalogue['lat_GC']
    long_gc = bh_catalogue['long_GC']
    d_sun = bh_catalogue['d_Sun']
    lat_sun = bh_catalogue['lat_Sun']
    long_sun = bh_catalogue['long_Sun']
    gamma_sp = bh_catalogue['gamma_sp']
    gamma_c = gamma_core(gamma_sp)
    bh_catalogue['gamma_c'] = gamma_c

    d_gc_main_galaxy = bh_catalogue_main_galaxy['d_GC']
    lat_gc_main_galaxy = bh_catalogue_main_galaxy['lat_GC']
    long_gc_main_galaxy = bh_catalogue_main_galaxy['long_GC']

    bh_plotter = damplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = bh_catalogue)
    bh_plotter_main_galaxy = damplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = bh_catalogue_main_galaxy)
    galaxy_plotter = damplot.GalaxyPlotter(sim_name = args.sim_name, table_galaxy = galaxy_catalogue, table_bh = bh_catalogue)

    # upsample the catalogue for 2D maps
    # upsample the catalogue, input: coordinates with origin in the GC, output: coordinates with origin in the Sun
    d_sun_upsampled, lat_sun_upsampled, long_sun_upsampled = bh_plotter.random_upsampling(
        d_gc = d_gc, 
        lat_gc = lat_gc, 
        long_gc = long_gc,
        upsampling_factor = args.upsampling_factor
        )
    
    d_sun_main_galaxy_upsampled, lat_sun_main_galaxy_upsampled, long_sun_main_galaxy_upsampled = bh_plotter.random_upsampling(
        d_gc = d_gc_main_galaxy,
        lat_gc = lat_gc_main_galaxy,
        long_gc = long_gc_main_galaxy,
        upsampling_factor = args.upsampling_factor
        )

    path_bh = f"plots/{args.sim_name}/black_hole_dist/{args.name}/"
    path_galaxy = f"plots/{args.sim_name}/galaxy_dist/{args.name}/"
    os.makedirs(path_bh, exist_ok = True)
    os.makedirs(path_galaxy, exist_ok = True)

    # plot the mass distribution of main galaxies and satellites
    galaxy_plotter.plot_galaxy_properties_dist(path_galaxy)

    # plot disc and bulge proprties of main galaxies
    galaxy_plotter.plot_morphology_dist(path_galaxy)

    # TODO: Add it to bh_plotter for consistency (not urgent)
    galaxy_plotter.plot_scatter_bh_galaxy_morphology(path_bh)

    # plot number distribution of satellites in main galaxies
    galaxy_plotter.plot_satellite_number_dist(path_galaxy)

    # plot histogram of satellites with no gas, no stars and no stars and gas
    galaxy_plotter.plot_satellite_types(path_galaxy)

    galaxy_plotter.plot_n_bh_satellite_types(path_bh)

    galaxy_plotter.plot_likelihood_bh_satellite_star_mass(path_bh)

    galaxy_plotter.plot_likelihood_bh_satellite_total_mass(path_bh)

    galaxy_plotter.plot_scatter_n_satellites_m200(path_galaxy)
    
    # # Plotting BH number distributions
    print("Plotting BH number distributions...")
    bh_plotter.plot_number_dist(path_bh + "number_dist.pdf", table_galaxy = galaxy_catalogue)
    # bh_plotter_main_galaxy.plot_number_dist(path_bh + "number_dist_main_galaxy.pdf")

    bh_plotter.plot_number_dist_satellites(path_bh)

    # plot two bar histogram of BHs in main galaxies and satellites
    bh_plotter.plot_n_bh_in_satellites(path_bh)

    # Plotting BH distributions
    print("Plotting BH distributions...")
    # bh_plotter.plot_dist_total(path_bh)
    bh_plotter.plot_dist_total_mean(path_bh)

    # bh_plotter.plot_cumulative_radial_distribution(d_gc, path_bh + "cumulative_radial_distribution.pdf")
    bh_plotter.plot_cumulative_radial_distribution_mean(path_bh + "cumulative_radial_distribution_mean.pdf")

    # bh_plotter_main_galaxy.plot_cumulative_radial_distribution_mean(path_bh + "cumulative_radial_distribution_mean_main_galaxy.pdf")

    # Plotting BH 2D maps
    print("Plotting BH 2D maps...")
    # bh_plotter.plot_2d_map(lat_gc, long_gc, path_bh + "2d_map_gc.pdf")
    bh_plotter.plot_2d_map(lat_sun, long_sun, path_bh + "2d_map_sun.pdf")

    # bh_plotter.plot_2d_map_contours(lat_gc, long_gc, args.upsampling_factor, path_bh + "2d_map_gc_contours.pdf", path_kde)
    bh_plotter.plot_2d_map_contours(
        lat = lat_sun_upsampled, 
        long = long_sun_upsampled, 
        upsampling_factor = args.upsampling_factor, 
        path = path_bh + f"2d_map_sun_contours_uf{args.upsampling_factor}.pdf", 
        path_kde = path_kde,
        wihtin_region = True
        )
    
    # bh_plotter.plot_2d_map_contours(
    #     lat = lat_sun_main_galaxy_upsampled, 
    #     long = long_sun_main_galaxy_upsampled, 
    #     upsampling_factor = args.upsampling_factor, 
    #     path = path_bh + "2d_map_sun_main_galaxy_contours.pdf", 
    #     path_kde = path_kde
    #     )

    # plot scatter plot between number of BHs vs number satellites in main galaxies
    bh_plotter.plot_scatter_bh_n_satellites(path_bh , table_galaxy = galaxy_catalogue)

    bh_plotter.plot_scatter_bh_galaxy_properties(path_bh , table_galaxy = galaxy_catalogue)

    # plot bar histogram of BHs for satellites types
    bh_plotter.plot_bh_in_satellite_types(path_bh)

    # Plot a 2D gaussian pdf of the BH distribution
    # bh_plotter.plot_2d_map_gaussian(lat_gc_upsampled, long_gc_upsampled, args.upsampling_factor, path_bh + "2d_map_sun_gaussian.pdf")

    print(f"BH plots saved in {path_bh}")
    print(f"Galaxy plots saved in {path_galaxy}")
