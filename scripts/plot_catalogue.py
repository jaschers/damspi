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

    # open the catalogue
    filename = f"catalogue/{args.sim_name}/imbh/catalogue_{args.name}.csv"
    bh_catalogue = pd.read_csv(filename)

    # get the latitude and longitude of the black holes
    d_gc = bh_catalogue['d_GC [kpc]']
    lat_gc = bh_catalogue['lat_GC [rad]']
    long_gc = bh_catalogue['long_GC [rad]']
    d_sun = bh_catalogue['d_Sun [kpc]']
    lat_sun = bh_catalogue['lat_Sun [rad]']
    long_sun = bh_catalogue['long_Sun [rad]']
    gamma_sp = bh_catalogue['gamma_sp']
    gamma_c = gamma_core(gamma_sp)
    bh_catalogue['gamma_c'] = gamma_c

    bh_plotter = damplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = bh_catalogue)

    # upsample the catalogue for 2D maps
    # upsample the catalogue
    d_sun_upsampled, lat_sun_upsampled, long_sun_upsampled = bh_plotter.random_upsampling(
        d_gc = d_gc, 
        lat_gc = lat_gc, 
        long_gc = long_gc,
        upsampling_factor = args.upsampling_factor)

    path = f"plots/{args.sim_name}/black_hole_dist/{args.name}/"
    os.makedirs(path, exist_ok = True)

    # # Plotting BH number distributions
    print("Plotting BH number distributions...")
    bh_plotter.plot_number_dist(path)

    # Plotting BH distributions
    print("Plotting BH distributions...")
    # bh_plotter.plot_dist_total(path)
    bh_plotter.plot_dist_total_mean(path)

    # bh_plotter.plot_cumulative_radial_distribution(d_gc, path + "cumulative_radial_distribution.pdf")
    bh_plotter.plot_cumulative_radial_distribution_mean(path + "cumulative_radial_distribution_mean.pdf")

    # Plotting BH 2D maps
    print("Plotting BH 2D maps...")
    bh_plotter.plot_2d_map(lat_gc, long_gc, path + "2d_map_gc.pdf")
    bh_plotter.plot_2d_map(lat_sun, long_sun, path + "2d_map_sun.pdf")

    bh_plotter.plot_2d_map_contours(lat_gc, long_gc, path + "2d_map_gc_contours.pdf")
    bh_plotter.plot_2d_map_contours(lat_sun_upsampled, long_sun_upsampled, args.upsampling_factor, path + "2d_map_sun_contours.pdf")

    bh_plotter.plot_healpix_map(lat_sun_upsampled, long_sun_upsampled, path + "2d_healpix_map_sun.pdf")

    # Plot a 2D gaussian pdf of the BH distribution
    bh_plotter.plot_2d_map_gaussian(lat_sun_upsampled, long_sun_upsampled, args.upsampling_factor, path + "2d_map_sun_gaussian.pdf")

    print(f"Plots saved in {path}")
