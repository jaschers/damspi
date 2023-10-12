import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import dammspi.plot as dammplot
from dammspi.utils import parse_args
import pandas as pd

if __name__ == '__main__':
    # get user input
    args = parse_args()

    # open the catalogue
    filename = f"catalogue/{args.sim_name}/catalogue.csv"
    bh_catalogue = pd.read_csv(filename)

    # get the latitude and longitude of the black holes
    d_gc = bh_catalogue['d_GC [kpc]']
    lat_gc = bh_catalogue['lat_GC [rad]']
    long_gc = bh_catalogue['long_GC [rad]']
    d_sun = bh_catalogue['d_Sun [kpc]']
    lat_sun = bh_catalogue['lat_Sun [rad]']
    long_sun = bh_catalogue['long_Sun [rad]']

    bh_plotter = dammplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = bh_catalogue)

    path = f"plots/{args.sim_name}/black_hole_dist/"
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

    # bh_plotter.plot_2d_map_contours(lat_gc, long_gc, path + "2d_map_gc_contours.pdf")
    bh_plotter.plot_2d_map_contours(lat_sun, long_sun, path + "2d_map_sun_contours.pdf")

    print(bh_catalogue)