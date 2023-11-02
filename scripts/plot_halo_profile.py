import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import matplotlib as mpl
mpl.rc_file("config/mpl_config.rc")

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from dammspi.utils import cored_profile
from scipy.odr import Model, RealData, ODR, Data
from dammspi.utils import nfw_profile, cored_profile
from astropy import constants as const
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import dammspi.catalogue as dammcat
import dammspi.plot as dammplot
from dammspi.utils import parse_args, cored_profile, nfw_profile
import pandas as pd
import yaml

if __name__ == "__main__":
    # open config file
    with open("config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load user input
    args = parse_args(include_cat = True, include_plot = True)

    # create directory for plots
    path_plots = f"plots/{args.sim_name}/dm_profiles/"
    os.makedirs(path_plots, exist_ok=True)

    # load temporary catalogue
    print("Loading temporary catalogue...")
    path_catalogue_temp = f"catalogue_temp/{args.sim_name}/"
    bh_catalogue = pd.read_csv(path_catalogue_temp + f"catalogue_temp_{args.name}.csv")

    # select first row of catalogue (as an example)
    row = bh_catalogue.iloc[0]

    # select values from row
    bh_id = row["bh_id"]
    galaxy_id = row["galaxy_id"]
    nsnap_c = row["nsnap_c"]

    # load data
    print("Loading BH data at formation redshift...")
    data_collector = dammcat.DataCollector(sim_name = args.sim_name, number_files = args.number_files)
    table_bh_zf_total = data_collector.black_hole_data(nsnap=nsnap_c)
    table_bh_zf = table_bh_zf_total[table_bh_zf_total["bh_id"] == bh_id]

    # determine dark matter halo profile assuming NFW profile
    print("Performing NFW fit...")
    dm_mini_spikes_nfw = dammcat.DMMiniSpikesCalculator(
        sim_name = args.sim_name, 
        box_size = args.box_size, 
        dm_profile = "nfw", 
        core_index = None,
        table_bh = table_bh_zf
        )

    # get raw EAGLE data
    data_r = dm_mini_spikes_nfw.ap_size * u.kpc
    data_rho_log = dm_mini_spikes_nfw.dm_rho_log
    data_rho = np.exp(data_rho_log) * u.Msun / u.kpc ** 3
    # convert dark matter density to units of GeV / cm^3
    data_rho = (data_rho * const.c ** 2).to(u.GeV / u.cm ** 3)

    # get best fit parameters
    rho_0_nfw = dm_mini_spikes_nfw.rho_0
    r_s_nfw = dm_mini_spikes_nfw.r_s

    # determine dark matter halo profile assuming cored profile
    print("Performing cored fit...")
    dm_mini_spikes_cored = dammcat.DMMiniSpikesCalculator(
        sim_name = args.sim_name, 
        box_size = args.box_size, 
        dm_profile = "cored", 
        core_index = None,
        table_bh = table_bh_zf
        )

    # get best fit parameters
    rho_0_cored = dm_mini_spikes_cored.rho_0
    r_s_cored = dm_mini_spikes_cored.r_s
    r_c_cored = dm_mini_spikes_cored.r_c
    gamma_c_cored = dm_mini_spikes_cored.gamma_c

    # plot data and best fit
    print("Plotting results...")
    galaxy_plotter = dammplot.GalaxyPlotter(sim_name = args.sim_name, table_galaxy = None, table_bh = None)
    galaxy_plotter.plot_dark_matter_profile(
        data = (data_r, data_rho),
        para_nfw = (rho_0_nfw, r_s_nfw),
        para_cored = (rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored),
        path = path_plots
        )
    print(f"Plots saved in {path_plots}")
