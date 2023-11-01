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

    # # plot data and best fit
    # fig = plt.figure(figsize=config["Figure_size"]["single_column_legend"])
    # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3 units for main plot, 1 unit for residual

    # ax = plt.subplot(gs[0])  # main plot

    # # Original plot
    # line_data, = ax.plot(data_r, data_rho, label="data", marker="o", linestyle="None", color=config["Colors"]["black"], markersize=3)
    # line_cored, = ax.plot(r, rho_cored, label="Cored", color=config["Colors"]["yellow"])
    # line_nfw, = ax.plot(r, rho_nfw, label="NFW", color=config["Colors"]["red"])
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_ylabel(r"$\rho(r)$ [GeV cm$^{-3}$]")
    # ymin, ymax = ax.get_ylim()
    # line_r_c = ax.vlines(r_c_cored.value, ymin, ymax, label=r"$r_\mathrm{c}$", color=config["Colors"]["black"], linestyle="dashed")
    # ax.set_ylim(ymin, ymax)
    # lines = [line_data, line_nfw, line_r_c, line_cored]
    # labels = [l.get_label() for l in lines]
    # # ax.legend(lines, labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="center", mode="expand", borderaxespad=0, ncol=2, alignment="center")
    # ax.legend(lines, labels, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="center", mode="expand", borderaxespad=0, ncol=2, alignment="center")
    # ax.xaxis.set_visible(False)

    # # Inset plot
    # r_min, r_max = 0.9 * u.kpc, 1.5 * u.kpc #kpc
    # ymax = cored_profile((rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored), r_min) * 1.1
    # ymin = cored_profile((rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored), r_max)
    # ymax = (ymax * const.c ** 2).to(u.GeV / u.cm ** 3)
    # ymin = (ymin * const.c ** 2).to(u.GeV / u.cm ** 3)

    # axins = inset_axes(ax, width='30%', height='30%', loc='upper right')
    # axins.plot(data_r, data_rho, marker="o", linestyle="None", color=config["Colors"]["black"], markersize=3)
    # axins.plot(r, rho_cored, color=config["Colors"]["yellow"])
    # axins.plot(r, rho_nfw, color=config["Colors"]["red"])
    # axins.vlines(r_c_cored.value, ymin.value, ymax.value, label=r"$r_\mathrm{c}$", color=config["Colors"]["black"], linestyle="dashed")
    # axins.set_xlim(r_min.value, r_max.value)  
    # axins.set_ylim(ymin.value, ymax.value)  
    # axins.set_xscale('log')
    # axins.set_yscale('log')
    # axins.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.4]))
    # axins.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # axins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # axins.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1., 1.3]))
    # axins.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # axins.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # #axins.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    # # Residual subplot
    # ax_res = plt.subplot(gs[1], sharex=ax)
    # ax_res.set_ylabel(r'$R$')
    # ax_res.set_xscale('log')
    # residual_nfw = (data_rho - rho_nfw_data) / rho_nfw_data 
    # residual_cored = (data_rho - rho_cored_data) / rho_cored_data 
    # ax_res.plot(data_r, residual_cored, color=config["Colors"]["yellow"], marker = "o", markersize = 3, linestyle = "None")
    # ax_res.plot(data_r, residual_nfw, color=config["Colors"]["red"], marker = "o", markersize = 3, linestyle = "None")
    # ax_res.plot(data_r, np.zeros(len(data_r)), color = config["Colors"]["black"], linestyle = "solid")
    # ax_res.set_xlabel(r"$r$ [kpc]")
    # ymin, ymax = ax_res.get_ylim()
    # ax_res.set_ylim(ymin * 1.2, ymax * 1.2)
    # plt.tight_layout()
    # plt.savefig(path_plots + "dm_profile_fit.pdf", dpi = 300)
    # # plt.show()