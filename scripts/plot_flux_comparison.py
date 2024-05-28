import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the damspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import matplotlib as mpl
mpl.rc_file("config/mpl_config.rc")

import damspi.catalogue as damcat
import damspi.plot as damplot
import damspi.flux as damflux
from damspi.utils import parse_args, format_energy
import pandas as pd
import numpy as np
import astropy.units as u
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import yaml

if __name__ == "__main__":

    with open("config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args(include_dm = True, include_labels = True)
    exp_names = args.name
    channels = args.channel

    if type(channels) is list and type(exp_names) is list:
        raise ValueError("Only one channel can be plotted at a time if mutliple --name inputs are specified.")
    # exp_names = ["nfw", "cored_gamma_free", "cored_gamma_0p4", "cored_gamma_0p0"]
    # labels = ["NFW", r"Cored (free $\gamma_\mathrm{c}$)", r"Cored ($\gamma_\mathrm{c} = 0.4$)", r"Cored ($\gamma_\mathrm{c} = 0.0$)"]
    # labels = ["NFW", r"free $\gamma_\mathrm{c}$", r"$\gamma_\mathrm{c} = 0.4$", r"$\gamma_\mathrm{c} = 0.0$"]
    labels = args.labels
    m_dm_string = format_energy(args.m_dm[0])
    E_th_string = format_energy(args.E_th)

    print("Start plotting...")

    cmap = LinearSegmentedColormap.from_list("", config["Colors"]["cmap_flux"])
    markers = config["Plots"]["markers"]
    marker_sizes = config["Plots"]["marker_sizes"]

    # open relevant flux catalogues
    flux_catalogues = []
    # if multiple channels are specified, plot them all in one plot
    if type(channels) is list:
        path_plots = f"plots/{args.sim_name}/flux/comparison/"
        os.makedirs(path_plots, exist_ok = True)

        indices = np.linspace(0, 1, len(channels))
        colors = cmap(indices)

        for channel in channels:
            filename = f"catalogue/{args.sim_name}/flux/{exp_names}/{channel}_channel/e_th_{E_th_string}/m_dm_{m_dm_string}.h5"
            print("Loading flux catalogue from", filename)
            flux_catalogues.append(pd.read_hdf(filename, key = "table"))
    # if multiple exp_names are specified, plot them all in one plot
    else:
        path_plots = f"plots/{args.sim_name}/flux/comparison/{args.channel}_channel/e_th_{E_th_string}/"
        os.makedirs(path_plots, exist_ok = True)

        indices = np.linspace(0, 1, len(exp_names))
        colors = cmap(indices)

        for exp_name in exp_names:
            filename = f"catalogue/{args.sim_name}/flux/{exp_name}/{args.channel}_channel/e_th_{E_th_string}/m_dm_{m_dm_string}.h5"
            print("Loading flux catalogue from", filename)
            flux_catalogues.append(pd.read_hdf(filename, key = "table"))

    flux_catalogues_conat = pd.concat(flux_catalogues, ignore_index = True)

    sigma_v_catalogue = np.unique(flux_catalogues_conat["sigma_v"].values) * u.Unit("cm3 s-1")

    if len(sigma_v_catalogue) > 1:
        raise ValueError("Multiple sigma_v values in flux catalogue.")
    if sigma_v_catalogue != args.sigma_v:
        raise ValueError("sigma_v value in flux catalogue does not match input. Catalogue = " + str(sigma_v_catalogue) + ". Input = " + str(args.sigma_v))

    flux_plotter = damplot.FluxPlotter(flux_catalogues_conat)
    flux = flux_catalogues_conat["flux"].values * u.Unit("cm-2 s-1")

    hess_flux_sensitivity = config["HESS"]["flux_sensitivity"] * u.Unit("cm-2 s-1")

    flux_min = np.min(flux)
    if flux_min < hess_flux_sensitivity:
        flux_th = flux_plotter.flux_thresholds(flux)
    else:
        flux_th = flux_plotter.flux_thresholds(flux, flux_min = hess_flux_sensitivity.value)

    plt.figure(figsize = config["Figure_size"]["single_column_legend"])
    for flux_catalogue, label, color, marker, marker_size in zip(flux_catalogues, labels, colors, markers, marker_sizes):
        flux_plotter = damplot.FluxPlotter(flux_catalogue)
        flux_plotter.plot_integrated_luminosity_comparison(flux_th, label, color, marker, marker_size)
    plt.xlabel(f"$\Phi (E > {int(np.rint(args.E_th.value))}$ GeV) [cm$^{{-2}}$ s$^{{-1}}$]")
    plt.ylabel(r"$N_{{\mathrm{BH}}}(>\Phi)$")
    ymin, ymax = 1, 21 #TODO: set automatically
    plt.vlines(hess_flux_sensitivity.value, ymin, ymax, color = "grey", linestyle = "dashed")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(ymin = ymin, ymax = ymax) #TODO: set automatically
    xmin, xmax = plt.xlim()
    plt.xlim(xmin = xmin, xmax = 2e-8) #TODO: set automatically
    # plt.legend(loc = "upper right", frameon = False, fontsize = 7)
    plt.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="center", mode="expand", borderaxespad=0, ncol=2, alignment="center")
    plt.tight_layout()
    plt.savefig(path_plots + "integrated_luminosity_comparison.pdf", dpi = 300)
    # plt.show()
    plt.close()