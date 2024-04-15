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

with open("config/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def extract_flux_catalogue(bh_catalogue, args, path, m_dm):
    flux_calculator = damflux.FluxCalculator(bh_catalogue = bh_catalogue)
    flux_catalogue = pd.DataFrame()
    for sigma_v in args.sigma_v:
        table = bh_catalogue.copy()[["main_galaxy_id", "bh_id"]]

        r_cut = flux_calculator.radius_cut(m_dm, sigma_v)
        flux = flux_calculator.gamma_flux(m_dm, args.channel, args.E_th, sigma_v)

        table["sigma_v"] = sigma_v.value
        table["r_cut"] = r_cut.to(u.pc).value
        table["flux"] = flux.to(1 / (u.cm**2 * u.s)).value

        flux_catalogue = pd.concat([flux_catalogue, table], ignore_index = True)

    # check if any flux value is zero or close to zero
    if np.any(flux_catalogue["flux"].values <= 0):
        print("WARNING: Flux catalogue contains zero (or close to zero) or negative flux values!")
        print("Catalogue will not be saved!")
        print(flux_catalogue.loc[flux_catalogue["flux"].values <= 0])
    else:
        m_dm_string = format_energy(m_dm)
        flux_catalogue.to_hdf(path + f"m_dm_{m_dm_string}.h5", key = "table", mode = "w")


if __name__ == "__main__":
    # initialise user input
    args = parse_args(include_dm = True, include_plot = True)

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/imbh/"
    print(f"Load black hole catalogue from {path_catalogue + f'catalogue_{args.name}.h5'}")
    bh_catalogue = pd.read_hdf(path_catalogue + f"catalogue_{args.name}.h5", key = "table")

    E_th_string = format_energy(args.E_th)

    path_flux_catalogue = f"catalogue/{args.sim_name}/flux/{args.name}/{args.channel}_channel/e_th_{E_th_string}/"
    os.makedirs(path_flux_catalogue, exist_ok = True)

    print("Start calculating fluxes...")
    print("Number of black holes:", len(bh_catalogue))
    print("Number of dark matter masses:", len(args.m_dm))
    print("Number of cross sections:", len(args.sigma_v))

    num_processes = mp.cpu_count()
    print("Start multiprocessing...")
    print("Number of CPU cores available:", num_processes)
    # if both m_dm and sigma_v are arrays, use multiprocessing to speed up the calculation
    with mp.Pool(processes=num_processes) as pool:
        # add necesarry arguments to the function except of DM masses since this is the variable to loop over
        extract_flux_catalogue_with_args = partial(
            extract_flux_catalogue, 
            bh_catalogue,
            args,
            path_flux_catalogue
            )
        # Use tqdm to visualize the progress of the loop
        # loop over all individual DM masses to calculate fluxes and save them to a file
        for _ in tqdm(pool.imap_unordered(extract_flux_catalogue_with_args, args.m_dm), total=len(args.m_dm)):
            pass

    print("Finished calculating fluxes!")
    print(f"Flux catalogue(s) saved to: {path_flux_catalogue}")

    if args.plot:
        if len(args.sigma_v) != 1:
            raise ValueError("Plotting is currently only possible for a single cross section value!")
        if len(args.sigma_v) == 1:
            print("Start plotting...")
            path_plots = f"plots/{args.sim_name}/flux/{args.name}/{args.channel}_channel/e_th_{E_th_string}/"
            os.makedirs(path_plots, exist_ok = True)

            cmap = LinearSegmentedColormap.from_list("", config["Colors"]["cmap_flux"])
            indices = np.linspace(0, 1, len(args.m_dm))
            colors = cmap(indices)
            markers = config["Plots"]["markers"]
            marker_sizes = config["Plots"]["marker_sizes"]

            hess_flux_sensitivity = config["HESS"]["flux_sensitivity"] * u.Unit("cm-2 s-1")
            fermi_flux_sensitivity_l0_b0 = config["Fermi"]["flux_sensitivity_l0_b0"] * u.Unit("cm-2 s-1")
            fermi_flux_sensitivity_l120_b45 = config["Fermi"]["flux_sensitivity_l120_b45"] * u.Unit("cm-2 s-1")

            # open relevant flux catalogues
            flux_catalogues = []
            for m_dm in args.m_dm:
                m_dm_string = format_energy(m_dm)
                flux_catalogues.append(pd.read_hdf(path_flux_catalogue + f"m_dm_{m_dm_string}.h5", key = "table"))
            
            flux_catalogues_conat = pd.concat(flux_catalogues, ignore_index = True)
            flux_plotter = damplot.FluxPlotter(flux_catalogues_conat)
            flux = flux_catalogues_conat["flux"].values * u.Unit("cm-2 s-1")
            flux_min = np.min(flux)
            if flux_min < hess_flux_sensitivity:
                flux_th = flux_plotter.flux_thresholds(flux)
            else:
                flux_th = flux_plotter.flux_thresholds(flux, flux_min = hess_flux_sensitivity.value)

            plt.figure(figsize = config["Figure_size"]["single_column"])
            for flux_catalogue, m_dm, color, marker, marker_size in zip(flux_catalogues, args.m_dm, colors, markers, marker_sizes):
                flux_plotter = damplot.FluxPlotter(flux_catalogue)
                flux_plotter.plot_integrated_luminosity(flux_th, m_dm, color, marker, marker_size, args)

            if args.instrument_comparison == "hess":
                plt.xlabel(f"$\Phi (E > {int(np.rint(args.E_th.value))}$ GeV) [cm$^{{-2}}$ s$^{{-1}}$]")
            elif args.instrument_comparison == "fermi":
                plt.xlabel(f"$\Phi (E > {int(np.rint(args.E_th.to(u.MeV).value))}$ MeV) [cm$^{{-2}}$ s$^{{-1}}$]")
            plt.ylabel(r"$N_{{\mathrm{BH}}}(>\Phi)$")
            ymin, ymax = 1, 30 #TODO: set automatically
            if args.instrument_comparison == "hess":
                plt.vlines(hess_flux_sensitivity.value, ymin, ymax, color = "grey", linestyle = "dashed")
            elif args.instrument_comparison == "fermi":
                plt.vlines(fermi_flux_sensitivity_l0_b0.value, ymin, ymax, color = "grey", linestyle = "dashed")
                plt.vlines(fermi_flux_sensitivity_l120_b45.value, ymin, ymax, color = "grey", linestyle = "dashdot")
            plt.xscale("log")
            plt.yscale("log")
            plt.ylim(ymin = ymin, ymax = ymax)
            xmin, xmax = plt.xlim()
            # plt.hlines(2.3, xmin, xmax, color = "grey", linestyle = "dashed")
            # TODO: set xlims automatically
            if args.instrument_comparison == "hess":
                if args.channel == "b":
                    plt.xlim(xmin = xmin, xmax = 1e-6)
                    # plt.xlim(xmin = xmin, xmax = 1e-7)
                elif args.channel == "tau":
                    plt.xlim(xmin = xmin, xmax = 1e-6)
                else:
                    plt.xlim(xmin = xmin, xmax = 1e-6)
            elif args.instrument_comparison == "fermi":
                if args.channel == "tau":
                    plt.xlim(xmin = fermi_flux_sensitivity_l120_b45.value * 0.5, xmax = 1e-1)
                    # plt.xlim(xmin = 1e-11, xmax = 1e-3)
                else: 
                    plt.xlim(xmin = fermi_flux_sensitivity_l120_b45.value * 0.5, xmax = 1e-1)
            plt.legend(loc = "upper right", frameon = False, fontsize = 7)
            plt.tight_layout()
            plt.savefig(path_plots + "integrated_luminosity.pdf", dpi = 300)
            # plt.show()
            plt.close()

            # plot r_cut distribution for highest DM mass
            # choose lowest DM mass
            flux_plotter = damplot.FluxPlotter(flux_catalogues[0])
            flux_plotter.plot_cuttoff_radius_dist(path_plots + "r_cut_dist.pdf")

            print(f"Plots saved in {path_plots}")
        else:
            raise ValueError("Plotting is currently only possible for a single cross section value!")