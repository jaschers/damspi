import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import dammspi.catalogue as dammcat
import dammspi.plot as dammplot
import dammspi.flux as dammflux
from dammspi.utils import parse_args
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

plt.rcParams.update({'font.size': 8}) # 8 (paper), 10 (poster)
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')#, weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'
cm_conversion_factor = 1/2.54  # centimeters in inches
single_column_fig_size = (7.0 * cm_conversion_factor, 7.0 * 3/4 * cm_conversion_factor)


def extract_flux_catalogue(bh_catalogue, flux_calculator, args, m_dm):
    flux_catalogue = pd.DataFrame()
    for sigma_v in args.sigma_v:
        table = bh_catalogue.copy()[["galaxy_id", "bh_id"]]

        r_cut = flux_calculator.radius_cut(m_dm, sigma_v)
        flux = flux_calculator.gamma_flux(m_dm, args.channel, args.E_th, sigma_v)

        table["sigma_v [cm3 s-1]"] = sigma_v.value
        table["r_cut [pc]"] = r_cut.to(u.pc).value
        table["flux [cm-2 s-1]"] = flux.to(1 / (u.cm**2 * u.s)).value
        
        flux_catalogue = pd.concat([flux_catalogue, table], ignore_index = True)
    flux_catalogue.to_hdf(path + f"m_dm_{int(np.rint(m_dm.value))}GeV.h5", key = "table", mode = "w")


if __name__ == "__main__":
    # initialise user input
    args = parse_args()

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/"
    print(f"Load black hole catalogue from {path_catalogue + f'{args.filename}.csv'}")
    bh_catalogue = pd.read_csv(path_catalogue + f"{args.filename}.csv")

    bh_catalogue["gamma_sp"] = 7 / 3

    flux_calculator = dammflux.FluxCalculator(bh_catalogue = bh_catalogue, dm_profile = args.dark_matter_profile)

    path = f"catalogue/{args.sim_name}/flux/{args.dark_matter_profile}/{args.channel}_channel/"
    os.makedirs(path, exist_ok = True)

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
            flux_calculator, 
            args
            )
        # Use tqdm to visualize the progress of the loop
        # loop over all individual DM masses to calculate fluxes and save them to a file
        for _ in tqdm(pool.imap_unordered(extract_flux_catalogue_with_args, args.m_dm), total=len(args.m_dm)):
            pass

    print("Finished calculating fluxes!")
    print(f"Flux catalogue(s) saved to: {path}")

    if args.plot:
        if len(args.sigma_v) == 1:
            print("Start plotting...")
            path_plots = f"plots/{args.sim_name}/flux/{args.dark_matter_profile}/{args.channel}_channel/"
            os.makedirs(path_plots, exist_ok = True)

            cmap = LinearSegmentedColormap.from_list("", ["#fca311", "#e83151", "#003049"])
            indices = np.linspace(0, 1, len(args.m_dm))
            colors = cmap(indices)

            # open relevant flux catalogues
            flux_catalogues = []
            for m_dm in args.m_dm:
                flux_catalogues.append(pd.read_hdf(path + f"m_dm_{int(np.rint(m_dm.value))}GeV.h5", key = "table"))
            
            flux_catalogues_conat = pd.concat(flux_catalogues, ignore_index = True)
            flux_plotter = dammplot.FluxPlotter(flux_catalogues_conat)
            flux = flux_catalogues_conat["flux [cm-2 s-1]"].values * u.Unit("cm-2 s-1")
            flux_th = flux_plotter.flux_thresholds(flux)

            plt.figure(figsize = single_column_fig_size)
            for flux_catalogue, m_dm, color in zip(flux_catalogues, args.m_dm, colors):
                flux_plotter = dammplot.FluxPlotter(flux_catalogue)
                flux_plotter.plot_integrated_luminosity(flux_th, m_dm, color)
            plt.xlabel(f"$\Phi (E_\mathrm{{th}} > {int(np.rint(args.E_th.value))}$ GeV) [cm$^{{-2}}$ s$^{{-1}}$]")
            plt.ylabel(r"$N_{{\mathrm{BH}}}(>\Phi)$")
            ymin, ymax = 1, 30 #TODO: set automatically
            plt.vlines(config["HESS"]["flux_sensitivity"], ymin, ymax, color = "grey", linestyle = "dashed")
            plt.xscale("log")
            plt.yscale("log")
            plt.ylim(ymin = ymin, ymax = ymax) #TODO: set automatically
            xmin, xmax = plt.xlim()
            # plt.hlines(2.3, xmin, 1e-7, color = "grey", linestyle = "dashdot")
            plt.xlim(xmin = xmin, xmax = 1e-7) #TODO: set automatically
            plt.legend(loc = "upper right", frameon = False, fontsize = 7)
            plt.tight_layout()
            plt.savefig(path_plots + "integrated_luminosity.pdf", dpi = 300)
            plt.close()

            # plot r_cut distribution for highest DM mass
            flux_plotter.plot_cuttoff_radius_dist(path_plots + "r_cut_dist.pdf")

            print("Finished plotting!")
            print(f"Plots saved to: {path}plots/")
        else:
            raise ValueError("Plotting is currently only possible for a single cross section value!")