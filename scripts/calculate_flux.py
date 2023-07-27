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

def fill_table(table, flux_calculator, m_dm, sigma_v, args):
    table["sigma_v [cm3 s-1]"] = sigma_v.value
    r_cut = flux_calculator.radius_cut(m_dm, sigma_v)
    flux = flux_calculator.gamma_flux(m_dm, args.channel, args.E_th, sigma_v)
    table["r_cut [pc]"] = r_cut.to(u.pc).value
    table["flux [cm-2 s-1]"] = flux.to(1 / (u.cm**2 * u.s)).value
    return(table)

def extract_flux(bh_catalogue, flux_calculator, args, m_dm):
    flux_catalogue = pd.DataFrame()
    for sigma_v in args.sigma_v:
        table = bh_catalogue.copy()[["galaxy_id", "bh_id"]]
        table = fill_table(table, flux_calculator, m_dm, sigma_v, args)
        flux_catalogue = pd.concat([flux_catalogue, table], ignore_index = True)
    flux_catalogue.to_hdf(path + f"m_dm_{int(np.rint(m_dm.value))}GeV.h5", key = "table", mode = "w")


if __name__ == "__main__":
    # initialise user input
    args = parse_args()

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/"
    print(f"Load black hole catalogue from {path_catalogue + 'catalogue.csv'}")
    bh_catalogue = pd.read_csv(path_catalogue + "catalogue.csv")
    flux_calculator = dammflux.FluxCalculator(bh_catalogue)

    path = f"catalogue/{args.sim_name}/flux/{args.channel}_channel/"
    os.makedirs(path, exist_ok = True)

    print("Start calculating fluxes...")
    print("Number of black holes:", len(bh_catalogue))
    print("Number of dark matter masses:", len(args.m_dm))
    print("Number of cross sections:", len(args.sigma_v))
    # create flux_catalogue based on DM mass and cross section input
    # if m_dm is an array and sigma_v is a single value, loop over m_dm and save flux_catalogue to a file for each m_dm
    if len(args.m_dm) > 1 and len(args.sigma_v) == 1:
        for m_dm in tqdm(args.m_dm):
            flux_catalogue = bh_catalogue.copy()[["galaxy_id", "bh_id"]]
            flux_catalogue = fill_table(flux_catalogue, flux_calculator, m_dm, args.sigma_v[0], args)
            flux_catalogue.to_hdf(path + f"m_dm_{int(np.rint(m_dm.value))}GeV.h5", key = "table", mode = "w")

    # if sigma_v is an array and m_dm is a single value, loop over sigma_v and save flux_catalogue to a single file
    elif len(args.m_dm) == 1 and len(args.sigma_v) > 1:
        flux_catalogue = pd.DataFrame()
        for sigma_v in tqdm(args.sigma_v):
            table = bh_catalogue.copy()[["galaxy_id", "bh_id"]]
            table = fill_table(table, flux_calculator, args.m_dm[0], sigma_v, args)
            flux_catalogue = pd.concat([flux_catalogue, table], ignore_index = True)
        flux_catalogue.to_hdf(path + f"m_dm_{int(np.rint(args.m_dm[0].value))}GeV.h5", key = "table", mode = "w")

    # if both m_dm and sigma_v are arrays, save flux_catalogue to a file for each m_dm
    elif len(args.m_dm) > 1 and len(args.sigma_v) > 1:
        # Set the number of processes to the number of available CPU cores or adjust as needed
        num_processes = mp.cpu_count()
        print("Start multiprocessing...")
        print("Number of CPU cores available:", num_processes)
        # if both m_dm and sigma_v are arrays, use multiprocessing to speed up the calculation
        with mp.Pool(processes=num_processes) as pool:
            # add necesarry arguments to the function except of DM masses since this is the variable to loop over
            extract_flux_with_args = partial(
                extract_flux, 
                bh_catalogue,
                flux_calculator, 
                args
                )
            # Use tqdm to visualize the progress of the loop
            # loop over all individual DM masses to calculate fluxes and save them to a file
            for _ in tqdm(pool.imap_unordered(extract_flux_with_args, args.m_dm), total=len(args.m_dm)):
                pass

    # if both m_dm and sigma_v are single values, save flux_catalogue to a single file
    else:
        flux_catalogue = bh_catalogue.copy()[["galaxy_id", "bh_id"]]
        table = fill_table(flux_catalogue, flux_calculator, args.m_dm[0], args.sigma_v[0], args)
        flux_catalogue.to_hdf(path + f"m_dm_{int(np.rint(args.m_dm[0].value))}GeV.h5", key = "table", mode = "w")

    print("Finished calculating fluxes!")
    print(f"Flux catalogue(s) saved to: {path}")

    if args.plot:
        if len(args.sigma_v) == 1:
            print("Start plotting...")
            path_plots = f"plots/{args.sim_name}/flux/{args.channel}_channel/"
            os.makedirs(path_plots, exist_ok = True)

            if len(args.m_dm) > 1:
                pass
            else:
                # get fluxes for different DM masses
                flux_plotter = dammplot.FluxPlotter(flux_catalogue)
                flux_plotter.plot_integrated_luminosity(args.m_dm, args.sigma_v, args.E_th, path_plots + "integrated_luminosity.pdf")

            print("Finished plotting!")
            print(f"Plots saved to: {path}plots/")
        else:
            raise ValueError("Plotting is currently only possible for a single cross section value!")