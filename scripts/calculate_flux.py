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
import multiprocessing as mp
import numpy as np
import astropy.units as u
from tqdm import tqdm

def calculate_flux():


if __name__ == "__main__":
    # initialise user input
    args = parse_args()

    # Set the number of processes to the number of available CPU cores or adjust as needed
    num_processes = mp.cpu_count()
    print("Number of CPU cores available:", num_processes)

    # define directory for catalogue
    path_catalogue = f"catalogue/{args.sim_name}/"

    bh_catalogue = pd.read_csv(path_catalogue + "catalogue.csv")

    print(args.m_dm, args.sigma_v)

    flux_calculator = dammflux.FluxCalculator(bh_catalogue)

    path = f"catalogue/{args.sim_name}/flux/{args.channel}_channel/"
    os.makedirs(path, exist_ok = True)

    if isinstance(args.m_dm.value, np.ndarray) == True and isinstance(args.sigma_v.value, np.ndarray) == False:
        for m_dm in tqdm(args.m_dm):
            flux_catalogue = bh_catalogue.copy()[["bh_id"]]

            flux_catalogue["sigma_v [cm3 s-1]"] = args.sigma_v.value

            r_cut = flux_calculator.radius_cut(m_dm, args.sigma_v)

            flux = flux_calculator.imbh_flux(m_dm, args.channel, args.E_th, args.sigma_v)

            flux_catalogue["r_cut [pc]"] = r_cut.to(u.pc).value
            flux_catalogue["flux [cm-2 s-1]"] = flux.to(1 / (u.cm**2 * u.s)).value
            
            print(flux_catalogue)

            flux_catalogue.to_hdf(path + f"m_{int(np.rint(m_dm.value))}GeV.h5", key = "table", mode = "w")

    elif isinstance(args.m_dm.value, np.ndarray) == False and isinstance(args.sigma_v.value, np.ndarray) == True:
        flux_catalogue = pd.DataFrame()
        for sigma_v in tqdm(args.sigma_v):
            table = bh_catalogue.copy()[["bh_id"]]

            table["sigma_v [cm3 s-1]"] = sigma_v.value

            r_cut = flux_calculator.radius_cut(args.m_dm, sigma_v)

            flux = flux_calculator.imbh_flux(args.m_dm, args.channel, args.E_th, sigma_v)

            table["r_cut [pc]"] = r_cut.to(u.pc).value
            table["flux [cm-2 s-1]"] = flux.to(1 / (u.cm**2 * u.s)).value

            flux_catalogue = pd.concat([flux_catalogue, table], ignore_index = True)
            
        print(flux_catalogue)

        flux_catalogue.to_hdf(path + f"m_{int(np.rint(args.m_dm.value))}GeV.h5", key = "table", mode = "w")

    elif isinstance(args.m_dm.value, np.ndarray) == True and isinstance(args.sigma_v.value, np.ndarray) == True:
        for m_dm in tqdm(args.m_dm):
            flux_catalogue = pd.DataFrame()
            for sigma_v in tqdm(args.sigma_v):
                table = bh_catalogue.copy()[["bh_id"]]

                table["sigma_v [cm3 s-1]"] = sigma_v.value

                r_cut = flux_calculator.radius_cut(m_dm, sigma_v)

                flux = flux_calculator.imbh_flux(m_dm, args.channel, args.E_th, sigma_v)

                table["r_cut [pc]"] = r_cut.to(u.pc).value
                table["flux [cm-2 s-1]"] = flux.to(1 / (u.cm**2 * u.s)).value

                flux_catalogue = pd.concat([flux_catalogue, table], ignore_index = True)
            
            print(flux_catalogue)

            flux_catalogue.to_hdf(path + f"m_{int(np.rint(m_dm.value))}GeV.h5", key = "table", mode = "w")
    else:
        flux_catalogue = bh_catalogue.copy()[["bh_id"]]

        flux_catalogue["sigma_v [cm3 s-1]"] = args.sigma_v.value

        r_cut = flux_calculator.radius_cut(args.m_dm, args.sigma_v)

        flux = flux_calculator.imbh_flux(args.m_dm, args.channel, args.E_th, args.sigma_v)

        flux_catalogue["r_cut [pc]"] = r_cut.to(u.pc).value
        flux_catalogue["flux [cm-2 s-1]"] = flux.to(1 / (u.cm**2 * u.s)).value

        flux_catalogue.to_hdf(path + f"m_{int(np.rint(args.m_dm.value))}GeV.h5", key = "table", mode = "w")

        print(flux_catalogue)
