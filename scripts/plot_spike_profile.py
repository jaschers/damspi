import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import matplotlib as mpl
mpl.rc_file("config/mpl_config.rc")

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const 
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import os
from dammspi.utils import spike_profile, imbh_profile, parse_args
import dammspi.plot as dammplot
import yaml

if __name__ == "__main__":
    # open config file
    with open("config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get arguments from command line
    args = parse_args(include_name = False)

    path_plots = f"plots/{args.sim_name}/dm_profiles/"
    os.makedirs(path_plots, exist_ok=True)

    # Get example parameters from config file
    rho_0 = config["Spike_profile_example"]["rho_0"] * u.GeV / u.cm**3  
    rho_0 = (rho_0 / const.c ** 2).to(u.M_sun / u.kpc ** 3)
    r_s = config["Spike_profile_example"]["r_s"] * u.kpc    
    r_sp = float(config["Spike_profile_example"]["r_sp"]) * u.kpc   
    gamma_sp = config["Spike_profile_example"]["gamma_sp"]  
    r_cut = float(config["Spike_profile_example"]["r_cut"]) * u.kpc  
    M_bh = float(config["Spike_profile_example"]["M_bh"]) * u.M_sun
    r_schw = ((2 * const.G * M_bh) / const.c**2).to(u.kpc) 

    # Generate radius values
    radii = np.logspace(-13, 2, 1000) * u.kpc

    # Evaluate density at each radius value
    rho_total = np.array([imbh_profile((rho_0, r_schw, r_cut, r_s, r_sp, gamma_sp), r) for r in radii]) 

    # plot data and best fit
    print("Plotting spike profile...")
    bh_plotter = dammplot.BlackHolePlotter(sim_name = None, table_bh = None)
    bh_plotter.plot_spike_profile(radii, rho_total, r_schw, r_cut, r_sp, path_plots)
    print(f"Plots saved in {path_plots}")
