import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the damspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import matplotlib as mpl
mpl.rc_file("config/mpl_config.rc")
import matplotlib.pyplot as plt
import astropy.units as u
from gammapy.astro.darkmatter import PrimaryFlux
import yaml
from damspi.utils import parse_args
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

if __name__ == "__main__":

    with open("config/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = parse_args(include_name = False, include_dm = True, include_labels = True)

    path_plots = f"plots/{args.sim_name}/flux/comparison/"
    os.makedirs(path_plots, exist_ok = True)

    m_dm = args.m_dm[0]
    cmap = LinearSegmentedColormap.from_list("", config["Colors"]["cmap_flux"])
    indices = np.linspace(0, 1, len(args.channel))
    colors = cmap(indices)

    fig, ax = plt.subplots(1, 1, figsize = config["Figure_size"]["single_column_legend"])
    for channel, color, label in zip(args.channel, colors, args.labels):
        fluxes = PrimaryFlux(mDM = m_dm, channel = channel)
        if m_dm != fluxes.mDM:
            raise ValueError("Specified DM mass does is not available in gammapy!" + "m_dm: " + str(m_dm) + ", Closest gammapy dark matter mass: " + str(fluxes.mDM))
        fluxes.table_model.plot(
            energy_bounds = [m_dm / 100, m_dm], 
            ax = ax, 
            label = label, 
            yunits = u.Unit("1/GeV"),
            color = color
            )
    ax.set_xlabel("$E$ [GeV]"),
    ax.set_ylabel("d$N$/d$E$ [GeV$^{-1}$]"),
    ax.set_yscale("log")
    plt.legend(bbox_to_anchor=(0, 1.05, 1, 0.2), loc="center", mode="expand", borderaxespad=0, ncol=2, alignment="center")
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_plots + "dm_spectra.pdf")
    print("Plot saved to", path_plots + "dm_spectra.pdf")