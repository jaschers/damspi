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

    bh_plotter = dammplot.BlackHolePlotter(sim_name = args.sim_name, table_bh = bh_catalogue)

    path = f"plots/{args.sim_name}/black_hole_dist/"
    os.makedirs(path, exist_ok = True)

    bh_plotter.plot_bh_dist_total(path)

    print(bh_catalogue)