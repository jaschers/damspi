import numpy as np
import h5py
import argparse
import os
from astropy import units as u

######################################## argparse setup ########################################
script_descr="""
This script downloads the EAGLE particle data and saves the black hole data into the data/ directory.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-sn", "--sim_name", type = str, required = False, default = "RefL0050N0752", metavar = "-", help = "Name of the EAGLE simulation, default: RefL0050N0752")
parser.add_argument("-nf", "--number_files", type = int, required = False, default = 128, metavar = "-", help = "Number of files for the particle data, default: 128")

args = parser.parse_args()
print("####### Setup #######")
print(vars(args))
##########################################################################################

nsnap_list = np.linspace(0, 28, 29)
redshift_list = [20., 15.132, 9.993, 8.988, 8.075, 7.050, 5.971, 5.487, 5.037, 4.485, 3.984, 3.528, 3.017, 2.478, 2.237, 2.012, 1.737, 1.487, 1.259, 1.004, 0.865, 0.736, 0.615, 0.503, 0.366, 0.271, 0.183, 0.101, 0.]

dict_redshift = dict(zip(nsnap_list, redshift_list))

def convert_float_to_six_digit(number):
    six_digit_number = '{:06d}'.format(int(np.round(number * 1000)))
    half_length = len(six_digit_number) // 2
    first_half = str(six_digit_number[:half_length])
    second_half = str(six_digit_number[half_length:])
    return(first_half, second_half)

def convert_float_to_three_digit(number):
    three_digit_number = '{:06d}'.format(int(number * 1000))
    half_length = len(three_digit_number) // 2
    first_half = str(three_digit_number[:half_length])
    return(first_half)

# convert scale factor to redshift
def redshift(a):
    z = (1 - a) / a
    z[z == np.inf] = 0
    return(z)

# reads particle data set
def extract_bh_data(nsnap, sim_name = args.sim_name, nfiles = args.number_files):
    """ Read a selected dataset, itype is the PartType and att is the attribute name. """

    data = []

    redshift = dict_redshift[int(nsnap)]
    nsnap = convert_float_to_three_digit(nsnap)
    redshift_1, redshift_2 = convert_float_to_six_digit(redshift)

    # Loop over each file and extract the PARTICLE data.
    for i in range(nfiles):
        # f = h5py.File("./data/snap_028_z000p000.%i.hdf5"%i, "r")
        path = f"data/{sim_name}/snapshot_{nsnap}_z{redshift_1}p{redshift_2}/snap_{nsnap}_z{redshift_1}p{redshift_2}.%i.hdf5"%i
        path_new = f"data/{sim_name}/snapshot_{nsnap}_z{redshift_1}p{redshift_2}/snap_{nsnap}_z{redshift_1}p{redshift_2}.%i_new.hdf5"%i

        # Open the existing .h5 file
        with h5py.File(path, 'r') as f:
            # Create a new .h5 file
            with h5py.File(path_new, 'w') as new_f:
                # Copy 'Header' dataset to the new file
                f.copy('Header', new_f)
                
                if 'PartType5' in f.keys():
                    # Copy 'PartType5' dataset to the new file
                    f.copy('PartType5', new_f)
                
                # Save the new file
                new_f.flush()

        os.remove(path)
        os.rename(path_new, path)

def nsnap_three_digit(nsnap):
    return convert_float_to_three_digit(nsnap)

os.makedirs("data/", exist_ok = True)
for i in range(29):
    if not os.path.exists(f"data/{args.sim_name}_snap_{convert_float_to_three_digit(i)}.tar"):
        weblink = f"https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots//download?run={args.sim_name}&snapnum={i}"

        command = f"wget -P data/ --user=dvd351 --password zqfARI55 --content-disposition '{weblink}'"

        os.system(command)

    os.system(f"tar -xvf data/{args.sim_name}_snap_{convert_float_to_three_digit(i)}.tar -C hdf")

    extract_bh_data(nsnap = i)

    os.system(f"rm data/{args.sim_name}_snap_{convert_float_to_three_digit(i)}.tar")
