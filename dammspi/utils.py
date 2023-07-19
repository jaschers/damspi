import numpy as np
import astropy.units as u
import argparse

def parse_args():
    script_descr="""
    Extracts IMBH catalogue from EAGLE data
    """

    # Open argument parser
    parser = argparse.ArgumentParser(description=script_descr)

    # Define expected arguments
    parser.add_argument("-sn", "--sim_name", type = str, required = False, default = "RefL0025N0376", metavar = "-", help = "Name of the EAGLE simulation, default: RefL0025N0376")
    parser.add_argument("-nf", "--number_files", type = int, required = False, default = 16, metavar = "-", help = "Number of files for the particle data, default: 16")
    parser.add_argument("-plt", "--plot", type = str, required = False, default = "n", metavar = "-", help = "Bool if plots for individual galaxies are saved (takes some time) [y, n], default: n")
    parser.add_argument("-sa", "--save_animation", type = str, required = False, default = "n", metavar = "-", help = "Bool if animations for individual galaxies are saved (takes a long time) [y, n], default: n")

    args = parser.parse_args()
    print("####### Setup #######")
    print(vars(args))

    args.plot = convert_to_bool(args.plot)
    args.save_animation = convert_to_bool(args.save_animation)

    return args

def convert_float_to_three_digit(number):
        three_digit_number = '{:06d}'.format(int(number * 1000))
        half_length = len(three_digit_number) // 2
        first_half = str(three_digit_number[:half_length])
        return(first_half)

def convert_float_to_six_digit(number):
    six_digit_number = '{:06d}'.format(int(np.round(number * 1000)))
    half_length = len(six_digit_number) // 2
    first_half = str(six_digit_number[:half_length])
    second_half = str(six_digit_number[half_length:])
    return(first_half, second_half)

def redshift(a):
    z = (1 - a) / a
    z[z == np.inf] = 0
    return(z)

def convert_to_bool(string):
    if string == 'y':
        return(True)
    elif string == 'n':
        return(False)
    else:
        raise ValueError('Input must be either y or n')

def nfw_profile(params, r):
    rho_0, r_s = params
    rho = (rho_0 * (r/r_s)**(-1) * (1+r/r_s)**(-2)).to(u.Msun/u.kpc**3)
    return(rho)

def nfw_integral(r, rho_0, r_s):
    y = (4 * np.pi * rho_0 * r_s**3 * (np.log((r_s + r) / r_s) - r / (r_s + r))).to(u.Msun)
    return(y)

def M_bh_2(M_bh):
    y = (2 * M_bh).to(u.Msun)
    return(y)

