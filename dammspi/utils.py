import numpy as np
import astropy.units as u
import argparse
import eagleSqlTools as sql
import pandas as pd

def parse_args():
    script_descr="""
    Extracts IMBH catalogue from EAGLE data
    """

    # Open argument parser
    parser = argparse.ArgumentParser(description=script_descr)

    # Define expected arguments
    parser.add_argument("-sn", "--sim_name", type = str, required = False, default = "RefL0100N1504", metavar = "-", help = "Name of the EAGLE simulation, default: RefL0100N1504") # RefL0025N0376 # RefL0050N0752 #RefL0100N1504
    parser.add_argument("-nf", "--number_files", type = int, required = False, default = 256, metavar = "-", help = "Number of files for the particle data, default: 256") # 16 # 128 # 256
    parser.add_argument("-bs", "--box_size", type = int, required = False, default = 100, metavar = "-", help = "Box size of the simulation in Mpc, default: 100")
    parser.add_argument("-plt", "--plot", type = str, required = False, default = "n", metavar = "-", help = "Bool if plots for individual galaxies are saved (takes some time) [y, n], default: n")
    parser.add_argument("-sa", "--save_animation", type = str, required = False, default = "n", metavar = "-", help = "Bool if animations for individual galaxies are saved (takes a lot of time) [y, n], default: n")
    parser.add_argument("-mdm", "--m_dm", type = float, required = False, nargs = "+", default = [500], metavar = "-", help = "Mass of dark matter particle in GeV. Can be single input or mass range + number of masses (three inputs). If mass range is given, scaling can be specified by the mass_dm_scaling argument. Default: 500 GeV")
    parser.add_argument("-mdms", "--m_dm_scaling", type = str, required = False, default = "linear", metavar = "-", help = "Scaling of dark matter particle mass. Can be linear or log. Default: linear")
    parser.add_argument("-sv", "--sigma_v", type = float, required = False, nargs = "+", default = [3e-26], metavar = "-", help = "Dark matter (velocity weighted) annihilation cross section in cm^3/s. Can be Can be single input or cross section range + number of cross sections (three inputs). If cross section range is given, scaling can be specified by the cross_section_scaling argument. Default: 3e-26 cm^3/s")
    parser.add_argument("-svs", "--sigma_v_scaling", type = str, required = False, default = "log", metavar = "-", help = "Scaling of dark matter (velocity weighted) annihilation cross section. Can be linear or log. Default: log")
    parser.add_argument("-c", "--channel", type = str, required = False, default = "b", metavar = "-", help = "Dark matter annihilation channel. Can be: 'V->e', 'V->mu', 'V->tau', 'W', 'WL', 'WT', 'Z', 'ZL', 'ZT', 'b', 'c', 'e', 'eL', 'eR', 'g', 'gamma', 'h', 'mu', 'muL', 'muR', 'nu_e', 'nu_mu', 'nu_tau', 'q', 't', 'tau', 'tauL', 'tauR'. Default: b")
    parser.add_argument("-eth", "--E_th", type = float, required = False, default = 100, metavar = "-", help = "Lower energy threshold to calculate number of gamma rays per dark matter annihilation in GeV. Default: 100 GeV")

    args = parser.parse_args()
    print("####### Setup #######")
    print(vars(args))

    args.plot = convert_to_bool(args.plot)
    args.save_animation = convert_to_bool(args.save_animation)
    args.E_th = args.E_th * u.GeV
    args.box_size = args.box_size * u.Mpc

    if len(args.m_dm) == 3:
        if args.m_dm_scaling == 'linear':
            args.m_dm = np.linspace(args.m_dm[0], args.m_dm[1], int(args.m_dm[2])) * u.GeV
        elif args.m_dm_scaling == 'log':
            args.m_dm = np.logspace(np.log10(args.m_dm[0]), np.log10(args.m_dm[1]), int(args.m_dm[2])) * u.GeV
    else:
        args.m_dm = args.m_dm * u.GeV

    if len(args.sigma_v) == 3:
        if args.sigma_v_scaling == 'linear':
            args.sigma_v = np.linspace(args.sigma_v[0], args.sigma_v[1], int(args.sigma_v[2])) * u.cm**3 / u.s
        elif args.sigma_v_scaling == 'log':
            args.sigma_v = np.logspace(np.log10(args.sigma_v[0]), np.log10(args.sigma_v[1]), int(args.sigma_v[2])) * u.cm**3 / u.s
    else:
        args.sigma_v = args.sigma_v * u.cm**3 / u.s

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

def rescaled_distance(r, m200):
    r = r * (1e12 * u.Msun / m200)**(1/3)
    return(r)

def remove_distant_satellites(table_bh, nsnap, args):
    group_number = table_bh['group number'].values[0]
    subgroup_numbers = tuple(np.unique(table_bh['subgroup number'].values))

    if len(subgroup_numbers) > 1:
        r_min, r_max = 0.040 * u.Mpc, 0.300 * u.Mpc

        query = f"SELECT \
                    SH.GroupNumber as group_number, \
                    SH.SubGroupNumber as subgroup_number, \
                    SH.CentreOfPotential_x as cop_x, \
                    SH.CentreOfPotential_y as cop_y, \
                    SH.CentreOfPotential_z as cop_z, \
                    FOF.Group_M_Crit200 as m200 \
                FROM \
                    {args.sim_name}_SubHalo as SH, \
                    {args.sim_name}_FOF as FOF \
                WHERE \
                    SH.Snapnum = {nsnap} \
                    and SH.GroupNumber = {group_number} \
                    and SH.SubGroupNumber in {subgroup_numbers} \
                    and FOF.Snapnum = SH.Snapnum \
                    and FOF.GroupID = SH.GroupID \
                    and SH.Spurious = 0"

        # Execute query.
        con = sql.connect("dvd351", password="zqfARI55") # username, password
        # load galaxy data
        data_galaxy = sql.execute_query(con, query)
        table_galaxy = pd.DataFrame(data_galaxy)

        host_galaxy = table_galaxy[table_galaxy['subgroup_number'] == 0]
        host_cop = host_galaxy[['cop_x', 'cop_y', 'cop_z']].values[0] * u.Mpc

        r = np.sqrt(
            (table_galaxy['cop_x'].values * u.Mpc - host_cop[0])**2 + 
            (table_galaxy['cop_y'].values * u.Mpc - host_cop[1])**2 + 
            (table_galaxy['cop_z'].values * u.Mpc - host_cop[2])**2
            )

        m200 = table_galaxy['m200'].values * u.Msun
        r_rescaled = rescaled_distance(r, m200)
        table_galaxy['rescaled distance [Mpc]'] = r_rescaled.value

        condition = (r_rescaled > r_min) & (r_rescaled < r_max)
        table_close_satellites = table_galaxy[condition]
        subgroup_numbers_close_satellites = np.unique(table_close_satellites['subgroup_number'].values)
        # add host galaxy back to valid subgroup numbers
        subgroup_numbers_valid = np.append(subgroup_numbers_close_satellites, 0)

        table_bh = table_bh[table_bh['subgroup number'].isin(subgroup_numbers_valid)].reset_index(drop=True)
        
        return(table_bh)

    else:
        return(table_bh)
    
def parameter_distr_mean(table, parameter, bins):
    hist_list = []
    table_parameter = table[parameter]
    for galaxy_id in np.unique(table["galaxy_id"].values):
        data_galaxy_id = table_parameter[table["galaxy_id"].values == galaxy_id]
        hist, _ = np.histogram(data_galaxy_id, bins = bins)
        hist_list.append(hist)
    
    hist_mean = np.mean(hist_list, axis = 0)
    hist_std = np.std(hist_list, ddof = 1, axis = 0)
    hist_mean_error = hist_std / np.sqrt(len(hist_list))

    return hist_mean, hist_mean_error