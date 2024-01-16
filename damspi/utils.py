import numpy as np
import astropy.units as u
import argparse
import eagleSqlTools as sql
import pandas as pd
import astropy.constants as const 

def check_core_index_range(value):
    """
    Checks if core index is in range [0, 1)

    Parameters
    ----------
    value : float
        Core index

    Returns
    -------
    fvlaue : float
        Core index
    """
    fvalue = float(value)
    if fvalue < 0 or fvalue >= 1:
        raise argparse.ArgumentTypeError(f"{value} is out of range [0, 1)")
    return fvalue

def add_catalogue_args(parser):
    """
    Adds relevant IMBH calalogue arguments to the parser in the parse_args function

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    """

    parser.add_argument("-dmp", "--dark_matter_profile", type = str, required = False, default = "nfw", metavar = "-", choices = ["nfw", "cored"], help = "Dark matter density profile. Can be: 'nfw', 'cored'. Default: nfw")
    parser.add_argument("-ci", "--core_index", required = False, default = None, metavar = "-", type = check_core_index_range, help = "Cored dark matter density profile core index. Needs to fullfill: 0 <= core_index < 1. If None, the core index is a free paramter. Otherwise, the core index is fixed to specified value. Only has an effect if --dark_matter_profile='cored'. Default: None")
    parser.add_argument("-sa", "--save_animation", type = str, required = False, default = "n", metavar = "-", help = "Bool if animations for individual galaxies are saved (takes a lot of time) [y, n], default: n")
    parser.add_argument("-ltc", "--load_temporary_catalogue", type = str, required = False, default = "n", metavar = "-", help = "Yes, if temporary catalogue extracted in a previous run should be loaded. Only possible if create_catalogue.py was already run with the same -n <name> option in the past. Can be: 'y' or 'n'. Default: 'n'")

def add_upsampling_args(parser):
    """
    Adds relevant upsampling arguments to the parser in the parse_args function

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    """

    parser.add_argument("-uf", "--upsampling_factor", type = int, required = False, default = 1, metavar = "-", help = "Upsampling factor of IMBHs for 2D map. Default: 1 (no upsampling)")

def add_dark_matter_args(parser):
    """
    Adds relevant dark matter arguments to the parser in the parse_args function

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    """

    parser.add_argument("-mdm", "--m_dm", type = float, required = False, nargs = "+", default = [500, 1500, 3], metavar = "-", help = "Mass of dark matter particle in GeV. Can be single input or mass range + number of masses (three inputs). If mass range is given, scaling can be specified by the mass_dm_scaling argument. Default: 500 1500 3")
    parser.add_argument("-mdms", "--m_dm_scaling", type = str, required = False, default = "linear", metavar = "-", help = "Scaling of dark matter particle mass. Can be linear or log. Default: linear")
    parser.add_argument("-sv", "--sigma_v", type = float, required = False, nargs = "+", default = [3e-26], metavar = "-", help = "Dark matter (velocity weighted) annihilation cross section in cm3/s. Default: 3e-26")
    parser.add_argument("-svs", "--sigma_v_scaling", type = str, required = False, default = "log", metavar = "-", help = "Scaling of dark matter (velocity weighted) annihilation cross section. Can be linear or log. Default: log")
    parser.add_argument("-c", "--channel", type = str, required = False, nargs = "+", default = ["b"], metavar = "-", help = "Dark matter annihilation channel. Can be: 'V->e', 'V->mu', 'V->tau', 'W', 'WL', 'WT', 'Z', 'ZL', 'ZT', 'b', 'c', 'e', 'eL', 'eR', 'g', 'gamma', 'h', 'mu', 'muL', 'muR', 'nu_e', 'nu_mu', 'nu_tau', 'q', 't', 'tau', 'tauL', 'tauR'. Default: b")
    parser.add_argument("-eth", "--E_th", type = float, required = False, default = 100, metavar = "-", help = "Lower energy threshold to calculate number of gamma rays per dark matter annihilation in GeV. Default: 100")
    parser.add_argument("-ic", "--instrument_comparison", type = str, required = False, default = "hess", choices = ["hess", "fermi"], metavar = "-", help = "Instrument to which the results are compared to. Default: hess")

def add_plot_args(parser):
    """
    Adds relevant plot arguments to the parser in the parse_args function

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    """

    parser.add_argument("-plt", "--plot", type = str, required = False, default = "n", metavar = "-", help = "Bool if plots are saved [y, n], default: n")

def add_labels_args(parser):
    """
    Adds relevant label arguments to the parser in the parse_args function
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    """

    parser.add_argument("-l", "--labels", type = str, nargs = "+", default = ["NFW", r"free $\gamma_\mathrm{c}$", r"$\gamma_\mathrm{c} = 0.4$", r"$\gamma_\mathrm{c} = 0.0$"], metavar = "-", help = "Labels used in the plot legend.")

def add_name_args(parser):
    """
    Adds relevant name arguments to the parser in the parse_args function

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser
    """

    parser.add_argument("-n", "--name", type = str, required = True,  nargs = "+", metavar = "-", help = "Suffix of the output filenames, e.g. catalogue_<name>.csv.")

def parse_args(include_name = True, include_cat = False, include_dm = False, include_plot = False, include_labels = False, include_upsampling = False):
    """
    Parses the arguments of the command line

    Parameters
    ----------
    include_name : bool, optional
        Bool if name arguments should be included, by default True
    include_cat : bool, optional
        Bool if catalogue arguments should be included, by default False
    include_dm : bool, optional
        Bool if dark matter arguments should be included, by default False
    include_plot : bool, optional
        Bool if plot arguments should be included, by default False
    include_labels : bool, optional
        Bool if label arguments should be included, by default False

    Returns
    -------
    args : argparse.Namespace
        Namespace containing all arguments
    """

    script_descr="""
    Extracts IMBH catalogue from EAGLE data
    """

    # Open argument parser
    parser = argparse.ArgumentParser(description=script_descr)

    parser.add_argument("-sn", "--sim_name", type = str, required = False, default = "RefL0100N1504", metavar = "-", help = "Name of the EAGLE simulation, default: RefL0100N1504") # RefL0025N0376 # RefL0050N0752 #RefL0100N1504
    parser.add_argument("-nf", "--number_files", type = int, required = False, default = 256, metavar = "-", help = "Number of files for the particle data, default: 256") # 16 # 128 # 256
    parser.add_argument("-bs", "--box_size", type = int, required = False, default = 100, metavar = "-", help = "Box size of the simulation in Mpc, default: 100")

    if include_cat:
        add_catalogue_args(parser)
    if include_dm:
        add_dark_matter_args(parser)
    if include_plot:
        add_plot_args(parser)
    if include_labels:
        add_labels_args(parser)
    if include_name:
        add_name_args(parser)
    if include_upsampling:
        add_upsampling_args(parser)

    # Define expected arguments
    args = parser.parse_args()
    print("####### Setup #######")
    print(vars(args))

    if include_name:
        if len(args.name) == 1:
            args.name = args.name[0]

    args.box_size = args.box_size * u.Mpc
    if include_plot:
        args.plot = convert_to_bool(args.plot)

    if include_cat:
        args.save_animation = convert_to_bool(args.save_animation)
        args.load_temporary_catalogue = convert_to_bool(args.load_temporary_catalogue)

    if include_dm:
        args.E_th = args.E_th * u.GeV
        if len(args.m_dm) == 3:
            if args.m_dm_scaling == 'linear':
                args.m_dm = np.linspace(args.m_dm[0], args.m_dm[1], int(args.m_dm[2])) * u.GeV
            elif args.m_dm_scaling == 'log':
                args.m_dm = np.logspace(np.log10(args.m_dm[0]), np.log10(args.m_dm[1]), int(args.m_dm[2])) * u.GeV
            print("Dark matter masses: ", args.m_dm)
        else:
            args.m_dm = args.m_dm * u.GeV

        if len(args.sigma_v) == 3:
            if args.sigma_v_scaling == 'linear':
                args.sigma_v = np.linspace(args.sigma_v[0], args.sigma_v[1], int(args.sigma_v[2])) * u.cm**3 / u.s
            elif args.sigma_v_scaling == 'log':
                args.sigma_v = np.logspace(np.log10(args.sigma_v[0]), np.log10(args.sigma_v[1]), int(args.sigma_v[2])) * u.cm**3 / u.s
            print("Dark matter (velocity weighted) annihilation cross sections: ", args.sigma_v)
        else:
            args.sigma_v = args.sigma_v * u.cm**3 / u.s

        if len(args.channel) == 1:
            args.channel = args.channel[0]

    if include_upsampling:
        if args.upsampling_factor < 1:
            raise ValueError('Upsampling factor must be >= 1')
        args.upsampling_factor = args.upsampling_factor - 1

    return args

def convert_float_to_three_digit(number):
    """
    Converts a float to a three digit number

    Parameters
    ----------
    number : float
        Float to be converted

    Returns
    -------
    three_digit_number : str
        Three digit number
    """

    three_digit_number = '{:06d}'.format(int(number * 1000))
    half_length = len(three_digit_number) // 2
    first_half = str(three_digit_number[:half_length])
    return(first_half)

def convert_float_to_six_digit(number):
    """
    Converts a float to a six digit number

    Parameters
    ----------
    number : float
        Float to be converted

    Returns
    -------
    first_half : str
        First half of the six digit number
    second_half : str
        Second half of the six digit number
    """

    six_digit_number = '{:06d}'.format(int(np.round(number * 1000)))
    half_length = len(six_digit_number) // 2
    first_half = str(six_digit_number[:half_length])
    second_half = str(six_digit_number[half_length:])
    return(first_half, second_half)

def redshift(a):
    """
    Calculates the redshift from the scale factor

    Parameters
    ----------
    a : float
        Scale factor

    Returns
    -------
    z : float
        Redshift
    """

    z = (1 - a) / a
    z[z == np.inf] = 0
    return(z)

def convert_to_bool(string):
    """
    Converts a string to a bool

    Parameters
    ----------
    string : str
        String to be converted

    Returns
    -------
    bool
        Bool
    """

    if string == 'y':
        return(True)
    elif string == 'n':
        return(False)
    else:
        raise ValueError('Input must be either y or n')

def nfw_profile(params, r):
    """
    NFW density profile

    Parameters
    ----------
    params : tuple
        Tuple of parameters (rho_0, r_s)
    r : float
        Distance to the centre of the halo

    Returns
    -------
    rho : float
        Density
    """

    rho_0, r_s = params
    rho = (rho_0 * (r/r_s)**(-1) * (1+r/r_s)**(-2)).to(u.Msun/u.kpc**3)
    return(rho)

def nfw_integral(r, rho_0, r_s):
    """
    NFW density profile integral

    Parameters
    ----------
    r : float
        Distance to the centre of the halo
    rho_0 : float
        Density at the scale radius
    r_s : float
        Scale radius

    Returns
    -------
    y : float
        Integral of the NFW density profile
    """

    y = (4 * np.pi * rho_0 * r_s**3 * (np.log((r_s + r) / r_s) - r / (r_s + r))).to(u.Msun)
    return(y)

def cored_profile(params, r):
    """
    Cored density profile

    Parameters
    ----------
    params : tuple
        Tuple of parameters (rho_0, r_s, r_c, gamma_c)
    r : float
        Distance to the centre of the halo

    Returns
    -------
    rho : float
        Cored density profile
    """

    rho_0, r_s, r_c, gamma_c = params
    rho_nfw = nfw_profile((rho_0, r_s), r)
    rho_nfw_rc = nfw_profile((rho_0, r_s), r_c)
    rho_cored = rho_nfw_rc * (r/r_c)**(-gamma_c)
    mask = r < r_c
    rho = np.where(mask, rho_cored, rho_nfw)
    return(rho)

def cored_integral(r, rho_0, r_s, r_c, gamma_c):
    """
    Cored density profile integral

    Parameters
    ----------
    r : float
        Distance to the centre of the halo
    rho_0 : float
        Density at the scale radius
    r_s : float
        Scale radius
    r_c : float
        Core radius
    gamma_c : float
        Core index

    Returns
    -------
    y : float
        Integral of the cored density profile
    """

    # integral if r < r_c
    y_cored = 4 * np.pi * nfw_profile((rho_0, r_s), r_c) * r_c**(gamma_c) * r**(3 - gamma_c) / (3 - gamma_c)
    # integral if r > r_c
    y_1 = 4 * np.pi * nfw_profile((rho_0, r_s), r_c) * r_c**(gamma_c) * r_c**(3 - gamma_c) / (3 - gamma_c)
    y_2 = 4 * np.pi * rho_0 * r_s**3 * (np.log((r + r_s) / r_s) + r_s / (r + r_s) - np.log((r_c + r_s) / r_s) - r_s / (r_c + r_s))
    y_outer = y_1 + y_2
    mask = r < r_c
    y_total = np.where(mask, y_cored, y_outer)
    return(y_total)

def M_bh_2(M_bh):
    """
    Twice the black hole mass

    Parameters
    ----------
    M_bh : float
        Black hole mass

    Returns
    -------
    y : float
        Twice the black hole mass
    """

    y = (2 * M_bh).to(u.Msun)
    return(y)

def rescaled_distance(r, m200):
    """
    Rescales the distance to the centre of the halo

    Parameters
    ----------
    r : float
        Distance to the centre of the halo
    m200 : float
        M200 of the halo

    Returns
    -------
    r : float
        Rescaled distance to the centre of the halo
    """

    r = r * (1e12 * u.Msun / m200)**(1/3)
    return(r)

def remove_distant_satellites(table_bh, nsnap, args):
    """
    Removes satellites that are too far away from the host galaxy

    Parameters
    ----------
    table_bh : pandas.DataFrame
        Table containing the black hole data
    nsnap : int
        Snapshot number
    args : argparse.Namespace
        Namespace containing all arguments

    Returns
    -------
    table_bh : pandas.DataFrame
        Table containing the black hole data
    """

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
    """
    Calculates the mean of the parameter distribution

    Parameters
    ----------
    table : pandas.DataFrame
        Table containing the black hole data
    parameter : str
        Parameter to be analysed
    bins : int
        Number of bins

    Returns
    -------
    hist_mean : numpy.ndarray
        Mean of the parameter distribution
    hist_mean_error : numpy.ndarray
        Error of the mean of the parameter distribution
    """

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

def gamma_core(gamma_sp):
    """
    Calculates the core index of the cored dark matter density profile

    Parameters
    ----------
    gamma_sp : float
        Spike index

    Returns
    -------
    gamma_core : float
        Core index
    """

    gamma_core = (9-4*gamma_sp)/(2-gamma_sp)
    return(gamma_core)

def spike_profile(params, r):
    """
    Spike density profile

    Parameters
    ----------
    params : tuple
        Tuple of parameters (rho_0, r_s, r_sp, gamma_sp)
    r : float
        Distance to the centre of the black hole

    Returns
    -------
    rho : float
        Spike density profile
    """

    rho_0, r_s, r_sp, gamma_sp = params
    return nfw_profile((rho_0, r_s), r_sp) * (r/r_sp)**(-gamma_sp)

def imbh_profile(params, r):
    """
    IMBH density profile

    Parameters
    ----------
    params : tuple
        Tuple of parameters (rho_0, r_schw, r_cut, r_s, r_sp, gamma_sp)
    r : float
        Distance to the centre of the black hole

    Returns
    -------
    rho : float
        IMBH density profile
    """

    rho_0, r_schw, r_cut, r_s, r_sp, gamma_sp = params

    if r <= 2 * r_schw:
        return None
    elif 2 * r_schw < r <= r_cut:
        y = spike_profile((rho_0, r_s, r_sp, gamma_sp), r_cut) * (r/r_cut)**(-0.5)
        y = (y * const.c ** 2).to(u.GeV / u.cm ** 3).value
        return y
    elif r_cut < r <= r_sp:
        y = spike_profile((rho_0, r_s, r_sp, gamma_sp), r)
        y = (y * const.c ** 2).to(u.GeV / u.cm ** 3).value
        return y
    else:  # r > r_sp
        y = nfw_profile((rho_0, r_s), r)
        y = (y * const.c ** 2).to(u.GeV / u.cm ** 3).value
        return y

def format_energy(energy):
    """
    Formats any energy value to a string expressed in GeV.

    Parameters:
    energy (Quantity): The energy value with its unit.

    Returns:
    str: Formatted string representation of the energy value in GeV.
    """
    energy_gev = energy.to(u.GeV).value
    return f"{energy_gev:.1f}GeV"