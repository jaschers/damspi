from astropy import units as u
from astropy import constants as const
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from gammapy.astro.darkmatter import PrimaryFlux

class FluxCalculator:
    """
    Class to calculate the gamma flux from DM annihilation around IMBHs

    Parameters
    ----------
    bh_catalogue : pandas.DataFrame
        BH catalogue with the following columns:
        - z_f: final redshift
        - r_sp [pc]: spacial radius
        - gamma_sp: spacial slope
        - rho(r_sp) [GeV/cm3]: spacial density
        - d_Sun [kpc]: distance to the Sun
        - m [M_solar]: BH mass
    
    Attributes
    ----------
    z_f : numpy.ndarray
        formation redshift of the IMBHs
    t_0 : numpy.ndarray
        age of the universe
    t_f : numpy.ndarray
        formation time of the IMBHs
    r_sp : numpy.ndarray
        spike radius
    gamma_sp : numpy.ndarray
        spike index
    rho_r_sp : numpy.ndarray
        density at spike radius
    distance : numpy.ndarray
        distance to the Sun
    M_bh : numpy.ndarray
        BH mass
    r_schw : numpy.ndarray
        Schwarzschild radius
    r_cut : numpy.ndarray
        cut-off radius
    """

    def __init__(self, bh_catalogue):
        self.bh_catalogue = bh_catalogue
        self.z_f = self.bh_catalogue['z_f'].values
        self.t_0 = self.redshift_to_time(0)
        self.t_f = self.redshift_to_time(self.z_f)
        self.r_sp = self.bh_catalogue['r_sp [pc]'].values * u.pc
        self.gamma_sp = self.bh_catalogue['gamma_sp'].values
        self.rho_r_sp = self.bh_catalogue['rho(r_sp) [GeV/cm3]'].values * u.GeV / u.cm**3
        self.distance = self.bh_catalogue['d_Sun [kpc]'].values * u.kpc
        self.M_bh = self.bh_catalogue['m [M_solar]'].values * u.M_sun
        self.r_schw = self.radius_schw(self.M_bh)

    @staticmethod
    def redshift_to_time(z):
        """
        Convert redshift to time
        
        Parameters
        ----------
        z : float
            redshift

        Returns
        -------
        time : astropy.units.quantity.Quantity
            time
        """

        # Convert redshift to time
        time = cosmo.age(z).to(u.Gyr)
        return time

    @staticmethod
    def radius_lim(m_dm, sigma_v, time_0, time_f, r_sp, rho_r_sp, gamma_sp):
        """
        Calculate limitation radius used to calculate the cut-off radius

        Parameters
        ----------
        m_dm : astropy.units.quantity.Quantity
            DM mass
        sigma_v : astropy.units.quantity.Quantity
            velocity weighted DM cross section
        time_0 : astropy.units.quantity.Quantity
            age of the universe
        time_f : astropy.units.quantity.Quantity
            formation time of the IMBHs
        r_sp : astropy.units.quantity.Quantity
            spike radius
        rho_r_sp : astropy.units.quantity.Quantity
            density at spike radius
        gamma_sp : float
            spike index

        Returns
        -------
        r_lim : astropy.units.quantity.Quantity
            limitation radius
        """

        rho_lim = (m_dm / (sigma_v * (time_0 - time_f))).to(u.GeV / u.cm**3)
        r_lim = (r_sp * (rho_lim / rho_r_sp)**(- 1 / gamma_sp)).to(u.pc)
        return(r_lim)

    @staticmethod
    def radius_schw(M_bh):
        """
        Calculate Schwarzschild radius

        Parameters
        ----------
        M_bh : astropy.units.quantity.Quantity
            BH mass

        Returns
        -------
        r_schw : astropy.units.quantity.Quantity
            Schwarzschild radius
        """

        r_schw = (2 * const.G * M_bh / const.c**2).to(u.pc)
        return(r_schw)

    def radius_cut(self, m_dm, sigma_v):
        """
        Calculate cut-off radius

        Parameters
        ----------
        m_dm : astropy.units.quantity.Quantity
            DM mass
        sigma_v : astropy.units.quantity.Quantity
            velocity weighted DM cross section

        Returns
        -------
        r_cut : astropy.units.quantity.Quantity
            cut-off radius
        """

        r_schw = self.radius_schw(self.M_bh)
        r_lim = self.radius_lim(m_dm, sigma_v, self.t_0, self.t_f, self.r_sp, self.rho_r_sp, self.gamma_sp)
        r_cut = np.maximum(4*r_schw.to(u.pc).value, r_lim.to(u.pc).value)
        self.r_cut = r_cut * u.pc
        return(self.r_cut)

    @staticmethod
    def N_gamma(m_dm, channel, E_th):
        """
        Calculate number of gamma photons from DM annihilation

        Parameters
        ----------
        m_dm : astropy.units.quantity.Quantity
            DM mass
        channel : str
            annihilation channel
        E_th : astropy.units.quantity.Quantity
            threshold energy

        Returns
        -------
        N : astropy.units.quantity.Quantity
            number of gamma photons from DM annihilation
        """

        # get number of gamma photons from DM annihilation with DM masses specified
        flux_annihi = PrimaryFlux(mDM = m_dm, channel = channel)

        if m_dm != flux_annihi.mDM:
            raise ValueError("Specified DM mass does is not available in gammapy!" + "m_dm: " + str(m_dm) + ", Closest gammapy dark matter mass: " + str(flux_annihi.mDM))

        N = flux_annihi.table_model.integral(E_th, m_dm).to(u.dimensionless_unscaled)
        return(N)

    def gamma_flux(self, m_dm, channel, E_th, sigma_v):
        """
        Calculate gamma flux from DM annihilation around IMBHs

        Parameters
        ----------
        m_dm : astropy.units.quantity.Quantity
            DM mass
        channel : str
            annihilation channel
        E_th : astropy.units.quantity.Quantity
            threshold energy
        sigma_v : astropy.units.quantity.Quantity
            velocity weighted DM cross section

        Returns
        -------
        y : astropy.units.quantity.Quantity
            gamma-ray flux from DM annihilation around IMBHs
        """

        N = self.N_gamma(m_dm, channel, E_th)

        # prefactor
        alpha = 0.5 * sigma_v * N / (m_dm**2 * self.distance**2)

        # term for 2r_schw < r < r_cut
        y1 = 0.5 * alpha * self.rho_r_sp**2 * self.r_cut * (self.r_sp / self.r_cut)**(2*self.gamma_sp) * (self.r_cut**2 - 4 * self.r_schw**2)

        # term for r_cut < r < r_sp
        # reference: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.72.103517?casa_token=-e4eEEaCw5oAAAAA%3A9KIbORPLYWRlSVC5MyI3HSIslOLws15IjLMCUkoM2E3uD9PaUV_cXtfBta2anEGB9Epsa-J9DZ9qUiI
        y2 = 2 * alpha * self.rho_r_sp**2 * self.r_sp**3 / (4*self.gamma_sp - 6) * (self.r_cut / self.r_sp)**(-2*self.gamma_sp + 3)

        y = y1 + y2

        y = y.to(1 / (u.cm**2 * u.s))

        return(y)
