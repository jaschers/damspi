from astropy import units as u
from astropy import constants as const
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from gammapy.astro.darkmatter import PrimaryFlux

class FluxCalculator:
    def __init__(self, bh_catalogue, dm_profile):
        self.bh_catalogue = bh_catalogue
        self.z_f = self.bh_catalogue['z_f'].values
        self.t_0 = self.redshift_to_time(0)
        self.t_f = self.redshift_to_time(self.z_f)
        self.r_sp = self.bh_catalogue['r_sp [pc]'].values * u.pc
        self.gamma_sp = self.bh_catalogue['gamma_sp'].values
        self.rho_r_sp = self.bh_catalogue['rho(r_sp) [GeV/cm3]'].values * u.GeV / u.cm**3
        self.distance = self.bh_catalogue['d_Sun [kpc]'].values * u.kpc
        self.M_bh = self.bh_catalogue['m [M_solar]'].values * u.M_sun

    @staticmethod
    def redshift_to_time(z):
        # Convert redshift to time
        time = cosmo.age(z).to(u.Gyr)
        return time

    @staticmethod
    def radius_lim(m_dm, sigma_v, time_0, time_f, r_sp, rho_r_sp, gamma_sp):
        rho_lim = (m_dm / (sigma_v * (time_0 - time_f))).to(u.GeV / u.cm**3)
        r_lim = (r_sp * (rho_lim / rho_r_sp)**(- 1 / gamma_sp)).to(u.pc)
        return(r_lim)

    @staticmethod
    def radius_schw(M_bh):
        r_schw = (2 * const.G * M_bh / const.c**2).to(u.pc)
        return(r_schw)

    def radius_cut(self, m_dm, sigma_v):
        r_schw = self.radius_schw(self.M_bh)
        r_lim = self.radius_lim(m_dm, sigma_v, self.t_0, self.t_f, self.r_sp, self.rho_r_sp, self.gamma_sp)
        r_cut = np.maximum(4*r_schw.to(u.pc).value, r_lim.to(u.pc).value)
        self.r_cut = r_cut * u.pc
        return(self.r_cut)

    @staticmethod
    def N_gamma(m_dm, channel, E_th):
        # get number of gamma photons from DM annihilation with DM masses specified
        flux_annihi = PrimaryFlux(mDM = m_dm, channel = channel)
        N = flux_annihi.table_model.integral(E_th, m_dm).to(u.dimensionless_unscaled)
        return(N)

    def gamma_flux(self, m_dm, channel, E_th, sigma_v):
        N = self.N_gamma(m_dm, channel, E_th)

        # reference: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.72.103517?casa_token=-e4eEEaCw5oAAAAA%3A9KIbORPLYWRlSVC5MyI3HSIslOLws15IjLMCUkoM2E3uD9PaUV_cXtfBta2anEGB9Epsa-J9DZ9qUiI
        y = N * (self.rho_r_sp**2 * sigma_v * self.r_sp**3) / ((4 * self.gamma_sp - 6) * m_dm**2 * self.distance**2) * (self.r_cut / self.r_sp)**(-2 * self.gamma_sp + 3)
        y = y.to(1 / (u.cm**2 * u.s))

        return(y)

