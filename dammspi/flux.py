from astropy import units as u
from astropy import constants as const
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from gammapy.astro.darkmatter import PrimaryFlux

class FluxCalculator:
    def __init__(self, bh_catalogue):
        self.bh_catalogue = bh_catalogue
        self.z_f = self.bh_catalogue['z_f'].values
        self.t_0 = self.redshift_to_time(0)
        self.t_f = self.redshift_to_time(self.z_f)
        self.r_sp = self.bh_catalogue['r_sp [pc]'].values * u.pc
        self.rho_r_sp = self.bh_catalogue['rho(r_sp) [GeV/cm3]'].values * u.GeV / u.cm**3
        self.distance = self.bh_catalogue['d_Sun [kpc]'].values * u.kpc
        self.M_bh = self.bh_catalogue['m [M_solar]'].values * u.M_sun
        self.gamma = 7/3

    @staticmethod
    def redshift_to_time(z):
        # Convert redshift to time
        time = cosmo.age(z).to(u.Gyr)
        return time

    @staticmethod
    def radius_lim(m_dm, sigma_v, time_0, time_f, r_sp, rho_r_sp, gamma):
        rho_lim = (m_dm / (sigma_v * (time_0 - time_f))).to(u.GeV / u.cm**3)
        r_lim = (r_sp * (rho_lim / rho_r_sp)**(- 1 / gamma)).to(u.pc)
        return(r_lim)

    @staticmethod
    def radius_schw(M_bh):
        r_schw = (2 * const.G * M_bh / const.c**2).to(u.pc)
        return(r_schw)

    def radius_cut(self, m_dm, sigma_v):
        r_schw = self.radius_schw(self.M_bh)
        r_lim = self.radius_lim(m_dm, sigma_v, self.t_0, self.t_f, self.r_sp, self.rho_r_sp, self.gamma)
        r_cut = np.maximum(4*r_schw.to(u.pc).value, r_lim.to(u.pc).value)
        self.r_cut = r_cut * u.pc
        return(self.r_cut)

    @staticmethod
    def N_gamma(m_dm, channel, E_th):
        # get number of gamma photons from DM annihilation with DM masses specified
        flux_annihi = PrimaryFlux(mDM = m_dm, channel = channel)
        N = flux_annihi.table_model.integral(E_th, m_dm).to(u.dimensionless_unscaled)
        return(N)

    def imbh_flux(self, m_dm, channel, E_th, sigma_v):
        N = self.N_gamma(m_dm, channel, E_th)

        # reference: https://journals.aps.org/prd/pdf/10.1103/PhysRevD.72.103517?casa_token=-e4eEEaCw5oAAAAA%3A9KIbORPLYWRlSVC5MyI3HSIslOLws15IjLMCUkoM2E3uD9PaUV_cXtfBta2anEGB9Epsa-J9DZ9qUiI
        phi_0 = 9e-10 * u.cm**(-2) * u.s**(-1)
        sigma_v0 = 1e-26 * u.cm**(3) * u.s**(-1)
        m_dm_0 = 100 * u.GeV
        d_0 = 1 * u.kpc
        rho_r_sp_0 = 1e2 * u.GeV * u.cm**(-3)
        r_sp_0 = 1 * u.pc
        r_cut_0 = 1e-3 * u.pc

        y = phi_0 * N * (sigma_v / sigma_v0) * (m_dm / m_dm_0)**(-2) * (self.distance / d_0)**(-2) * (self.rho_r_sp / rho_r_sp_0)**2 * (self.r_sp / r_sp_0) ** (14/3) * (self.r_cut / r_cut_0)**(-5/3)

        y = y.to(1 / (u.cm**2 * u.s))

        return(y)

    def imbh_integrated_lum(sigma_v, m_dm, E_th, channel, d, rho_r_sp, r_sp, r_cut, flux_th):
        int_lum = [[] for _ in range(len(m_dm))]
        for i, m in enumerate(m_dm):
            flux_annihi = PrimaryFlux(mDM = m, channel = channel)
            N_gamma = flux_annihi.table_model.integral(E_th, m).to(u.dimensionless_unscaled)
            flux = imbh_flux(N_gamma, sigma_v, m, d, rho_r_sp, r_sp, r_cut)
            for threshold in flux_th:
                int_lum[i].append(len(flux[flux >= threshold]))
        return(int_lum)