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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import numpy as np
import requests
from damspi.utils import nfw_profile, nfw_integral, cored_profile, cored_integral, M_bh_2, parameter_distr_mean, median_error
from astropy import units as u
from astropy.coordinates import SkyCoord, cartesian_to_spherical, spherical_to_cartesian
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.integrate import dblquad
from astropy.wcs import WCS
from astropy.io import fits
import math
from astropy import constants as const
import yaml
from scipy.optimize import curve_fit
import scipy.stats
from scipy.odr import Model, RealData, ODR, Data
from sklearn.neighbors import KernelDensity
import pandas as pd
from joblib import dump

with open("config/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class GalaxyPlotter:
    def __init__(self, sim_name, table_galaxy, table_bh):
        self.sim_name = sim_name
        self.table_galaxy = table_galaxy
        self.table_bh = table_bh
        # shift coorindates system to position of the sun (8.33 kpc), https://iopscience.iop.org/article/10.1088/1475-7516/2011/03/051/pdf
        self.distance_sun = config["Milky_way"]["distance_sun"] # kpc

    @staticmethod
    def download_image(url, save_path):
        # Send a GET request to the URL
        response = requests.get(url, stream = True)
        response.raise_for_status()

        # Extract the filename from the URL
        filename = os.path.basename(url)

        # Construct the full path to save the image
        image_path = os.path.join(save_path, filename)

        # Open a file for writing in binary mode
        with open(image_path, 'wb') as file:
            # Iterate over the response content in chunks
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    def save_gri_images(self):
        """
        Save gri images of galaxies.
        """
        for i in range(len(self.table_galaxy)):
            galaxy_id = self.table_galaxy["main_galaxy_id"].values[i]

            path = f"plots/{self.sim_name}/galaxy_id_{galaxy_id}/images/"
            os.makedirs(path, exist_ok = True)

            url_face = self.table_galaxy["img_face"].values[i].decode()
            url_edge = self.table_galaxy["img_edge"].values[i].decode()
            url_box = self.table_galaxy["img_box"].values[i].decode()

            # download only galaxy images for which images are available
            if url_face != "":
                url_face = url_face.split("'")[-2]
                self.download_image(url_face, path)
            if url_edge != "":
                url_edge = url_edge.split("'")[-2]
                self.download_image(url_edge, path)
            if url_box != "":
                url_box = url_box.split("'")[-2]
                self.download_image(url_box, path)

    def plot_galaxy_distributions(self):
        path = f"plots/{self.sim_name}/galaxy_dist/"
        os.makedirs(path, exist_ok = True)

        # plot galaxy mass distribution
        mass = self.table_galaxy["m"].values
        plt.figure(figsize = config["Figure_size"]["single_column"])
        # plt.title("Total galaxy mass: {0:.1e} $M_{{\odot}}$".format(galaxy_mass_z0))
        plt.hist(mass, bins = 5, color = config["Colors"]["darkblue"])
        plt.xlabel(r"Galaxy mass [$M_{\odot}$]")
        plt.ylabel("Number of main galaxies")
        plt.tight_layout()
        plt.savefig(path + "mass.pdf", dpi = 500)
        plt.close()

        # to be continued

    def update(self, frame, ax):
        self.rotate_z(frame * 5, ax)  # Adjust the rotation speed by modifying the multiplier
        return ax

    @staticmethod
    def rotate_z(angle, ax):
        ax.view_init(elev=30, azim=angle)

    def plot_3d_map(self, coordinates, galaxy_spin, galaxy_id, path, shifted = False, save_animation = False):
        plot_max = np.max(np.abs(coordinates))

        fig = plt.figure(figsize = config["Figure_size"]["quadruple_column"])
        ax = fig.add_subplot(projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, len(coordinates)))
        for coord, color in zip(coordinates, colors):
            ax.scatter(coord[0], coord[1], coord[2], color = "black", s = 40)
        if shifted:
            ax.scatter(0, 0, 0, color = "yellow", s = 40, label = "Sun")
            ax.scatter(-self.distance_sun, 0, 0, color = "black", s = 40, label = "Galaxy centre")
            ax.quiver(-self.distance_sun, 0, 0, *galaxy_spin *  np.max(coordinates), color = "black", label = "Galaxy spin vector")
        else:
            ax.scatter(0, 0, 0, color = "grey", s = 120, label = "Galaxy centre")
            ax.quiver(0, 0, 0, *galaxy_spin * np.max(coordinates), color = "black", label = "Galaxy spin vector")
        text = str(int(galaxy_spin[0] * np.max(coordinates))) + ', ' + str(int(galaxy_spin[1] * np.max(coordinates))) + ', ' + str(int(galaxy_spin[2] * np.max(coordinates)))
        # ax.text(*galaxy_spin * np.max(coordinates), text)
        ax.set_xlabel("x [kpc]", fontsize = 18)
        ax.set_ylabel("y [kpc]", fontsize = 18)
        ax.set_zlabel("z [kpc]", fontsize = 18)
        ax.set_xlim(np.array([-plot_max, plot_max]))
        ax.set_ylim(np.array([-plot_max, plot_max]))
        ax.set_zlim(np.array([-plot_max, plot_max]))
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.legend(loc = "upper left", fontsize = 18)
        plt.savefig(path + ".pdf", dpi = 500)
        if save_animation:
            ani = animation.FuncAnimation(fig, self.update, fargs = [ax], frames=72, interval=50)
            ani.save(path + ".gif", writer='imagemagick')
        plt.close()

    @staticmethod
    def plot_dark_matter_profile(data, para_nfw, para_cored, path):
        data_r, data_rho = data
        rho_0_nfw, r_s_nfw = para_nfw
        rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored = para_cored

        # define radii for plotting
        r = np.logspace(np.log10(np.min(data_r.value)), np.log10(np.max(data_r.value)), 1000) * u.kpc

        # get dark matter density profiles for r
        # nfw 
        rho_nfw = nfw_profile((rho_0_nfw, r_s_nfw), r)
        rho_nfw = (rho_nfw * const.c ** 2).to(u.GeV / u.cm ** 3)
        rho_nfw_data = nfw_profile((rho_0_nfw, r_s_nfw), data_r)
        rho_nfw_data = (rho_nfw_data * const.c ** 2).to(u.GeV / u.cm ** 3)
        # cored
        rho_cored = cored_profile((rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored), r)
        rho_cored = (rho_cored * const.c ** 2).to(u.GeV / u.cm ** 3)
        rho_cored_data = cored_profile((rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored), data_r)
        rho_cored_data = (rho_cored_data * const.c ** 2).to(u.GeV / u.cm ** 3)  

        # plot data and best fit
        fig = plt.figure(figsize=config["Figure_size"]["single_column_legend"])
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3 units for main plot, 1 unit for residual

        ax = plt.subplot(gs[0])  # main plot

        # Original plot
        line_data, = ax.plot(data_r, data_rho, label="data", marker="o", linestyle="None", color=config["Colors"]["black"], markersize=3)
        line_cored, = ax.plot(r, rho_cored, label="Cored", color=config["Colors"]["yellow"])
        line_nfw, = ax.plot(r, rho_nfw, label="NFW", color=config["Colors"]["red"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\rho(r)$ [GeV cm$^{-3}$]")
        ymin, ymax = ax.get_ylim()
        line_r_c = ax.vlines(r_c_cored.value, ymin, ymax, label=r"$r_\mathrm{c}$", color=config["Colors"]["black"], linestyle="dashed")
        ax.set_ylim(ymin, ymax)
        lines = [line_data, line_nfw, line_r_c, line_cored]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, bbox_to_anchor=(0, 1.1, 1, 0.2), loc="center", mode="expand", borderaxespad=0, ncol=2, alignment="center")
        ax.xaxis.set_visible(False)

        # Inset plot
        r_min, r_max = 0.9 * u.kpc, 1.5 * u.kpc #kpc
        ymax = cored_profile((rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored), r_min) * 1.1
        ymin = cored_profile((rho_0_cored, r_s_cored, r_c_cored, gamma_c_cored), r_max)
        ymax = (ymax * const.c ** 2).to(u.GeV / u.cm ** 3)
        ymin = (ymin * const.c ** 2).to(u.GeV / u.cm ** 3)

        axins = inset_axes(ax, width='30%', height='30%', loc='upper right')
        axins.plot(data_r, data_rho, marker="o", linestyle="None", color=config["Colors"]["black"], markersize=3)
        axins.plot(r, rho_cored, color=config["Colors"]["yellow"])
        axins.plot(r, rho_nfw, color=config["Colors"]["red"])
        axins.vlines(r_c_cored.value, ymin.value, ymax.value, label=r"$r_\mathrm{c}$", color=config["Colors"]["black"], linestyle="dashed")
        axins.set_xlim(r_min.value, r_max.value)  
        axins.set_ylim(ymin.value, ymax.value)  
        axins.set_xscale('log')
        axins.set_yscale('log')
        axins.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1.4]))
        axins.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axins.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1., 1.3]))
        axins.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axins.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        # Residual subplot
        ax_res = plt.subplot(gs[1], sharex=ax)
        ax_res.set_ylabel(r'$R$')
        ax_res.set_xscale('log')
        residual_nfw = (data_rho - rho_nfw_data) / rho_nfw_data 
        residual_cored = (data_rho - rho_cored_data) / rho_cored_data 
        ax_res.plot(data_r, residual_cored, color=config["Colors"]["yellow"], marker = "o", markersize = 3, linestyle = "None")
        ax_res.plot(data_r, residual_nfw, color=config["Colors"]["red"], marker = "o", markersize = 3, linestyle = "None")
        ax_res.plot(data_r, np.zeros(len(data_r)), color = config["Colors"]["black"], linestyle = "solid")
        ax_res.set_xlabel(r"$r$ [kpc]")
        ymin, ymax = ax_res.get_ylim()
        ax_res.set_ylim(ymin * 1.2, ymax * 1.2)
        plt.tight_layout()
        plt.savefig(path + "halo_profile_fit.pdf", dpi = 300)

    @staticmethod
    def load_milky_way_satellite_data():
        # load data
        table_mw_satelittes = pd.read_csv("data/mw_satellites/mw_satellites.csv")
        return table_mw_satelittes


    def plot_galaxy_properties_dist(self, path):
        # get main galaxies and satellites
        table_galaxies = self.table_galaxy[self.table_galaxy["subgroup_number"] == 0]
        table_satellites = self.table_galaxy[self.table_galaxy["subgroup_number"] != 0]
        table_satellites_has_stars = table_satellites[table_satellites["m_star"] > 0]

        parameters = ["m", "m200", "m_gas", "m_star", "sfr"]
        log_parameters = ["m", "m_star", "m_gas"]
        names = ["total_mass", "m200", "gas_mass", "star_mass", "sfr"]
        x_labels = [r"$m_\mathrm{tot}$ [$M_{\odot}$]", r"$m_{200}$ [$M_{\odot}$]", r"$m_{\mathrm{gas}}$ [$M_{\odot}$]", r"$m_\mathrm{star}$ [$M_{\odot}$]", "SFR [$M_{\odot}$ / yr]"]

        for parameter, name, x_label in zip(parameters, names, x_labels):
            # get masses for main galaxies
            mass_galaxies = table_galaxies[parameter].values
            median_mass_galaxies = np.median(mass_galaxies)

            # create a histogram with logarithmic bins
            if parameter in ["sfr"]:
                bins_galaxies = np.linspace(np.min(mass_galaxies), np.max(mass_galaxies), config["Plots"]["number_bins"])
            else:
                bins_galaxies = np.logspace(np.log10(np.min(mass_galaxies)), np.log10(np.max(mass_galaxies)), config["Plots"]["number_bins"])

            # plot galaxy mass distribution
            plt.figure(figsize = config["Figure_size"]["single_column"])
            plt.hist(mass_galaxies, bins = bins_galaxies, color = config["Colors"]["darkblue"])
            plt.xlabel(x_label)
            plt.ylabel("Number of main galaxies")
            if parameter not in ["sfr"]:
                plt.xscale("log")
            plt.yscale("log")
            ymin, ymax = plt.ylim()
            plt.vlines(median_mass_galaxies, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = r"$\tilde{{\mu}}$ = " + f"{median_mass_galaxies:.2e}")
            plt.ylim(ymin, ymax)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path + f"main_galaxy_{name}_dist.pdf", dpi = 300)
            plt.close()

            if parameter != "m200":
                # get masses for satellites
                mass_satellites = table_satellites[parameter].values
                mass_satellites_has_stars = table_satellites_has_stars[parameter].values
                median_mass_satellites = np.median(mass_satellites)
                median_mass_satellites_has_stars = np.median(mass_satellites_has_stars)

                # create a histogram with logarithmic bins
                if parameter in log_parameters and np.min(mass_satellites) != 0:
                    # log bins
                    bins_satellites = np.logspace(np.log10(np.min(mass_satellites)), np.log10(np.max(mass_satellites)), config["Plots"]["number_bins"])
                    error_x_position_satellites = np.sqrt(bins_satellites[1:] * bins_satellites[:-1])
                else:
                    # linear bins
                    bins_satellites = np.linspace(np.min(mass_satellites), np.max(mass_satellites), config["Plots"]["number_bins"])
                    error_x_position_satellites = (bins_satellites[1:] + bins_satellites[:-1]) / 2
                    
                if parameter in log_parameters and np.min(mass_satellites_has_stars) != 0:
                    bins_satellites_has_stars = np.logspace(np.log10(np.min(mass_satellites_has_stars)), np.log10(np.max(mass_satellites_has_stars)), config["Plots"]["number_bins"] // 2)
                    error_x_position_satellites_has_stars = np.sqrt(bins_satellites_has_stars[1:] * bins_satellites_has_stars[:-1])
                else:
                    bins_satellites_has_stars = np.linspace(np.min(mass_satellites_has_stars), np.max(mass_satellites_has_stars), config["Plots"]["number_bins"] // 2)
                    error_x_position_satellites_has_stars = (bins_satellites_has_stars[1:] + bins_satellites_has_stars[:-1]) / 2

                bins_satellites_centre = (bins_satellites[1:] + bins_satellites[:-1]) / 2
                bins_satellites_width = bins_satellites[1:] - bins_satellites[:-1]

                bins_satellites_has_stars_centre = (bins_satellites_has_stars[1:] + bins_satellites_has_stars[:-1]) / 2
                bins_satellites_has_stars_width = bins_satellites_has_stars[1:] - bins_satellites_has_stars[:-1]

                hist_satellites_mean, hist_satellites_mean_error = parameter_distr_mean(table_satellites, parameter, bins_satellites, group_identifier = "group_number")
                hist_satellites_has_stars_mean, hist_satellites_has_stars_mean_error = parameter_distr_mean(table_satellites_has_stars, parameter, bins_satellites_has_stars, group_identifier = "group_number")

                plt.figure(figsize = config["Figure_size"]["single_column"])
                # plt.hist(mass_satellites, bins = bins_satellites, color = config["Colors"]["darkblue"])
                plt.bar(bins_satellites_centre, hist_satellites_mean, width = bins_satellites_width, color = config["Colors"]["darkblue"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"])
                plt.errorbar(error_x_position_satellites, hist_satellites_mean, yerr = hist_satellites_mean_error, color = config["Colors"]["lightblue"], linestyle = "")
                plt.xlabel(x_label)
                plt.ylabel("Number of satellites")
                if parameter in log_parameters and np.min(mass_satellites) != 0:
                    plt.xscale("log")
                plt.yscale("log")
                ymin, ymax = plt.ylim()
                plt.vlines(median_mass_satellites, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = r"$\tilde{{\mu}}$ = " + f"{median_mass_satellites:.2e}")
                plt.ylim(ymin, ymax)
                plt.legend()
                plt.tight_layout()
                plt.savefig(path + f"satellites_{name}_dist.pdf", dpi = 300)
                plt.close()

                if parameter == "m_star":
                    table_mw_satelittes = self.load_milky_way_satellite_data()
                    mass_satellites_mw = table_mw_satelittes["M_star (1e6 Msun)"].values
                    mass_satellites_mw = mass_satellites_mw * 1e6 
                    # mass_satellites_mw has lower masses than mass_satellites. Add logaithmic bins to bins_satellites_has_stars to cover full range of mass_satellites_mw 
                    lower_mass_bins = np.logspace(np.log10(np.min(mass_satellites_mw)), np.log10(np.min(mass_satellites_has_stars)), config["Plots"]["number_bins"] // 2)
                    bins_satellites_mw = np.concatenate((lower_mass_bins, bins_satellites_has_stars))
                    bins_satellites_mw_centre = (bins_satellites_mw[1:] + bins_satellites_mw[:-1]) / 2
                    bins_satellites_mw_width = bins_satellites_mw[1:] - bins_satellites_mw[:-1]
 
                    hist_satellites_mw, _ = np.histogram(mass_satellites_mw, bins = bins_satellites_mw)
                    median_mass_satellites_mw = np.median(mass_satellites_mw[mass_satellites_mw > np.min(mass_satellites_has_stars)])

                plt.figure(figsize = config["Figure_size"]["single_column"])
                # plt.hist(mass_satellites_has_stars, bins = bins_satellites_has_stars, color = config["Colors"]["darkblue"])
                plt.bar(bins_satellites_has_stars_centre, hist_satellites_has_stars_mean, width = bins_satellites_has_stars_width, color = config["Colors"]["darkblue"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"], alpha = 0.5)
                plt.errorbar(error_x_position_satellites_has_stars, hist_satellites_has_stars_mean, yerr = hist_satellites_has_stars_mean_error, color = config["Colors"]["lightblue"], linestyle = "")
                if parameter == "m_star":
                    plt.bar(bins_satellites_mw_centre, hist_satellites_mw, width = bins_satellites_mw_width, color = config["Colors"]["yellow"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"], alpha = 0.5)
                plt.xlabel(x_label)
                plt.ylabel("Number of satellites with stars")
                if parameter in log_parameters and np.min(mass_satellites_has_stars) != 0:
                    plt.xscale("log")
                plt.yscale("log")
                ymin, ymax = plt.ylim()
                plt.vlines(median_mass_satellites_has_stars, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = r"$\tilde{{\mu}}_{{EAGLE}}$ = " + f"{median_mass_satellites_has_stars:.2e}")
                if parameter == "m_star":
                    plt.vlines(median_mass_satellites_mw, ymin = ymin, ymax = ymax, color = config["Colors"]["yellow"], linestyle = "solid", label = r"$\tilde{{\mu}}_{{MW}}$ = " + f"{median_mass_satellites_mw:.2e}")
                plt.ylim(ymin, ymax)
                plt.legend(bbox_to_anchor = (0, 1.1, 1, 0.2), loc = "center", mode = "expand", borderaxespad = 0, ncol = 1, alignment = "center")
                plt.tight_layout()
                plt.savefig(path + f"satellites_{name}_has_stars_dist.pdf", dpi = 300)
                plt.close()

    def plot_morphology_dist(self, path):
        table_galaxies = self.table_galaxy[self.table_galaxy["subgroup_number"] == 0]
        parameters = ["fdisk", "fbulge", "fihl"]
        parameter_labels = ["$f_\mathrm{disk}$", "$f_\mathrm{bulge}$", "$f_\mathrm{IHL}$"]

        for parameter, parameter_label in zip(parameters, parameter_labels):
            # plot parameter distribution for main galaxies
            parameter_galaxies = table_galaxies[parameter].values
            median_parameter_galaxies, median_parameter_galaxies_lower_error, median_parameter_galaxies_upper_error = median_error(parameter_galaxies)
            median_parameter_galaxies_lower_percentile = median_parameter_galaxies - median_parameter_galaxies_lower_error
            median_parameter_galaxies_upper_percentile = median_parameter_galaxies + median_parameter_galaxies_upper_error

            plt.figure(figsize = config["Figure_size"]["single_column"])
            plt.hist(parameter_galaxies, bins = config["Plots"]["number_bins"], color = config["Colors"]["darkblue"])
            plt.xlabel(parameter_label)
            plt.ylabel("Number of main galaxies")
            ymin, ymax = plt.ylim()
            plt.vlines(median_parameter_galaxies, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = r"$\tilde{{\mu}}$ = " + f"${median_parameter_galaxies:.2f}^{{+{median_parameter_galaxies_upper_error:.2f}}}_{{-{median_parameter_galaxies_lower_error:.2f}}}$")
            plt.axvspan(median_parameter_galaxies_lower_percentile, median_parameter_galaxies_upper_percentile, alpha = 0.25, facecolor = config["Colors"]["red"], edgecolor = "None")
            plt.ylim(ymin, ymax)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path + f"main_galaxy_{parameter}_dist.pdf", dpi = 300)
            plt.close()
 
    def plot_satellite_types(self, path):
        # get satellites
        table_satellites = self.table_galaxy[self.table_galaxy["subgroup_number"] != 0]

        n_satellites = len(table_satellites)
        n_satellites_no_gas_no_star = len(table_satellites[(table_satellites["m_gas"] == 0) & (table_satellites["m_star"] == 0)])
        n_satellites_has_gas = len(table_satellites[table_satellites["m_gas"] > 0])
        n_satellites_has_star = len(table_satellites[table_satellites["m_star"] > 0])
        n_satellites_has_gas_has_star = len(table_satellites[(table_satellites["m_gas"] > 0) & (table_satellites["m_star"] > 0)])

        # plot bar plot
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.bar(["All", "Has stars", "Has gas", "Has stars $\&$ gas", "No stars $\&$ no gas"], [n_satellites, n_satellites_has_star, n_satellites_has_gas, n_satellites_has_gas_has_star, n_satellites_no_gas_no_star], color = config["Colors"]["darkblue"])
        plt.ylabel("Number of satellites")
        # rotate the x-axis labels
        plt.xticks(rotation = 90)
        plt.tight_layout()
        plt.savefig(path + "satellite_types.pdf", dpi = 300)
        plt.close()

    def plot_n_bh_satellite_types(self, path):
        # get satellites
        table_satellites = self.table_galaxy[self.table_galaxy["subgroup_number"] != 0]
        bh_table_satellites = self.table_bh[self.table_bh["satellite"] == True]

        # get percentage of satellites with BHs for all satellites
        galaxy_ids_satellites_with_bh = np.unique(bh_table_satellites["host_galaxy_id"])
        n_satellites = len(table_satellites)
        n_satellites_with_bh = len(galaxy_ids_satellites_with_bh)
        n_satellites_no_bh = n_satellites - n_satellites_with_bh
        percentage_satellites_with_bh = n_satellites_with_bh / n_satellites * 100
    
        # get percentage of satellites with BHs for satellites with stars
        galaxy_ids_satellites_with_stars = np.unique(table_satellites[table_satellites["m_star"] > 0]["galaxy_id"])
        galaxy_ids_satellites_with_bh_and_stars = np.intersect1d(galaxy_ids_satellites_with_bh, galaxy_ids_satellites_with_stars)
        n_satellites_with_stars = len(galaxy_ids_satellites_with_stars)
        n_satellites_with_bh_and_stars = len(galaxy_ids_satellites_with_bh_and_stars)
        percentage_satellites_with_bh_and_stars = n_satellites_with_bh_and_stars / n_satellites_with_stars * 100

        # get percentage of satellites with BHs for satellites with stars and gas
        galaxy_ids_satellites_with_stars_gas = np.unique(table_satellites[(table_satellites["m_star"] > 0) & (table_satellites["m_gas"] > 0)]["galaxy_id"])
        galaxy_ids_satellites_with_bh_and_stars_gas = np.intersect1d(galaxy_ids_satellites_with_bh, galaxy_ids_satellites_with_stars_gas)
        n_satellites_with_stars_gas = len(galaxy_ids_satellites_with_stars_gas)
        n_satellites_with_bh_and_stars_gas = len(galaxy_ids_satellites_with_bh_and_stars_gas)
        percentage_satellites_with_bh_and_stars_gas = n_satellites_with_bh_and_stars_gas / n_satellites_with_stars_gas * 100

        # create three bar subplots side by side. Plot number of satellites with BHs for all satellites, satellites with stars and satellites with stars and gas as bars with percentages as text in the plots
        fig, axs = plt.subplots(1, 3, figsize = config["Figure_size"]["double_column_squeezed"])
        axs[0].set_title("All satellites")
        axs[0].bar(["All", "Contain BH"], [n_satellites, n_satellites_with_bh], color = config["Colors"]["darkblue"])
        axs[0].text(1, n_satellites_with_bh, f"{percentage_satellites_with_bh:.1f} $\%$", ha = "center", va = "bottom")
        axs[0].set_ylabel("Number of satellites")
        axs[1].set_title("Satellites with stars")
        axs[1].bar(["All", "Contain BH"], [n_satellites_with_stars, n_satellites_with_bh_and_stars], color = config["Colors"]["darkblue"])
        axs[1].text(1, n_satellites_with_bh_and_stars, f"{percentage_satellites_with_bh_and_stars:.1f} $\%$", ha = "center", va = "bottom")
        axs[2].set_title("Satellites with stars and gas")
        axs[2].bar(["All", "Contain BH"], [n_satellites_with_stars_gas, n_satellites_with_bh_and_stars_gas], color = config["Colors"]["darkblue"])
        axs[2].text(1, n_satellites_with_bh_and_stars_gas, f"{percentage_satellites_with_bh_and_stars_gas:.1f} $\%$", ha = "center", va = "bottom")
        plt.tight_layout()
        plt.savefig(path + "n_bh_satellite_types.pdf", dpi = 300)
        plt.close()

    def plot_satellite_number_dist(self, path):
        # get main galaxies and their corresponding number of satellites
        table_galaxies = self.table_galaxy[self.table_galaxy["subgroup_number"] == 0]
        table_satellites = self.table_galaxy[self.table_galaxy["subgroup_number"] != 0]
        table_satellites_has_stars = table_satellites[table_satellites["m_star"] > 0]
        
        # get number of satellites for each main galaxy
        n_satellites = table_galaxies["n_satellites"].values
        n_satellites_median = np.median(n_satellites)

        # determine the number distribution of satellites with stars
        # determine how many satellite galaxies with stars each host galaxy has
        grouped_satellites = table_satellites_has_stars.groupby('group_number')['subgroup_number'].count().reset_index()
        
        # add the number of satellite galaxies to the host galaxies
        table_galaxies_satellites_has_stars = table_galaxies.merge(grouped_satellites, on='group_number', suffixes=('', '_count'))

        # rename the column
        table_galaxies_satellites_has_stars = table_galaxies_satellites_has_stars.rename(columns = {'subgroup_number_count': 'n_satellites_has_stars'})

        # get number of satellites with stars for each main galaxy
        n_satellites_has_stars = table_galaxies_satellites_has_stars["n_satellites_has_stars"].values
        # calculate error on median based on 
        n_satellites_has_stars_median, n_satellites_has_stars_median_lower_err, n_satellites_has_stars_median_upper_err = median_error(n_satellites_has_stars)
        n_satellites_has_stars_median_lower_percentile = n_satellites_has_stars_median - n_satellites_has_stars_median_lower_err
        n_satellites_has_stars_median_upper_percentile = n_satellites_has_stars_median + n_satellites_has_stars_median_upper_err

        # plot satellite number distribution
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.hist(n_satellites, bins = 10, color = config["Colors"]["darkblue"])
        ymin, ymax = plt.ylim()
        plt.vlines(n_satellites_median, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = r"$\tilde{{\mu}}$ = " + f"{n_satellites_median:.2f}")
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.xlabel("Number of satellites")
        plt.ylabel("Number of main galaxies")
        plt.tight_layout()
        plt.savefig(path + "n_satellites_dist.pdf", dpi = 300)
        plt.close()  

        # load mw satellites data
        table_mw_satellites = self.load_milky_way_satellite_data()
        n_satellites_mw = len(table_mw_satellites[table_mw_satellites["M_star (1e6 Msun)"] > 1])

        # plot satellite number distribution for satellites with stars
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.hist(n_satellites_has_stars, bins = 10, color = config["Colors"]["darkblue"])
        ymin, ymax = plt.ylim()
        plt.vlines(n_satellites_has_stars_median, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = r"$\tilde{{\mu}}$ = " + f"${n_satellites_has_stars_median:.0f}^{{+{n_satellites_has_stars_median_upper_err:.0f}}}_{{-{n_satellites_has_stars_median_lower_err:.0f}}}$")
        plt.axvspan(n_satellites_has_stars_median_lower_percentile, n_satellites_has_stars_median_upper_percentile, alpha = 0.25, facecolor = config["Colors"]["red"], edgecolor = "None")
        plt.vlines(n_satellites_mw, ymin = ymin, ymax = ymax, color = config["Colors"]["yellow"], linestyle = "solid", label = "$N_\mathrm{MW}$ = " + f"{n_satellites_mw}")
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.xlabel("Number of satellites with stars")
        plt.ylabel("Number of main galaxies")
        plt.tight_layout()
        plt.savefig(path + "n_satellites_has_stars_dist.pdf", dpi = 300)
        plt.close()

    def plot_scatter_bh_galaxy_morphology(self, path):
        table_bh_main_galaxies = self.table_bh[self.table_bh["satellite"] == False]
        table_main_galaxies = self.table_galaxy[self.table_galaxy["subgroup_number"] == 0]
        main_galaxy_ids = np.unique(table_bh_main_galaxies["main_galaxy_id"].values)
        galaxy_ids = np.unique(self.table_bh["main_galaxy_id"].values)

        parameters = ["fdisk", "fbulge", "fihl"]
        x_labels = ["$f_\mathrm{disk}$", "$f_\mathrm{bulge}$", "$f_\mathrm{ihl}$"]

        for parameter, x_label in zip(parameters, x_labels):
            # extract the number of BHs and the parameter values for the main galaxies (considering BHs in the main galaxy and the satellites)
            n_bh = []
            parameter_values = []
            for galaxy_id in galaxy_ids:
                table_bh_galaxy = table_bh_main_galaxies[table_bh_main_galaxies["main_galaxy_id"] == galaxy_id]
                table_main_galaxy = table_main_galaxies[table_main_galaxies["galaxy_id"] == galaxy_id]
                n_bh_galaxy = len(table_bh_galaxy)
                parameter_value_galaxy = table_main_galaxy[parameter].values[0]
                n_bh.append(n_bh_galaxy)
                parameter_values.append(parameter_value_galaxy)

            # plot scatter plot
            plt.figure(figsize = config["Figure_size"]["single_column"])
            plt.scatter(parameter_values, n_bh, color = config["Colors"]["darkblue"], marker = "o", s = config["Plots"]["markersize"])
            plt.xlabel(x_label)
            plt.ylabel("$N_\mathrm{BH}$ (in main galaxy)")
            plt.tight_layout()
            plt.savefig(path + f"scatter_bh_{parameter}_main_galaxy.pdf", dpi = 300)
            plt.close()

class BlackHolePlotter:
    def __init__(self, sim_name, table_bh):
        self.sim_name = sim_name
        self.table_bh = table_bh

    def plot_bh_dist_galaxy(self, path):
        os.makedirs(path, exist_ok = True)

        parameters = ["m", "z_f", "d_GC", "lat_GC", "long_GC", "d_Sun", "lat_Sun", "long_Sun"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]"]
        filenames = ["mass", "redshift", "distance_GC", "latitude_GC", "longitude_GC", "distance_Sun", "latitude_Sun", "longitude_Sun"]
        
        for parameter, x_label, filename in zip(parameters, x_labels, filenames):
            # plot galaxy mass distribution
            data = self.table_bh[parameter].values
            plt.figure(figsize = config["Figure_size"]["single_column"])
            if parameter == "m":
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 5)
                plt.xscale("log")
            else:
                bins = 5
            plt.hist(data, bins = bins, color = config["Colors"]["darkblue"])
            plt.xlabel(x_label)
            plt.ylabel("$N_\mathrm{BH}$")
            plt.tight_layout()
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

    @staticmethod
    def plot_nfw(r, rho, rho_0, r_s, galaxy_mass, z, path):
        lablel_fit = f"Fit\n$\\rho_0$ = {rho_0.value:.2e} M$_{{\\odot}}$ / kpc$^3$\n$r_s$ = {r_s.value:.2f} kpc"
        plt.figure(figsize = config["Figure_size"]["double_column"])
        plt.title(f"M$_{{halo}}$ = {galaxy_mass.value:.2e} M$_{{\\odot}}$, z = {z:.2f}")
        plt.plot(r, rho, color = config["Colors"]["black"], label = "Data")
        plt.plot(r, nfw_profile((rho_0, r_s), r), color = config["Colors"]["red"], linestyle = "dashed", label = lablel_fit)
        plt.xlabel(r"$r$ [kpc]")
        plt.ylabel(r"$\rho$ [M$_{\odot}$ / kpc$^3$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.legend()
        plt.savefig(path + "nfw.pdf", dpi = 300)
        plt.close()

    @staticmethod
    def plot_cored(r, rho, rho_0, r_s, r_c, gamma_c, galaxy_mass, z, path):
        # lablel_fit = f"Fit\n$\\rho_0$ = {rho_0.value:.2e} M$_{{\\odot}}$ / kpc$^3$\n$r_s$ = {r_s.value:.2f} kpc"
        lablel_fit = f"Fit\n$\\rho_0$ = {rho_0.value:.2e} M$_{{\\odot}}$ / kpc$^3$\n$r_s$ = {r_s.value:.2f} kpc\n$r_c$ = {r_c.value:.2f} kpc\n$\\gamma_c$ = {gamma_c:.2f}"
        plt.figure(figsize = config["Figure_size"]["double_column"])
        plt.title(f"M$_{{halo}}$ = {galaxy_mass.value:.2e} M$_{{\\odot}}$, z = {z:.2f}")
        plt.plot(r, rho, color = config["Colors"]["black"], label = "Data")
        plt.plot(r, cored_profile((rho_0, r_s, r_c, gamma_c), r), color = config["Colors"]["red"], linestyle = "dashed", label = lablel_fit)
        plt.xlabel(r"$r$ [kpc]")
        plt.ylabel(r"$\rho$ [M$_{\odot}$ / kpc$^3$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.legend()
        plt.savefig(path + "cored.pdf", dpi = 300)
        plt.close()

    @staticmethod
    def plot_radius_gravitational_influence_nfw(r_h, rho_0, r_s, M_bh, galaxy_mass, z, path):
        r_min, r_max = 0.1e-3 * u.kpc, 200e-3 * u.kpc #kpc
        r = np.linspace(r_min, r_max, 1000)
        plt.figure()
        plt.title(f"M$_{{halo}}$ = {galaxy_mass:.2e} M$_{{\odot}}$, z = {z:.2f}")
        plt.plot(r, nfw_integral(r, rho_0, r_s), color = config["Colors"]["black"], label = "NFW integral")
        plt.hlines(M_bh_2(M_bh).value, xmin = np.min(r).value, xmax = np.max(r).value, linestyle = "dotted", color = config["Colors"]["black"], label = f"$2 \cdot M_{{BH}}$ = {M_bh_2(M_bh).value:.2e} M$_{{\\odot}}$")
        plt.vlines(r_h.value, ymin = np.min(nfw_integral(r, rho_0, r_s)).value, ymax = np.max(nfw_integral(r, rho_0, r_s)).value, linestyle = "dashed", color = config["Colors"]["red"], label = f"$r_h$ = {r_h:.2e}\n$r_{{sp}}$ = {0.2 * r_h:.2e}")
        plt.xlabel(r"$r$ [kpc]")
        plt.ylabel(r"$Y$ [M$_{\odot}$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "r_h_nfw.pdf", dpi = 300)
        plt.close()

    @staticmethod
    def plot_radius_gravitational_influence_cored(r_h, rho_0, r_s, r_c, gamma_c, M_bh, galaxy_mass, z, path):
        r_min, r_max = 0.1e-3 * u.kpc, 200e-3 * u.kpc #kpc
        r = np.linspace(r_min, r_max, 1000)
        plt.figure()
        plt.title(f"M$_{{halo}}$ = {galaxy_mass:.2e} M$_{{\odot}}$, z = {z:.2f}")
        plt.plot(r, cored_integral(r, rho_0, r_s, r_c, gamma_c), color = config["Colors"]["black"], label = "Cored integral")
        plt.hlines(M_bh_2(M_bh).value, xmin = np.min(r).value, xmax = np.max(r).value, linestyle = "dotted", color = config["Colors"]["black"], label = f"$2 \cdot M_{{BH}}$ = {M_bh_2(M_bh).value:.2e} M$_{{\\odot}}$")
        plt.vlines(r_h.value, ymin = np.min(cored_integral(r, rho_0, r_s, r_c, gamma_c)).value, ymax = np.max(cored_integral(r, rho_0, r_s, r_c, gamma_c)).value, linestyle = "dashed", color = config["Colors"]["red"], label = f"$r_h$ = {r_h:.2e}\n$r_{{sp}}$ = {0.2 * r_h:.2e}")
        plt.xlabel(r"$r$ [kpc]")
        plt.ylabel(r"$Y$ [M$_{\odot}$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "r_h_cored.pdf", dpi = 300)
        plt.close()

    def plot_dist_total(self, path):
        parameters = ["m", "z_f", "d_GC", "lat_GC", "long_GC", "d_Sun", "lat_Sun", "long_Sun", "r_sp", "rho(r_sp)"]
        log_parameters = ["m", "r_sp", "rho(r_sp)"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]", "r_sp", r"$\rho(r_\mathrm{sp})$ [GeV/cm$^3$]"]
        filenames = ["mass", "redshift", "distance_gc", "latitude_GC", "longitude_GC", "distance_sun", "latitude_Sun", "longitude_Sun", "r_sp", "rho_sp"]
        
        for parameter, x_label, filename in zip(parameters, x_labels, filenames):
            data = self.table_bh[parameter].values
            plt.figure(figsize = config["Figure_size"]["single_column"])
            if parameter in log_parameters:
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 9)
                plt.xscale("log")
            else:
                bins = 9
            plt.hist(data, bins = bins, color = config["Colors"]["darkblue"])
            plt.xlabel(x_label)
            plt.ylabel("$N_\mathrm{BH}$")
            plt.tight_layout()
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

            if parameter == "lat_GC" or parameter == "lat_Sun":
                data = np.cos(data)
                if parameter == "lat_GC":
                    x_label = r"$\cos(b_\mathrm{GC})$"
                    filename = "latitude_GC_cos"
                if parameter == "lat_Sun":
                    x_label = r"$\cos(b_\mathrm{Sun})$"
                    filename = "latitude_Sun_cos"
                plt.figure(figsize = config["Figure_size"]["single_column"])
                bins = 9
                plt.hist(data, bins = bins, color = config["Colors"]["darkblue"])
                plt.xlabel(x_label)
                plt.ylabel("$N_\mathrm{BH}$")
                plt.tight_layout()
                plt.savefig(path + f"{filename}.pdf", dpi = 500)
                plt.close()

    def plot_2d_map(self, lat, long, path):
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')

        fig = plt.figure(figsize  = config["Figure_size"]["single_column"])
        ax = fig.add_subplot(111, projection='aitoff')
        ax.grid(True, alpha = 0.5)
        ax.scatter(coords.l.wrap_at('180d').radian, coords.b.radian, color = config["Colors"]["darkblue"], marker = 'o', s = config["Plots"]["markersize"] / 100)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()

    @staticmethod
    def random_rotate_z(phi):
        # Random angle between 0 and 2*pi for rotation
        random_angle = np.random.rand() * 2 * np.pi

        # Rotate phi by the random angle
        phi_rotated = phi + random_angle

        # Ensuring phi_rotated is within the range [0, 2*pi]
        phi_rotated = np.mod(phi_rotated, 2 * np.pi)

        return(phi_rotated)
    
    def random_upsampling(self, d_gc, lat_gc, long_gc, upsampling_factor):
        # add original coordinates to the upsampled coordinates
        d_gc_upsampled = d_gc
        lat_gc_upsampled = lat_gc
        long_gc_upsampled = long_gc

        for _ in range(upsampling_factor):
            # rotate the coordinates
            long_gc_rotated = self.random_rotate_z(long_gc)

            # add the rotated coordinates to the upsampled coordinates
            lat_gc_upsampled = np.append(lat_gc_upsampled, lat_gc)
            long_gc_upsampled = np.append(long_gc_upsampled, long_gc_rotated)
            d_gc_upsampled = np.append(d_gc_upsampled, d_gc)
        
        # convert to cartesian coordinates
        x_gc_upsampled, y_gc_upsampled, z_gc_upsampled = spherical_to_cartesian(d_gc_upsampled, lat_gc_upsampled, long_gc_upsampled)

        # shift the coordinates to the sun
        x_sun_upsampled, y_sun_upsampled, z_sun_upsampled = self.shift_to_sun(x_gc_upsampled, y_gc_upsampled, z_gc_upsampled)

        # convert back to spherical coordinates [rad]
        d_sun_upsampled, lat_sun_upsampled, long_sun_upsampled = cartesian_to_spherical(x_sun_upsampled, y_sun_upsampled, z_sun_upsampled)

        return(d_sun_upsampled.value, lat_sun_upsampled.value, long_sun_upsampled.value)
    
    @staticmethod
    def shift_to_sun(x, y, z):
        # shift the coordinates to the sun
        x += config["Milky_way"]["distance_sun"]
        return(x, y, z)

    def plot_2d_map_contours(self, lat, long, upsampling_factor, path, path_kde, wihtin_region = False):
        # Extract IMBH coordinates
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')
        coord_stacked = np.vstack([lat, coords.l.wrap_at('180d').radian]).T
        coord_stacked_contours = np.vstack([coords.l.wrap_at('180d').radian, lat])

        # choose subsample of coordinates for scatter plot
        # Calculate number of coordinates to be plotted
        n_coord_scatter = len(coords) // (upsampling_factor + 1)
        # Randomly select half of the indices
        scatter_indices = np.random.choice(len(coords), n_coord_scatter, replace = False)
        # Extract the scatter coordinates
        scatter_coords = coords[scatter_indices]

        # define grid for kernel density estimation
        num_grid = 250 # 250
        lat_min, lat_max = -np.pi / 2, np.pi / 2
        long_min, long_max = -np.pi, np.pi
        dlat = (lat_max - lat_min) / num_grid
        dlon = (long_max - long_min) / num_grid
        lat_grid = np.linspace(lat_min, lat_max, num_grid)
        long_grid = np.linspace(long_min, long_max, num_grid)
        Lat_grid, Long_grid = np.meshgrid(lat_grid, long_grid)
        coord_grid = np.vstack([Lat_grid.flatten(), Long_grid.flatten()]).T

        # define grid for contours
        coord_grid_contours = np.vstack([Long_grid.flatten(), Lat_grid.flatten()])

        print(f"Calculating kernel density estimation for {len(coords)} IMBHs (may take a while)...")
        # calculate kernel density estimation on IMBH coordinates
        kde = KernelDensity(bandwidth="scott", metric='haversine')
        kde.fit(coord_stacked)

        # save kde
        dump(kde, path_kde + "kde_model.joblib")

        # evaluate kde on grid coordinates
        pdf = np.exp(kde.score_samples(coord_grid))
        pdf = pdf.reshape(Lat_grid.shape)

        # # save the pdf and grid coordinates in a single csv file with pandas
        # columns = ["Lat", "Long", "pdf"]
        # df = pd.DataFrame(columns = columns)
        # df["Lat"] = Lat_grid.flatten()
        # df["Long"] = Long_grid.flatten()
        # df["pdf"] = pdf.flatten()
        # df.to_csv(path + "pdf.csv", index = False)

        # get HESS galactic plane survey values and convert them to radians
        num_grid_survey = 100
        hess_lat_min, hess_lat_max = config["HESS"]["gps_lat_min"], config["HESS"]["gps_lat_max"]
        hess_long_min, hess_long_max = config["HESS"]["gps_long_min"], config["HESS"]["gps_long_max"]
        hess_lat_min, hess_lat_max = hess_lat_min * np.pi / 180, hess_lat_max * np.pi / 180 # rad
        hess_long_min, hess_long_max = hess_long_min * np.pi / 180, hess_long_max * np.pi / 180 # rad

        # CTA galactic plane survey values and convert them to radians
        cta_lat_min, cta_lat_max = config["CTA"]["gps_lat_min"], config["CTA"]["gps_lat_max"]
        cta_long_min, cta_long_max = config["CTA"]["gps_long_min"], config["CTA"]["gps_long_max"]
        cta_lat_min, cta_lat_max = cta_lat_min * np.pi / 180, cta_lat_max * np.pi / 180 # rad
        cta_long_min, cta_long_max = cta_long_min * np.pi / 180, cta_long_max * np.pi / 180 # rad

        if wihtin_region:
            # get expected number of IMBHs within HESS and CTA galactic plane surveys
            self.expected_number_within_region(
                coord_stacked, 
                hess_lat_min, 
                hess_lat_max, 
                hess_long_min, 
                hess_long_max, 
                kde, 
                num_grid_survey, 
                "HESS"
                )

            self.expected_number_within_region(
                coord_stacked,
                cta_lat_min,
                cta_lat_max,
                cta_long_min,
                cta_long_max,
                kde,
                num_grid_survey,
                "CTA"
                )

        # calculate cdf
        # define desired percentage contours to be calculated
        percentages_desired = [10, 20, 30, 40, 50]
        # initialize variables that will be filled in the loop later
        percentages_diff = np.full(len(percentages_desired), np.inf)
        percentages_cont = np.zeros(len(percentages_desired))
        percentages_cont_collections = np.empty(len(percentages_desired), dtype = object)

        # calculate the desired percentage contours with matplotlib
        fig_cdf = plt.figure(figsize = config["Figure_size"]["double_column"])
        ax_cdf = fig_cdf.add_subplot(111, projection="aitoff")
        # extract the PDF contours
        levels = np.linspace(pdf.min(), pdf.max(), 500) #150
        cont = ax_cdf.contour(Long_grid, Lat_grid, pdf, levels = levels)

        def is_path_closed(path):
            """Check if a Matplotlib path is closed."""
            return path.codes is not None and path.codes[-1] == mpl.path.Path.CLOSEPOLY

        # Get the contour collections
        cont_collections = cont.collections

        print("Calculating contours...")
        # Loop through each contour collection
        for collection in cont_collections:
            # Get the contour path
            cont_path = collection.get_paths()
            if cont_path: # check if the contour is not empty
                cont_path = cont_path[0]
                
                if is_path_closed(cont_path) == True:
                    mask = cont_path.contains_points(coord_grid_contours.T)
                    coord_grid_contours_masked = coord_grid_contours.T[mask]
                    lat_contours_masked = coord_grid_contours_masked[:, 1]
                    area_element = np.cos(lat_contours_masked) * dlat * dlon

                    pdf_masked = pdf.flatten()[mask]
                    pdf_integrated = np.sum(pdf_masked * area_element.ravel())
                    cont_percentage = pdf_integrated * 100

                    # Check if the current contour is closer to the desired percentage than the previous one and update the variables
                    for i, percentage in enumerate(percentages_desired):
                        if np.abs(percentage - cont_percentage) < percentages_diff[i]:
                            percentages_cont[i] = cont_percentage
                            percentages_cont_collections[i] = collection
                            percentages_diff[i] = np.abs(percentage - cont_percentage)

        plt.close()

        # # check if countors percentages agree with number of IMBHs
        # for cont_collection, percentage in zip(percentages_cont_collections, percentages_cont):
        #     cont_collections_path = cont_collection.get_paths()[0]
        #     mask = cont_collections_path.contains_points(coord_stacked_contours.T)
        #     print("Contours percentage:", percentage, "%")
        #     print("Number of IMBHs within contour:", np.sum(mask) / len(mask) * 100, "%")

        # Plot in Aitoff projection
        fig = plt.figure(figsize = config["Figure_size"]["double_column"])
        ax = fig.add_subplot(111, projection="aitoff")
        ax.grid(True, alpha = 0.5)
        # plot the PDF
        im = ax.pcolormesh(Long_grid, Lat_grid, pdf, cmap = LinearSegmentedColormap.from_list("", config["Colors"]["cmap_2d_map"]), edgecolors = "face", linewidth = 0, rasterized=True)

        # plot the contours
        x_max_previous = 0
        for i, collection in enumerate(percentages_cont_collections):
            cont_path = collection.get_paths()[0]
            x, y = zip(*cont_path.vertices)
            x_max, y_min = np.max(x), np.min(y)
            ax.plot(x, y, color = "white", alpha = 0.6, linestyle = "dashed")
            ax.text(
                x_max_previous + np.abs(x_max - x_max_previous) / 2, 
                - 5 * (2*np.pi/360), 
                r"${0:.0f}$".format(percentages_cont[i]), 
                color = "white",
                horizontalalignment = "center", 
                verticalalignment = "center",
                alpha = 0.6,
                fontsize = 8
                )
            x_max_previous = x_max

        # Add colorbar
        cbar = fig.colorbar(im, shrink = 0.5)
        cbar.set_label(r"PDF [sr$^{-1}$]")

        # Plot the individual BHs as points on the map
        ax.scatter(scatter_coords.l.wrap_at('180d').radian, scatter_coords.b.radian, color = "white", marker='o', s = config["Plots"]["markersize"], alpha = 0.15, edgecolor = "None")

        # Set labels and title
        ax.set_xlabel('Galactic Longitude')
        ax.set_ylabel('Galactic Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)

    @staticmethod
    def expected_number_within_region(coord_stacked, lat_min, lat_max, long_min, long_max, kde, num_grid, name):
        # define the grids for the galactic plane surveys
        lat_grid = np.linspace(lat_min, lat_max, num_grid)
        long_grid = np.linspace(long_min, long_max, num_grid)
        Lat_grid, Long_grid = np.meshgrid(lat_grid, long_grid)
        coord_grid = np.vstack([Lat_grid.flatten(), Long_grid.flatten()]).T

        # evaluate kde on grid coordinates
        likelihood = np.exp(kde.score_samples(coord_grid))
        likelihood = likelihood.reshape(Lat_grid.shape)

        # determine the probability density function (PDF) by multiplying the likelihood with the area element
        dlat = (lat_max - lat_min) / num_grid
        dlon = (long_max - long_min) / num_grid
        area_element = np.cos(lat_grid) * dlat * dlon

        likelihood_integrated = np.sum(likelihood * area_element.ravel())

        coord_stacked_region = coord_stacked[(coord_stacked[:, 0] > lat_min) & (coord_stacked[:, 0] < lat_max) & (coord_stacked[:, 1] > long_min) & (coord_stacked[:, 1] < long_max)]

        percentage = len(coord_stacked_region) / len(coord_stacked) * 100

        print(f"Percentage of sources in {name} galactic plane survey:", np.round(percentage, 1), "%")
        print(f"Likelihood integral {name} galactic plane survey", np.round(likelihood_integrated * 100, 1), "%")


    def plot_2d_map_gaussian(self, lat, long, upsampling_factor, path):
        # Extract coordinates
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')
        coord_stacked = np.stack((coords.l.wrap_at('180d').radian, coords.b.radian), axis = -1)

        # choose subsample of coordinates for scatter plot
        # Calculate number of coordinates to be plotted
        n_coord_scatter = len(coords) // (upsampling_factor + 1)
        # Randomly select half of the indices
        scatter_indices = np.random.choice(len(coords), n_coord_scatter, replace = False)
        # Extract the scatter coordinates
        scatter_coords = coords[scatter_indices]

        # define grid for gaussian distribution
        num_grid = 250 # 250
        lat_min, lat_max = -np.pi / 2, np.pi / 2
        long_min, long_max = -np.pi, np.pi
        dlat = (lat_max - lat_min) / num_grid
        dlon = (long_max - long_min) / num_grid
        lat_grid = np.linspace(lat_min, lat_max, num_grid)
        long_grid = np.linspace(long_min, long_max, num_grid)
        Lat_grid, Long_grid = np.meshgrid(lat_grid, long_grid)
        coord_grid = np.vstack([Lat_grid.flatten(), Long_grid.flatten()]).T

        # define grid for contours
        coord_grid_contours = np.vstack([Long_grid.flatten(), Lat_grid.flatten()])

        # Define grid
        x = np.linspace(-np.pi, np.pi, num_grid)
        y = np.linspace(-np.pi / 2, np.pi / 2, num_grid)
        X, Y = np.meshgrid(x, y)

        positions = np.vstack([X.ravel(), Y.ravel()])

        # Calculate Mean
        mean_lat = np.mean(coords.b.radian)
        mean_lat_err = np.std(coords.b.radian) / np.sqrt(len(coords.b.radian))
        mean_long = np.mean(coords.l.wrap_at('180d').radian)
        mean_long_err = np.std(coords.l.wrap_at('180d').radian) / np.sqrt(len(coords.l.wrap_at('180d').radian))
        mean = [mean_long, mean_lat]

        print("Mean galactic latitude [deg]:", mean_lat * 180 / np.pi, "+-", mean_lat_err * 180 / np.pi)
        print("Mean galactic longitude [deg]:", mean_long * 180 / np.pi, "+-", mean_long_err * 180 / np.pi)

        # Calculate Covariance Matrix
        covariance = np.cov(coords.l.wrap_at('180d').radian, coords.b.radian)

        print("Covariance matrix [deg]:")
        print(covariance * 180 / np.pi)

        # Define Gaussian Distribution
        gaussian_model = multivariate_normal(mean, covariance)

        # Evaluate the Gaussian model at grid points
        gaussian_pdf = gaussian_model.pdf(np.dstack((X, Y)))

        # calculate cdf
        # define desired percentage contours to be calculated
        percentages_desired = [10, 20, 30, 40, 50]
        # initialize variables that will be filled in the loop later
        percentages_diff = np.full(len(percentages_desired), np.inf)
        percentages_cont = np.zeros(len(percentages_desired))
        percentages_cont_collections = np.empty(len(percentages_desired), dtype = object)

        # calculate the desired percentage contours with matplotlib
        fig_cdf = plt.figure(figsize = config["Figure_size"]["double_column"])
        ax_cdf = fig_cdf.add_subplot(111, projection="aitoff")
        # extract the PDF contours
        levels = np.linspace(gaussian_pdf.min(), gaussian_pdf.max(), 150)
        # levels = np.linspace(pdf.min(), pdf.max(), 30)
        cont = ax_cdf.contour(X, Y, gaussian_pdf, levels = levels)

        # Get the contour collections
        cont_collections = cont.collections

        def is_path_closed(path):
            """Check if a Matplotlib path is closed."""
            return path.codes is not None and path.codes[-1] == mpl.path.Path.CLOSEPOLY

        print("Calculating contours...")
        # Loop through each contour collection
        for collection in cont_collections:
            # Get the contour path
            cont_path = collection.get_paths()
            if cont_path: # check if the contour is not empty
                cont_path = cont_path[0]
                
                if is_path_closed(cont_path) == True:
                    mask = cont_path.contains_points(coord_grid_contours.T)
                    coord_grid_contours_masked = coord_grid_contours.T[mask]
                    lat_contours_masked = coord_grid_contours_masked[:, 1]
                    area_element = np.cos(lat_contours_masked) * dlat * dlon

                    pdf_masked = gaussian_pdf.flatten()[mask]
                    pdf_integrated = np.sum(pdf_masked * area_element.ravel())
                    cont_percentage = pdf_integrated * 100

                    # Check if the current contour is closer to the desired percentage than the previous one and update the variables
                    for i, percentage in enumerate(percentages_desired):
                        if np.abs(percentage - cont_percentage) < percentages_diff[i]:
                            percentages_cont[i] = cont_percentage
                            percentages_cont_collections[i] = collection
                            percentages_diff[i] = np.abs(percentage - cont_percentage)

        plt.close()

        # Plot in Aitoff projection
        fig = plt.figure(figsize = config["Figure_size"]["double_column"])
        ax = fig.add_subplot(111, projection="aitoff")
        ax.grid(True, alpha = 0.5)
        # plot the PDF
        im = ax.pcolormesh(X, Y, gaussian_pdf, cmap = LinearSegmentedColormap.from_list("", config["Colors"]["cmap_2d_map"]), edgecolors = "face", linewidth = 0, rasterized=True)

        # plot the contours
        x_max_previous = 0
        for i, collection in enumerate(percentages_cont_collections):
            cont_path = collection.get_paths()[0]
            x, y = zip(*cont_path.vertices)
            x_max, y_min = np.max(x), np.min(y)
            ax.plot(x, y, color = "white", alpha = 0.6, linestyle = "dashed")
            ax.text(
                x_max_previous + np.abs(x_max - x_max_previous) / 2, 
                - 5 * (2*np.pi/360), 
                r"${0:.0f}$".format(percentages_cont[i]), 
                color = "white",
                horizontalalignment = "center", 
                verticalalignment = "center",
                alpha = 0.6,
                fontsize = 8
                )
            x_max_previous = x_max

        # Add colorbar
        cbar = fig.colorbar(im, shrink = 0.5)
        cbar.set_label(r"PDF [sr$^{-1}$]")

        # Plot the individual BHs as points on the map
        ax.scatter(scatter_coords.l.wrap_at('180d').radian, scatter_coords.b.radian, color = "white", marker='o', s = config["Plots"]["markersize"], alpha = 0.15, edgecolor = "None")

        # Set labels and title
        ax.set_xlabel('Galactic Longitude')
        ax.set_ylabel('Galactic Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)

    def plot_cumulative_radial_distribution(self, distance, path):
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.hist(distance, bins = config["Plots"]["number_bins"], cumulative = True, density = True, color = config["Colors"]["darkblue"])
        plt.xlabel(r"$d_\mathrm{GC}$ [kpc]")
        plt.ylabel(r"$N(<d_\mathrm{GC}) / N_\mathrm{tot}$")
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()

    def cumulative_radial_distribution(self, distance, bins):
        cumulative_hist = []
        for bin in bins:
            cumulative_hist.append(np.sum(distance < bin) / len(distance))
        # skip first bin which is zero
        cumulative_hist = np.array(cumulative_hist[1:])
        return cumulative_hist

    def plot_cumulative_radial_distribution_mean(self, path):
        distance = self.table_bh["d_GC"].values
        galaxy_ids = np.unique(self.table_bh["main_galaxy_id"].values)
        d_min, d_max = np.min(distance), np.max(distance)
        # bins = np.logspace(np.log10(d_min), np.log10(d_max), config["Plots"]["number_bins"])
        bins = np.linspace(d_min, d_max, config["Plots"]["number_bins"])
        bins_width = bins[1:] - bins[:-1]
        bins_centre = (bins[1:] + bins[:-1]) / 2

        cumulative_hist_list = []
        for galaxy_id in galaxy_ids:
            distance_galaxy_id = distance[self.table_bh["main_galaxy_id"].values == galaxy_id]
            cumulative_hist = self.cumulative_radial_distribution(distance_galaxy_id, bins)
            cumulative_hist_list.append(cumulative_hist)
        
        cumulative_hist_mean = np.mean(cumulative_hist_list, axis = 0)
        cumulative_hist_std = np.std(cumulative_hist_list, ddof = 1, axis = 0)
        cumulative_hist_mean_error = cumulative_hist_std / np.sqrt(len(galaxy_ids))

        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.bar(bins_centre, cumulative_hist_mean, yerr = cumulative_hist_mean_error, width = bins_width, color = config["Colors"]["darkblue"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"], ecolor = config["Colors"]["lightblue"])
        plt.xlabel(r"$d_\mathrm{GC}$ [kpc]")
        plt.ylabel(r"$N(<d_\mathrm{GC}) / N_\mathrm{tot}$")
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()

    def parameter_distr_mean(self, parameter, bins):
        hist_list = []
        table_parameter = self.table_bh[parameter]
        for galaxy_id in np.unique(self.table_bh["main_galaxy_id"].values):
            data_galaxy_id = table_parameter[self.table_bh["main_galaxy_id"].values == galaxy_id]
            hist, _ = np.histogram(data_galaxy_id, bins = bins)
            hist_list.append(hist)
        
        hist_mean = np.mean(hist_list, axis = 0)
        hist_std = np.std(hist_list, ddof = 1, axis = 0)
        hist_mean_error = hist_std / np.sqrt(len(hist_list))

        return hist_mean, hist_mean_error


    def plot_dist_total_mean(self, path):
        parameters = ["m", "z_f", "d_GC", "lat_GC", "long_GC", "d_Sun", "lat_Sun", "long_Sun", "r_sp", "rho(r_sp)", "gamma_sp", "gamma_c"]
        units = ["$M_{\odot}$", "", "kpc", "rad", "rad", "kpc", "rad", "rad", "pc", "GeV/cm$^3$", "", ""]
        log_parameters = ["m", "r_sp", "rho(r_sp)"]
        # log_parameters = ["m"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]", "$r_\mathrm{sp}$ [pc]", r"$\rho(r_\mathrm{sp})$ [GeV/cm$^3$]", "$\gamma_\mathrm{sp}$", "$\gamma_\mathrm{c}$"]
        filenames = ["mass_mean", "redshift_mean", "distance_gc_mean", "latitude_GC_mean", "longitude_GC_mean", "distance_sun_mean", "latitude_Sun_mean", "longitude_Sun_mean", "r_sp_mean", "rho_sp_mean", "gamma_sp_mean", "gamma_c_mean"]
        
        for parameter, unit, x_label, filename in zip(parameters, units, x_labels, filenames):
            data = self.table_bh[parameter].values
            data_sorted = np.sort(data)
            data_mean = np.mean(data)
            data_std = np.std(data, ddof = 1)
            data_mean_error = data_std / np.sqrt(len(data))
            data_median = np.median(data)
            # Calculate the values at the 16th and 84th percentiles to get the error on the median
            lower_percentile = np.percentile(data, 16)
            upper_percentile = np.percentile(data, 84)
            median_lower_error = data_median - lower_percentile
            median_upper_error = upper_percentile - data_median

            # for plotting
            # Calculate the exponent (order of magnitude)
            # Calculate the exponent (order of magnitude)
            if data_median == 0:
                exponent = 0  # Handle zero separately
            else:
                exponent = int(math.floor(math.log10(abs(data_median))))

            # Normalize the values
            normalized_median = data_median / 10**exponent
            normalized_upper_error = median_upper_error / 10**exponent
            normalized_lower_error = median_lower_error / 10**exponent

            # Format the values for display
            formatted_median = "{:.2f}".format(normalized_median)
            formatted_upper_error = "{:.2f}".format(normalized_upper_error)
            formatted_lower_error = "{:.2f}".format(normalized_lower_error)

            # Create the label
            if exponent != 0:
                label = r"$\tilde{{\mu}} = ({0}^{{+{1}}}_{{-{2}}}) \cdot 10^{{ {3} }}$ {4}".format(formatted_median, formatted_upper_error, formatted_lower_error, exponent if exponent < 0 else " "+str(exponent), unit)
            else:
                label = r"$\tilde{{\mu}} = {0}^{{+{1}}}_{{-{2}}}$ {3}".format(formatted_median, formatted_upper_error, formatted_lower_error, unit)

            
            fig, ax = plt.subplots(figsize = config["Figure_size"]["single_column"])
            if parameter in log_parameters:
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), config["Plots"]["number_bins"])
                error_x_position = np.sqrt(bins[1:] * bins[:-1])
                ax.set_xscale("log")
                ax.set_yscale("log")
            else:
                bins = np.linspace(np.min(data), np.max(data), config["Plots"]["number_bins"])
                error_x_position = (bins[1:] + bins[:-1]) / 2

            bins_centre = (bins[1:] + bins[:-1]) / 2
            bins_width = bins[1:] - bins[:-1]
            
            hist_mean, hist_mean_error = self.parameter_distr_mean(parameter, bins)
            ax.bar(bins_centre, hist_mean, width = bins_width, color = config["Colors"]["darkblue"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"])
            ax.errorbar(error_x_position, hist_mean, yerr = hist_mean_error, color = config["Colors"]["lightblue"], linestyle = "")
            if parameter in log_parameters:
                ymin, ymax = ax.get_ylim()
                factor_ymax = 2
                ymin = np.min(hist_mean[hist_mean > 0]) * 0.5
                ymax = np.max(hist_mean[hist_mean > 0]) * factor_ymax
            else: 
                ymin, ymax = ax.get_ylim()
                factor_ymax = 1.
                ymax = ymax * factor_ymax
            
            ax.vlines(data_median, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = label)
            ax.axvspan(lower_percentile, upper_percentile, alpha = 0.25, facecolor = config["Colors"]["red"], edgecolor = "None")
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(x_label)
            ax.set_ylabel("$N_\mathrm{BH}$")
            if parameter in ["r_sp", "rho(r_sp)"]:
                legend = plt.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc = "center", mode = "expand", borderaxespad = 0, ncol = 1, alignment = "center")
            else:
                legend = plt.legend(loc = "upper right")
            plt.tight_layout()
            if parameter == "m":
                plt.xticks([2e5, 3e5, 4e5, 6e5], [r'$2 \times 10^5$', "", "", r'$6 \times 10^5$']) # TODO: find a better way to do this, plt.xticks() does not work
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

    @staticmethod
    def gaussian(params, x):
        norm, mu, sigma = params
        return norm * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

    @staticmethod
    def lognorm(params, x):
        norm, mu, sigma = params
        return norm * (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(x) - mu) / sigma)**2)

    @staticmethod
    def non_central_chi2_pdf(params, x):
        norm, df, nc = params
        return norm * scipy.stats.ncx2.pdf(x, df, nc)

    @staticmethod
    def chi_squared(observed, expected, errors):
        return np.sum(((observed - expected) ** 2) / errors ** 2)

    def plot_number_dist(self, path):
        galaxy_ids, n_bh = np.unique(self.table_bh["main_galaxy_id"].values, return_counts = True)
        n_bh_median = np.median(n_bh)
        n_bh_sorted = np.sort(n_bh)
        lower_percentile = np.percentile(n_bh_sorted, 16)
        upper_percentile = np.percentile(n_bh_sorted, 84)
        n_bh_median_lower_error = n_bh_median - lower_percentile
        n_bh_median_upper_error = upper_percentile - n_bh_median
        hist, bins = np.histogram(n_bh, bins = config["Plots"]["number_bins"])
        bins_width = bins[1:] - bins[:-1]
        bins_centre = (bins[1:] + bins[:-1]) / 2
        hist_err = np.sqrt(hist)

        # fit lognormal pdf to the distribution
        # Create a Model
        lognorm_model = Model(self.lognorm)

        # Create a RealData object
        data = RealData(bins_centre, hist, sy=hist_err)

        # Set up ODR with the model and data
        odr_lognorm = ODR(data, lognorm_model, beta0=[2000, np.mean(n_bh), np.std(bins_centre)])

        # Run the regression
        out_lognorm = odr_lognorm.run()
        print("Best fit values for lognorm number distribution fit:")
        print("Norm = {0:.2f} +- {1:.2f}".format(out_lognorm.beta[0], out_lognorm.sd_beta[0]))
        print("Mu = {0:.2f} +- {1:.2f}".format(out_lognorm.beta[1], out_lognorm.sd_beta[1]))
        print("Sigma = {0:.2f} +- {1:.2f}".format(out_lognorm.beta[2], out_lognorm.sd_beta[2]))

        # Use the fitted parameters to plot the fitted curve
        x_fit = np.linspace(0.1, max(bins_centre) + 20, 1000)
        y_fit_lognorm = self.lognorm(out_lognorm.beta, x_fit)

        # Format the values for display
        formatted_median = "{:.0f}".format(n_bh_median)
        formatted_upper_error = "{:.0f}".format(n_bh_median_upper_error)
        formatted_lower_error = "{:.0f}".format(n_bh_median_lower_error)
        
        label_median = r"$\tilde{{\mu}} = {0}^{{+{1}}}_{{-{2}}}$".format(formatted_median, formatted_upper_error, formatted_lower_error)
        # label_fit = f"Fit\n$\\mu$ = {np.round(out_lognorm.beta[1], 1)}\n$\\sigma$ = {np.round(out_lognorm.beta[2], 1)}"
        # label_fit = r"$f_\mathrm{ln}(N_\mathrm{BH}| \alpha, \mu, \sigma)$"
        label_fit = r"$f^0_\mathrm{ln}(N_\mathrm{BH})$"

        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.bar(bins_centre, hist, width = bins_width, color = config["Colors"]["darkblue"], yerr = hist_err, ecolor = config["Colors"]["lightblue"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"])
        x_min, x_max = plt.xlim()
        plt.plot(x_fit, y_fit_lognorm, color = config["Colors"]["black"], linestyle = "solid", label = label_fit)

        ymin, ymax = plt.ylim()
        plt.vlines(n_bh_median, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = label_median)
        plt.axvspan(lower_percentile, upper_percentile, alpha = 0.25, facecolor = config["Colors"]["red"], edgecolor = "None")
        plt.xlabel(r"$N_\mathrm{BH}$")
        plt.ylabel(r"$N_\mathrm{g}$")
        plt.ylim(ymin, ymax)
        plt.xlim(x_min, x_max)
        plt.legend(loc = "upper right")
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()

    def plot_number_dist_satellites(self, path):
        table_bh_satellites = self.table_bh[self.table_bh["satellite"] == True]
        table_bh_satellites_has_stars = table_bh_satellites[table_bh_satellites["m_star_host_galaxy"] > 0]

        n_bh_satellites = np.unique(table_bh_satellites["host_galaxy_id"].values, return_counts = True)[1]
        n_bh_satellites_has_stars = np.unique(table_bh_satellites_has_stars["host_galaxy_id"].values, return_counts = True)[1]

        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.hist(n_bh_satellites, bins = np.max(n_bh_satellites), color = config["Colors"]["darkblue"])
        plt.xlabel(r"Number of IMBHs")
        plt.ylabel(r"Number of satellite galaxies")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(path + "number_dist_satellites.pdf", dpi = 500)
        plt.close()

        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.hist(n_bh_satellites_has_stars, bins = np.max(n_bh_satellites_has_stars), color = config["Colors"]["darkblue"])
        plt.xlabel(r"Number of IMBHs")
        plt.ylabel(r"Number of satellite galaxies with stars", fontsize = 5)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(path + "number_dist_satellites_has_stars.pdf", dpi = 500)
        plt.close()

    @staticmethod
    def plot_spike_profile(radii, rho_total, r_schw, r_cut, r_sp, path):
        radii = radii.to(u.pc)
        r_schw = r_schw.to(u.pc)
        r_cut = r_cut.to(u.pc)
        r_sp = r_sp.to(u.pc)
        # Generate a log-log plot
        plt.figure(figsize=config["Figure_size"]["single_column_extended"])
        plt.loglog(radii, rho_total, color = config["Colors"]["black"])
        ymin, ymax = plt.ylim()
        plt.vlines(2*r_schw.value, ymin, ymax, color=config["Colors"]["red"], linestyle='--', label = "$2r_\mathrm{schw}$")
        plt.vlines(r_cut.value, ymin, ymax, color=config["Colors"]["red"], linestyle='-.', label = "$r_\mathrm{cut}$")
        plt.vlines(r_sp.value, ymin, ymax, color=config["Colors"]["red"], linestyle='dotted', label = "$r_\mathrm{sp}$")
        plt.ylim(ymin, ymax)
        plt.xlabel(r'$r$ [pc]')
        plt.ylabel(r'$\rho(r)$ [GeV cm$^{-3}$]')
        plt.legend()
        # plt.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc = "lower left", mode = "expand", borderaxespad = 0, ncol = 3)
        plt.tight_layout()
        plt.savefig(path + "spike_profile.pdf", dpi = 300)

    def plot_n_bh_in_satellites(self, path):
        n_bh_in_galaxies = len(self.table_bh[self.table_bh["satellite"] == False])
        n_bh_in_satellites = len(self.table_bh[self.table_bh["satellite"] == True])

        # create a bar plot with these two numbers
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.bar(["Main galaxies", "Satellites"], [n_bh_in_galaxies, n_bh_in_satellites], color = config["Colors"]["darkblue"])
        plt.ylabel("Number of IMBHs")
        plt.tight_layout()
        plt.savefig(path + "n_bh_in_satellites.pdf", dpi = 300)
        plt.close()

    def plot_scatter_bh_n_satellites(self, path):
        # two things: number satellites vs. number of IMBHs.
        galalaxy_ids = np.unique(self.table_bh["main_galaxy_id"].values)
        # for each galaxy, determine the number of satellites and the number of IMBHs
        n_satellites = []
        n_bh = []
        for galaxy_id in galalaxy_ids:
            n_bh_galaxy = len(self.table_bh[self.table_bh["main_galaxy_id"] == galaxy_id])
            n_satellites_galaxy_unique = np.unique(self.table_bh[self.table_bh["main_galaxy_id"] == galaxy_id]["n_satellites"].values)
            if len(n_satellites_galaxy_unique) != 1:
                print(f"WARNING: More than one number of satellites for galaxy {galaxy_id}! This should not be possible! Check catalogue!")
            else:
                n_satellites_galaxy = n_satellites_galaxy_unique[0]
            n_satellites.append(n_satellites_galaxy)
            n_bh.append(n_bh_galaxy)
        
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.scatter(n_satellites, n_bh, color = config["Colors"]["darkblue"], marker = "o", s = config["Plots"]["markersize"])
        plt.xlabel("Number of satellites")
        plt.ylabel("Number of IMBHs")
        plt.tight_layout()
        plt.savefig(path + "scatter_bh_n_satellites.pdf", dpi = 300)
        plt.close()

    def plot_scatter_bh_galaxy_properties(self, path):
        table_bh_main_galaxies = self.table_bh[self.table_bh["satellite"] == False]
        main_galaxy_ids = np.unique(table_bh_main_galaxies["main_galaxy_id"].values)
        galaxy_ids = np.unique(self.table_bh["main_galaxy_id"].values)

        table_bh_satellites = self.table_bh[self.table_bh["satellite"] == True]
        satellites_galaxy_ids = np.unique(table_bh_satellites["host_galaxy_id"].values)

        parameters = ["m_host_galaxy", "m200_main_galaxy", "m_gas_host_galaxy", "m_star_host_galaxy", "sfr_host_galaxy"]
        log_parameters = ["m_host_galaxy", "m200_main_galaxy", "m_gas_host_galaxy", "m_star_host_galaxy"] 
        names = ["total_mass", "m200", "gas_mass", "star_mass", "sfr"]
        x_labels = [r"$m_\mathrm{tot}$ [$M_{\odot}$]", r"$m_{200}$ [$M_{\odot}$]", r"$m_{\mathrm{gas}}$ [$M_{\odot}$]", r"$m_\mathrm{star}$ [$M_{\odot}$]", "SFR [$M_{\odot}$ / yr]"]

        for parameter, name, x_label in zip(parameters, names, x_labels):
            # extract the number of BHs and the parameter values for the main galaxies (considering BHs in the main galaxy and the satellites)
            n_bh = []
            parameter_values = []
            for galaxy_id in galaxy_ids:
                table_bh_galaxy = self.table_bh[self.table_bh["main_galaxy_id"] == galaxy_id]
                n_bh_galaxy = len(table_bh_galaxy)
                parameter_value_galaxy = np.unique(table_bh_galaxy[table_bh_galaxy["satellite"] == False][parameter].values) # remove the satellite entries here to get the property of the main galaxy
                if len(parameter_value_galaxy) != 1:
                    print(f"WARNING: More than one value for parameter {parameter} for galaxy {galaxy_id}! This should not be possible! Check catalogue!")
                n_bh.append(n_bh_galaxy)
                parameter_values.append(parameter_value_galaxy[0])

            # extract the number of BHs and the parameter values for the main galaxies (excluding BHs in the satellites)
            n_bh_main_galaxies = []
            parameter_values_main_galaxies = []
            for main_galaxy_id in main_galaxy_ids:
                table_bh_main_galaxy = table_bh_main_galaxies[table_bh_main_galaxies["main_galaxy_id"] == main_galaxy_id]
                n_bh_main_galaxy = len(table_bh_main_galaxy)
                parameter_value_galaxy = np.unique(table_bh_main_galaxy[parameter].values)
                if len(parameter_value_galaxy) != 1:
                    print(f"WARNING: More than one value for parameter {parameter} for galaxy {main_galaxy_id}! This should not be possible! Check catalogue!")
                n_bh_main_galaxies.append(n_bh_main_galaxy)
                parameter_values_main_galaxies.append(parameter_value_galaxy[0])

            plt.figure(figsize = config["Figure_size"]["single_column"])
            plt.scatter(parameter_values, n_bh, color = config["Colors"]["darkblue"], marker = "o", s = config["Plots"]["markersize"])
            plt.xlabel(x_label + " (of main galaxy)")
            plt.ylabel("$N_\mathrm{BH}$ (in main galaxy + satellites)")
            if parameter in log_parameters:
                plt.xscale("log")
            plt.tight_layout()
            plt.savefig(path + f"scatter_bh_{name}.pdf", dpi = 300)
            plt.close()

            plt.figure(figsize = config["Figure_size"]["single_column"])
            plt.scatter(parameter_values_main_galaxies, n_bh_main_galaxies, color = config["Colors"]["darkblue"], marker = "o", s = config["Plots"]["markersize"])
            plt.xlabel(x_label + " (of main galaxy)")
            plt.ylabel("$N_\mathrm{BH}$ (in main galaxy)")
            if parameter in log_parameters:
                plt.xscale("log")
            plt.tight_layout()
            plt.savefig(path + f"scatter_bh_{name}_main_galaxy.pdf", dpi = 300)
            plt.close()

            if parameter != "m200_main_galaxy":
                # extract the number of BHs and the parameter values for the satellite galaxies
                n_bh_satellites = []
                parameter_values_satellites = []
                for satellite_galaxy_id in satellites_galaxy_ids:
                    table_bh_satellite_galaxy = table_bh_satellites[table_bh_satellites["host_galaxy_id"] == satellite_galaxy_id]
                    n_bh_satellite_galaxy = len(table_bh_satellite_galaxy)
                    parameter_value_galaxy = np.unique(table_bh_satellite_galaxy[parameter].values)
                    if len(parameter_value_galaxy) != 1:
                        print(f"WARNING: More than one value for parameter {parameter} for galaxy {satellite_galaxy_id}! This should not be possible! Check catalogue!")
                    n_bh_satellites.append(n_bh_satellite_galaxy)
                    parameter_values_satellites.append(parameter_value_galaxy[0])

                plt.figure(figsize = config["Figure_size"]["single_column"])
                plt.scatter(parameter_values_satellites, n_bh_satellites, color = config["Colors"]["darkblue"], marker = "o", s = config["Plots"]["markersize"])
                plt.xlabel(x_label + " (of satellite galaxy)")
                plt.ylabel("$N_\mathrm{BH}$ (in satellites)")
                if parameter in log_parameters:
                    plt.xscale("log")
                plt.tight_layout()
                plt.savefig(path + f"scatter_bh_{name}_satellites.pdf", dpi = 300)
                plt.close()

    def plot_bh_in_satellite_types(self, path):
        # get satellites
        table_satellites = self.table_bh[self.table_bh["satellite"] == True]

        n_bh_in_satellites = len(table_satellites)
        n_bh_in_satellites_no_gas_no_star = len(table_satellites[(table_satellites["m_gas_host_galaxy"] == 0) & (table_satellites["m_star_host_galaxy"] == 0)])
        n_bh_in_satellites_has_gas = len(table_satellites[table_satellites["m_gas_host_galaxy"] > 0])
        n_bh_in_satellites_has_star = len(table_satellites[table_satellites["m_star_host_galaxy"] > 0])
        n_bh_in_satellites_has_gas_has_star = len(table_satellites[(table_satellites["m_gas_host_galaxy"] > 0) & (table_satellites["m_star_host_galaxy"] > 0)])

        # plot bar plot
        plt.figure(figsize = config["Figure_size"]["single_column"])
        plt.bar(["All", "No gas, no stars", "Has gas", "Has stars", "Has gas, has stars"], [n_bh_in_satellites, n_bh_in_satellites_no_gas_no_star, n_bh_in_satellites_has_gas, n_bh_in_satellites_has_star, n_bh_in_satellites_has_gas_has_star], color = config["Colors"]["darkblue"])
        plt.ylabel("$N_\mathrm{BH}$")
        # rotate the x-axis labels
        plt.xticks(rotation = 90)
        plt.tight_layout()
        plt.savefig(path + "satellite_types.pdf", dpi = 300)
        plt.close()        


class FluxPlotter:
    def __init__(self, flux_catalogue):
        self.flux_catalogue = flux_catalogue

    @staticmethod
    def flux_thresholds(flux, flux_min = None, flux_max = None):
        if flux_min is None:
            flux_min = np.min(flux.value)
        if flux_max is None:
            flux_max = np.max(flux.value)
        flux_th = np.logspace(np.log10(flux_min), np.log10(flux_max), 30) * flux.unit
        return(flux_th)

    @staticmethod
    def integrated_luminosity(flux, flux_th):
        int_lum = []
        for threshold in flux_th:
            int_lum.append(len(flux[flux >= threshold]))
        return(int_lum)
    
    def integrated_luminosity_mean(self, flux_th):
        int_lum_list = []
        galaxy_ids = np.unique(self.flux_catalogue["main_galaxy_id"].values)
        for galaxy_id in galaxy_ids:
            flux_catalogue_id = self.flux_catalogue[self.flux_catalogue["main_galaxy_id"] == galaxy_id]
            flux_id = flux_catalogue_id["flux"].values * u.Unit("cm-2 s-1")
            int_lum_id = self.integrated_luminosity(flux_id, flux_th)
            int_lum_list.append(int_lum_id)
        int_lum_mean = np.mean(int_lum_list, axis = 0)
        int_lum_std = np.std(int_lum_list, ddof = 1, axis = 0)
        int_lum_mean_error = int_lum_std / np.sqrt(len(int_lum_list))
        return(int_lum_mean, int_lum_mean_error)
 
    def plot_integrated_luminosity(self, flux_th, m_dm, color, marker, marker_size, args):
        int_lum_mean, int_lum_error = self.integrated_luminosity_mean(flux_th)
        if args.instrument_comparison == "hess":
            plt.errorbar(flux_th, int_lum_mean, yerr = int_lum_error, label = f"$m_{{\chi}}$ = {np.round(m_dm.to(u.TeV).value, 1)} TeV", linestyle = "", marker = marker, capsize = 3, color = color, markersize = marker_size)
        elif args.instrument_comparison == "fermi":
            plt.errorbar(flux_th, int_lum_mean, yerr = int_lum_error, label = r"$m_{{\chi}}$ = {0:.0f} GeV".format(m_dm.to(u.GeV).value), linestyle = "", marker = marker, capsize = 3, color = color, markersize = marker_size)


    def plot_integrated_luminosity_comparison(self, flux_th, label, color,  marker, marker_size):
        int_lum_mean, int_lum_error = self.integrated_luminosity_mean(flux_th)
        plt.errorbar(flux_th, int_lum_mean, yerr = int_lum_error, label = label, linestyle = "", marker = marker, capsize = 3, color = color, markersize = marker_size)

    def plot_cuttoff_radius_dist(self, path):
        r_cut = self.flux_catalogue["r_cut"].values
        r_cut_mean = np.mean(r_cut)
        r_cut_mean_error = np.std(r_cut, ddof = 1) / np.sqrt(len(r_cut))
        r_cut_median = np.median(r_cut)
        # Calculate the values at the 16th and 84th percentiles to get the error on the median
        lower_percentile = np.percentile(r_cut, 16)
        upper_percentile = np.percentile(r_cut, 84)
        median_lower_error = r_cut_median - lower_percentile
        median_upper_error = upper_percentile - r_cut_median

        exponent = int(math.floor(math.log10(abs(r_cut_median))))

        # Normalize the values
        normalized_median = r_cut_median / 10**exponent
        normalized_upper_error = median_upper_error / 10**exponent
        normalized_lower_error = median_lower_error / 10**exponent

        # Format the values for display
        formatted_median = "{:.2f}".format(normalized_median)
        formatted_upper_error = "{:.2f}".format(normalized_upper_error)
        formatted_lower_error = "{:.2f}".format(normalized_lower_error)

        # Create the label
        label = r"$\tilde{{\mu}} = ({0}^{{+{1}}}_{{-{2}}}) \cdot 10^{{{3}}}$ {4}".format(formatted_median, formatted_upper_error, formatted_lower_error, exponent, "pc")

        bins = np.logspace(np.log10(np.min(r_cut)), np.log10(np.max(r_cut)), config["Plots"]["number_bins"])
        bins_centre = (bins[1:] + bins[:-1]) / 2
        bins_width = bins[1:] - bins[:-1]

        error_x_position = np.sqrt(bins[1:] * bins[:-1])

        plt.figure(figsize = config["Figure_size"]["single_column"])
        hist_mean, hist_mean_error = parameter_distr_mean(table = self.flux_catalogue, parameter = "r_cut", bins = bins)
        plt.bar(bins_centre, hist_mean, width = bins_width, color = config["Colors"]["darkblue"], edgecolor = config["Plots"]["bar_edge_color"], linewidth = config["Plots"]["bar_edge_width"])
        plt.errorbar(error_x_position, hist_mean, yerr = hist_mean_error, color = config["Colors"]["lightblue"], linestyle = "")
        ymin, ymax = plt.ylim()
        ymax = ymax * 2
        plt.vlines(r_cut_median, ymin = ymin, ymax = ymax, color = config["Colors"]["red"], linestyle = "solid", label = label)
        plt.axvspan(lower_percentile, upper_percentile, alpha = 0.25, facecolor = config["Colors"]["red"], edgecolor = "None")
        plt.ylim(1e-3, ymax)
        plt.xlabel("$r_\mathrm{cut}$ [pc]")
        plt.ylabel("$N_\mathrm{BH}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc = "lower left", mode = "expand", borderaxespad = 0, ncol = 1)
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()