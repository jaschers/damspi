import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the dammspi module directory to the Python path
module_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(module_dir)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import requests
from dammspi.utils import nfw_profile, nfw_integral, cored_profile, cored_integral, M_bh_2, parameter_distr_mean
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from astropy.wcs import WCS
from astropy.io import fits
import math

# set matplotlib parameters for nice looking plots
plt.rcParams.update({'font.size': 8}) # 8 (paper), 10 (poster)
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')#, weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'
cm_conversion_factor = 1/2.54  # centimeters in inches
# single_column_fig_size = (8.85679 * cm_conversion_factor, 8.85679 * 3/4 * cm_conversion_factor)
single_column_fig_size = (7.0 * cm_conversion_factor, 7.0 * 3/4 * cm_conversion_factor)
single_column_fig_size_legend = (8.85679 * cm_conversion_factor, 8.85679 * 3/4 * 7/6 * cm_conversion_factor)
# double_column_fig_size = (18.34621 * cm_conversion_factor, 18.34621 * 3/4 * cm_conversion_factor)
double_column_fig_size = (14.5 * cm_conversion_factor, 14.5 * 3/4 * cm_conversion_factor)
double_column_squeezed_fig_size = (18.34621 * cm_conversion_factor, 18.34621 * 1/2 * cm_conversion_factor)
markersize = 2
bar_edge_color = "white"
bar_edge_width = 0.4
number_bins = 11

cmap = LinearSegmentedColormap.from_list("", ["#FFF275", "#E83151", "#003049", "#171B1D"])
cmap_r = LinearSegmentedColormap.from_list("", ["#171B1D", "#003049", "#E83151", "#FFF275"])
color_black = "#171b1d"
color_red = "#e83151"
color_red_presentation = "#F44D4D"
color_darkblue = "#003049"
color_lightblue = "#6c8ead"
color_yellow = "#fca311"

class GalaxyPlotter:
    def __init__(self, sim_name, table_galaxy, table_bh):
        self.sim_name = sim_name
        self.table_galaxy = table_galaxy
        self.table_bh = table_bh
        # shift coorindates system to position of the sun (8.33 kpc), https://iopscience.iop.org/article/10.1088/1475-7516/2011/03/051/pdf
        self.distance_sun = 8.33 # kpc

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
            galaxy_id = self.table_galaxy["galaxy_id"].values[i]

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
        plt.figure(figsize = single_column_fig_size)
        # plt.title("Total galaxy mass: {0:.1e} $M_{{\odot}}$".format(galaxy_mass_z0))
        plt.hist(mass, bins = 5, color = color_darkblue)
        plt.xlabel(r"Galaxy mass [$M_{\odot}$]")
        plt.ylabel("Number of galaxies")
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

        fig = plt.figure(figsize = single_column_fig_size)
        ax = fig.add_subplot(projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, len(coordinates)))
        for coord, color in zip(coordinates, colors):
            ax.scatter(coord[0], coord[1], coord[2], color = color, s = 10)
        if shifted:
            ax.scatter(0, 0, 0, color = "yellow", s = 40, label = "Sun")
            ax.scatter(-self.distance_sun, 0, 0, color = "black", s = 40, label = "Galaxy centre")
            ax.quiver(-self.distance_sun, 0, 0, *galaxy_spin * np.max(coordinates), color = "black", label = "Galaxy spin vector")
        else:
            ax.scatter(0, 0, 0, color = "black", s = 40, label = "Galaxy centre")
            ax.quiver(0, 0, 0, *galaxy_spin * np.max(coordinates), color = "black", label = "Galaxy spin vector")
        text = str(int(galaxy_spin[0] * np.max(coordinates))) + ', ' + str(int(galaxy_spin[1] * np.max(coordinates))) + ', ' + str(int(galaxy_spin[2] * np.max(coordinates)))
        ax.text(*galaxy_spin * np.max(coordinates), text)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.set_zlabel("z [kpc]")
        ax.set_xlim(np.array([-plot_max, plot_max]))
        ax.set_ylim(np.array([-plot_max, plot_max]))
        ax.set_zlim(np.array([-plot_max, plot_max]))
        plt.legend(loc = "upper left")
        plt.savefig(path + ".pdf", dpi = 500)
        if save_animation:
            ani = animation.FuncAnimation(fig, self.update, fargs = [ax], frames=72, interval=50)
            ani.save(path + ".gif", writer='imagemagick')
        plt.close()

class BlackHolePlotter:
    def __init__(self, sim_name, table_bh):
        self.sim_name = sim_name
        self.table_bh = table_bh

    def plot_bh_dist_galaxy(self, path):
        os.makedirs(path, exist_ok = True)

        parameters = ["m [M_solar]", "z_f", "d_GC [kpc]", "lat_GC [rad]", "long_GC [rad]", "d_Sun [kpc]", "lat_Sun [rad]", "long_Sun [rad]"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]"]
        filenames = ["mass", "redshift", "distance_GC", "latitude_GC", "longitude_GC", "distance_Sun", "latitude_Sun", "longitude_Sun"]
        
        for parameter, x_label, filename in zip(parameters, x_labels, filenames):
            # plot galaxy mass distribution
            data = self.table_bh[parameter].values
            plt.figure(figsize = single_column_fig_size)
            if parameter == "m [M_solar]":
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 5)
                plt.xscale("log")
            else:
                bins = 5
            plt.hist(data, bins = bins, color = color_darkblue)
            plt.xlabel(x_label)
            plt.ylabel("$N_\mathrm{BH}$")
            plt.tight_layout()
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

    @staticmethod
    def plot_nfw(r, rho, rho_0, r_s, galaxy_mass, z, path):
        lablel_fit = f"Fit\n$\\rho_0$ = {rho_0.value:.2e} M$_{{\\odot}}$ / kpc$^3$\n$r_s$ = {r_s.value:.2f} kpc"
        plt.figure(figsize = double_column_fig_size)
        plt.title(f"M$_{{halo}}$ = {galaxy_mass.value:.2e} M$_{{\\odot}}$, z = {z:.2f}")
        plt.plot(r, rho, color = color_black, label = "Data")
        plt.plot(r, nfw_profile((rho_0, r_s), r), color = color_red, linestyle = "dashed", label = lablel_fit)
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
        plt.figure(figsize = double_column_fig_size)
        plt.title(f"M$_{{halo}}$ = {galaxy_mass.value:.2e} M$_{{\\odot}}$, z = {z:.2f}")
        plt.plot(r, rho, color = color_black, label = "Data")
        plt.plot(r, cored_profile((rho_0, r_s, r_c, gamma_c), r), color = color_red, linestyle = "dashed", label = lablel_fit)
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
        plt.plot(r, nfw_integral(r, rho_0, r_s), color = color_black, label = "NFW integral")
        plt.hlines(M_bh_2(M_bh).value, xmin = np.min(r).value, xmax = np.max(r).value, linestyle = "dotted", color = color_black, label = f"$2 \cdot M_{{BH}}$ = {M_bh_2(M_bh).value:.2e} M$_{{\\odot}}$")
        plt.vlines(r_h.value, ymin = np.min(nfw_integral(r, rho_0, r_s)).value, ymax = np.max(nfw_integral(r, rho_0, r_s)).value, linestyle = "dashed", color = color_red, label = f"$r_h$ = {r_h:.2e}\n$r_{{sp}}$ = {0.2 * r_h:.2e}")
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
        plt.plot(r, cored_integral(r, rho_0, r_s, r_c, gamma_c), color = color_black, label = "Cored integral")
        plt.hlines(M_bh_2(M_bh).value, xmin = np.min(r).value, xmax = np.max(r).value, linestyle = "dotted", color = color_black, label = f"$2 \cdot M_{{BH}}$ = {M_bh_2(M_bh).value:.2e} M$_{{\\odot}}$")
        plt.vlines(r_h.value, ymin = np.min(cored_integral(r, rho_0, r_s, r_c, gamma_c)).value, ymax = np.max(cored_integral(r, rho_0, r_s, r_c, gamma_c)).value, linestyle = "dashed", color = color_red, label = f"$r_h$ = {r_h:.2e}\n$r_{{sp}}$ = {0.2 * r_h:.2e}")
        plt.xlabel(r"$r$ [kpc]")
        plt.ylabel(r"$Y$ [M$_{\odot}$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "r_h_cored.pdf", dpi = 300)
        plt.close()

    def plot_dist_total(self, path):
        parameters = ["m [M_solar]", "z_f", "d_GC [kpc]", "lat_GC [rad]", "long_GC [rad]", "d_Sun [kpc]", "lat_Sun [rad]", "long_Sun [rad]", "r_sp [pc]", "rho(r_sp) [GeV/cm3]"]
        log_parameters = ["m [M_solar]", "r_sp [pc]", "rho(r_sp) [GeV/cm3]"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]", "r_sp [pc]", r"$\rho(r_\mathrm{sp})$ [GeV/cm$^3$]"]
        filenames = ["mass", "redshift", "distance_gc", "latitude_GC", "longitude_GC", "distance_sun", "latitude_Sun", "longitude_Sun", "r_sp", "rho_sp"]
        
        for parameter, x_label, filename in zip(parameters, x_labels, filenames):
            data = self.table_bh[parameter].values
            plt.figure(figsize = single_column_fig_size)
            if parameter in log_parameters:
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 9)
                plt.xscale("log")
            else:
                bins = 9
            plt.hist(data, bins = bins, color = color_darkblue)
            plt.xlabel(x_label)
            plt.ylabel("$N_\mathrm{BH}$")
            plt.tight_layout()
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

            if parameter == "lat_GC [rad]" or parameter == "lat_Sun [rad]":
                data = np.cos(data)
                if parameter == "lat_GC [rad]":
                    x_label = r"$\cos(b_\mathrm{GC})$"
                    filename = "latitude_GC_cos"
                if parameter == "lat_Sun [rad]":
                    x_label = r"$\cos(b_\mathrm{Sun})$"
                    filename = "latitude_Sun_cos"
                plt.figure(figsize = single_column_fig_size)
                bins = 9
                plt.hist(data, bins = bins, color = color_darkblue)
                plt.xlabel(x_label)
                plt.ylabel("$N_\mathrm{BH}$")
                plt.tight_layout()
                plt.savefig(path + f"{filename}.pdf", dpi = 500)
                plt.close()

    def plot_2d_map(self, lat, long, path):
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')

        fig = plt.figure(figsize  = single_column_fig_size)
        ax = fig.add_subplot(111, projection='aitoff')
        ax.grid(True, alpha = 0.5)
        ax.scatter(coords.l.wrap_at('180d').radian, coords.b.radian, color = color_darkblue, marker = 'o', s = markersize / 100)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()


    def plot_2d_map_contours(self, lat, long, path):
        # Extract coordinates
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')
        coord_stacked = np.stack((coords.l.wrap_at('180d').radian, coords.b.radian), axis = -1)

        # Calculate kernel density estimation
        kde = gaussian_kde([coords.l.wrap_at('180d').radian, coords.b.radian])

        # Define grid for evaluating KDE
        x = np.linspace(-np.pi, np.pi, 480)
        y = np.linspace(-np.pi / 2, np.pi / 2, 240)
        X, Y = np.meshgrid(x, y)

        # HESS galactic plane survey
        hess_lat_min, hess_lat_max = -3, 3
        hess_long_min, hess_long_max = -65, 110

        # CTA galactic plane survey
        cta_lat_min, cta_lat_max = -6, 6
        cta_long_min, cta_long_max = -90, 90

        # convert coordinates to rad
        hess_lat_min, hess_lat_max = hess_lat_min * np.pi / 180, hess_lat_max * np.pi / 180
        hess_long_min, hess_long_max = hess_long_min * np.pi / 180, hess_long_max * np.pi / 180
        cta_lat_min, cta_lat_max = cta_lat_min * np.pi / 180, cta_lat_max * np.pi / 180
        cta_long_min, cta_long_max = cta_long_min * np.pi / 180, cta_long_max * np.pi / 180

        positions = np.vstack([X.ravel(), Y.ravel()])

        # Evaluate KDE at grid points, i.e. estimating the probability density function (PDF)
        pdf = np.reshape(kde(positions).T, X.shape)

        # normalize pdf
        pdf = pdf / np.sum(pdf)

        # sum up all values in pdf within HESS/CTA galactic plane survey
        pdf_galactic_plane_hess = np.sum(pdf[(Y > hess_lat_min) & (Y < hess_lat_max) & (X > hess_long_min) & (X < hess_long_max)])
        pdf_galactic_plane_cta = np.sum(pdf[(Y > cta_lat_min) & (Y < cta_lat_max) & (X > cta_long_min) & (X < cta_long_max)])
        print("PDF integral HESS galactic plane survey", np.round(pdf_galactic_plane_hess * 100), "%")
        print("PDF integral CTA galactic plane survey", np.round(pdf_galactic_plane_cta * 100), "%")

        # calculate cdf
        # define desired percentage contours to be calculated
        percentages_desired = [10, 20, 40, 60, 80]
        # percentages_desired = [10, 20, 30]
        # initialize variables that will be filled in the loop later
        percentages_diff = np.full(len(percentages_desired), np.inf)
        percentages_cont = np.zeros(len(percentages_desired))
        percentages_cont_collections = np.empty(len(percentages_desired), dtype = object)

        # calculate the desired percentage contours with matplotlib
        fig_cdf = plt.figure(figsize=double_column_fig_size)
        ax_cdf = fig_cdf.add_subplot(111, projection="aitoff")
        # extract the PDF contours
        levels = np.linspace(pdf.min(), pdf.max(), 150)
        # levels = np.linspace(pdf.min(), pdf.max(), 30)
        cont = ax_cdf.contour(X, Y, pdf, levels = levels)

        # Get the contour collections
        cont_collections = cont.collections

        # Loop through each contour collection
        for collection in cont_collections:
            # Get the contour path
            cont_path = collection.get_paths()
            if cont_path: # check if the contour is not empty
                cont_path = cont_path[0]
                # Get the grid points within the contour path
                mask = cont_path.contains_points(positions.T)
                # Get the sum of PDF values within the contour path
                cont_percentage = pdf.flatten()[mask].sum() * 100
                # Check if the current contour is closer to the desired percentage than the previous one and update the variables
                for i, percentage in enumerate(percentages_desired):
                    if np.abs(percentage - cont_percentage) < percentages_diff[i]:
                        percentages_cont[i] = cont_percentage
                        percentages_cont_collections[i] = collection
                        percentages_diff[i] = np.abs(percentage - cont_percentage)

        plt.close()

        # Plot in Aitoff projection
        fig = plt.figure(figsize=double_column_fig_size)
        ax = fig.add_subplot(111, projection="aitoff")
        ax.grid(True, alpha = 0.5)
        # plot the PDF
        im = ax.pcolormesh(X, Y, pdf, cmap = cmap_r, edgecolors = "face", linewidth = 0, rasterized=True)

        # plot the contours
        x_max_previous = 0
        for i, collection in enumerate(percentages_cont_collections):
            cont_path = collection.get_paths()[0]
            x, y = zip(*cont_path.vertices)
            x_max, y_min = np.max(x), np.min(y)
            ax.plot(x, y, color = "white", alpha = 0.8)
            ax.text(
                x_max_previous + np.abs(x_max - x_max_previous) / 2, 
                - 5 * (2*np.pi/360), 
                r"${0:.0f}$".format(percentages_cont[i]), 
                color = "white",
                horizontalalignment = "center", 
                verticalalignment = "center",
                alpha = 0.8,
                fontsize = 8
                )
            x_max_previous = x_max

        # Add colorbar
        cbar = fig.colorbar(im, shrink = 0.5)
        cbar.set_label("Probability density function")

        # Plot the individual BHs as points on the map
        ax.scatter(coords.l.wrap_at('180d').radian, coords.b.radian, color = "white", marker='o', s = markersize, alpha = 0.15, edgecolor = "None")

        # Set labels and title
        ax.set_xlabel('Galactic Longitude')
        ax.set_ylabel('Galactic Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)

    def plot_cumulative_radial_distribution(self, distance, path):
        plt.figure(figsize = single_column_fig_size)
        plt.hist(distance, bins = number_bins, cumulative = True, density = True, color = color_darkblue)
        plt.xlabel(r"$d_\mathrm{GC}$ [kpc]")
        plt.ylabel(r"$N(<r) / N_\mathrm{tot}$")
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
        distance = self.table_bh["d_GC [kpc]"].values
        galaxy_ids = np.unique(self.table_bh["galaxy_id"].values)
        d_min, d_max = np.min(distance), np.max(distance)
        # bins = np.logspace(np.log10(d_min), np.log10(d_max), number_bins)
        bins = np.linspace(d_min, d_max, number_bins)
        bins_width = bins[1:] - bins[:-1]
        bins_centre = (bins[1:] + bins[:-1]) / 2

        cumulative_hist_list = []
        for galaxy_id in galaxy_ids:
            distance_galaxy_id = distance[self.table_bh["galaxy_id"].values == galaxy_id]
            cumulative_hist = self.cumulative_radial_distribution(distance_galaxy_id, bins)
            cumulative_hist_list.append(cumulative_hist)
        
        cumulative_hist_mean = np.mean(cumulative_hist_list, axis = 0)
        cumulative_hist_std = np.std(cumulative_hist_list, ddof = 1, axis = 0)
        cumulative_hist_mean_error = cumulative_hist_std / np.sqrt(len(galaxy_ids))

        plt.figure(figsize = single_column_fig_size)
        # plt.errorbar(bins, cumulative_hist_mean, yerr = cumulative_hist_mean_error, color = color_darkblue, linestyle = "", marker = ".", markersize = 0.01)
        plt.bar(bins_centre, cumulative_hist_mean, yerr = cumulative_hist_mean_error, width = bins_width, color = color_darkblue, edgecolor = bar_edge_color, linewidth = bar_edge_width, ecolor = color_lightblue)
        plt.xlabel(r"$d_\mathrm{GC}$ [kpc]")
        plt.ylabel(r"$N(<r) / N_\mathrm{tot}$")
        # plt.xscale("log")
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()

    def parameter_distr_mean(self, parameter, bins):
        hist_list = []
        table_parameter = self.table_bh[parameter]
        for galaxy_id in np.unique(self.table_bh["galaxy_id"].values):
            data_galaxy_id = table_parameter[self.table_bh["galaxy_id"].values == galaxy_id]
            hist, _ = np.histogram(data_galaxy_id, bins = bins)
            hist_list.append(hist)
        
        hist_mean = np.mean(hist_list, axis = 0)
        hist_std = np.std(hist_list, ddof = 1, axis = 0)
        hist_mean_error = hist_std / np.sqrt(len(hist_list))

        return hist_mean, hist_mean_error


    def plot_dist_total_mean(self, path):
        parameters = ["m [M_solar]", "z_f", "d_GC [kpc]", "lat_GC [rad]", "long_GC [rad]", "d_Sun [kpc]", "lat_Sun [rad]", "long_Sun [rad]", "r_sp [pc]", "rho(r_sp) [GeV/cm3]", "gamma_sp", "gamma_c"]
        units = ["$M_{\odot}$", "", "kpc", "rad", "rad", "kpc", "rad", "rad", "pc", "GeV/cm$^3$", "", ""]
        log_parameters = ["m [M_solar]", "r_sp [pc]", "rho(r_sp) [GeV/cm3]"]
        # log_parameters = ["m [M_solar]"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]", "$r_\mathrm{sp}$ [pc]", r"$\rho(r_\mathrm{sp})$ [GeV/cm$^3$]", "$\gamma_\mathrm{sp}$", "$\gamma_\mathrm{c}$"]
        filenames = ["mass_mean", "redshift_mean", "distance_gc_mean", "latitude_GC_mean", "longitude_GC_mean", "distance_sun_mean", "latitude_Sun_mean", "longitude_Sun_mean", "r_sp_mean", "rho_sp_mean", "gamma_sp_mean", "gamma_c_mean"]
        
        for parameter, unit, x_label, filename in zip(parameters, units, x_labels, filenames):
            print(filename)
            data = self.table_bh[parameter].values
            data_sorted = np.sort(data)
            data_mean = np.mean(data)
            data_mean_error = np.std(data, ddof = 1) / np.sqrt(len(data))
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
            # label = f"$\mu = ({formatted_median} + {formatted_upper_error} - {formatted_lower_error}) \cdot 10^{exponent}$ {unit}"
            if exponent != 0:
                label = r"$\tilde{{\mu}} = ({0}^{{+{1}}}_{{-{2}}}) \cdot 10^{{ {3} }}$ {4}".format(formatted_median, formatted_upper_error, formatted_lower_error, exponent if exponent < 0 else " "+str(exponent), unit)
            else:
                label = r"$\tilde{{\mu}} = {0}^{{+{1}}}_{{-{2}}}$ {3}".format(formatted_median, formatted_upper_error, formatted_lower_error, unit)

            
            fig, ax = plt.subplots(figsize = single_column_fig_size)
            if parameter in log_parameters:
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), number_bins)
                # bins_centre = np.sqrt(bins[1:] * bins[:-1])
                # bin_width_left = bins_centre - bins[:-1]
                # bin_width_right = bins[1:] - bins_centre    
                # bins_width = np.vstack([bin_width_left, bin_width_right])
                # print("bins_width", bins_width)
                error_x_position = np.sqrt(bins[1:] * bins[:-1])
                ax.set_xscale("log")
                # if parameter == "m [M_solar]":
                ax.set_yscale("log")
            else:
                bins = np.linspace(np.min(data), np.max(data), number_bins)
                error_x_position = (bins[1:] + bins[:-1]) / 2

            bins_centre = (bins[1:] + bins[:-1]) / 2
            bins_width = bins[1:] - bins[:-1]
            
            hist_mean, hist_mean_error = self.parameter_distr_mean(parameter, bins)
            ax.bar(bins_centre, hist_mean, width = bins_width, color = color_darkblue, edgecolor = bar_edge_color, linewidth = bar_edge_width)
            ax.errorbar(error_x_position, hist_mean, yerr = hist_mean_error, color = color_lightblue, linestyle = "")
            # ax.vlines(data_mean, ymin = 0, ymax = np.max(hist_mean) + np.max(hist_mean_error), color = color_yellow, linestyle = "dashed", label = r"$\mu$ = {0:.2e} $\pm$ {1:.2e} {2}".format(data_mean, data_mean_error, unit))
            if parameter in log_parameters:
                ymin, ymax = ax.get_ylim()
                factor_ymax = 2
                ymin = np.min(hist_mean[hist_mean > 0]) * 0.5
                ymax = np.max(hist_mean[hist_mean > 0]) * factor_ymax
            else: 
                ymin, ymax = ax.get_ylim()
                factor_ymax = 1.
                ymax = ymax * factor_ymax
            
            ax.vlines(data_median, ymin = ymin, ymax = ymax, color = color_red, linestyle = "solid", label = label)
            ax.axvspan(lower_percentile, upper_percentile, alpha = 0.25, facecolor = color_red, edgecolor = "None")
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(x_label)
            ax.set_ylabel("$N_\mathrm{BH}$")
            if parameter in ["r_sp [pc]", "rho(r_sp) [GeV/cm3]"]:
                legend = plt.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc = "center", mode = "expand", borderaxespad = 0, ncol = 1, alignment = "center")
            else:
                legend = plt.legend(loc = "upper right")
            plt.tight_layout()
            if parameter == "m [M_solar]":
                # # Get the current x-axis tick locations and labels
                # tick_locations, tick_labels = plt.xticks()
                # print("tick_locations", tick_locations)
                # print("tick_labels", tick_labels)

                # # Use the first and last tick locations and labels
                # first_tick_location = tick_locations[0]
                # last_tick_location = tick_locations[-1]
                # first_tick_label = tick_labels[0]
                # last_tick_label = tick_labels[-1]

                # # Set only the first and last x-axis tick labels
                # plt.xticks([first_tick_location, last_tick_location], [first_tick_label, last_tick_label])
                plt.xticks([2e5, 3e5, 4e5, 6e5], [r'$2 \times 10^5$', "", "", r'$6 \times 10^5$']) # TODO: find a better way to do this, plt.xticks() does not work
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

            # if parameter == "lat_GC [rad]" or parameter == "lat_Sun [rad]":
            #     data = np.cos(data)
            #     bins = np.linspace(np.min(data), np.max(data), 9)
            #     hist_mean, hist_mean_error, hist_edges = self.parameter_distr_mean(parameter, bins)
            #     if parameter == "lat_GC [rad]":
            #         x_label = r"$\cos(b_\mathrm{GC})$"
            #         filename = "latitude_GC_cos_mean"
            #     if parameter == "lat_Sun [rad]":
            #         x_label = r"$\cos(b_\mathrm{Sun})$"
            #         filename = "latitude_Sun_cos_mean"
            #     plt.figure(figsize = single_column_fig_size)
            #     plt.errorbar(hist_edges[:-1], hist_mean, yerr = hist_mean_error, color = color_darkblue, linestyle = "", marker = ".")
            #     plt.xlabel(x_label)
            #     plt.ylabel("$N_\mathrm{BH}$")
            #     plt.tight_layout()
            #     plt.savefig(path + f"{filename}.pdf", dpi = 500)
            #     plt.close()

    def plot_number_dist(self, path):
        galaxy_ids, n_bh = np.unique(self.table_bh["galaxy_id"].values, return_counts = True)
        n_bh_median = np.median(n_bh)
        n_bh_sorted = np.sort(n_bh)
        lower_percentile = np.percentile(n_bh_sorted, 16)
        upper_percentile = np.percentile(n_bh_sorted, 84)
        n_bh_median_lower_error = n_bh_median - lower_percentile
        n_bh_median_upper_error = upper_percentile - n_bh_median
        hist, bins = np.histogram(n_bh, bins = number_bins)
        bins_width = bins[1:] - bins[:-1]
        bins_centre = (bins[1:] + bins[:-1]) / 2
        hist_err = np.sqrt(hist)

        # Format the values for display
        formatted_median = "{:.0f}".format(n_bh_median)
        formatted_upper_error = "{:.0f}".format(n_bh_median_upper_error)
        formatted_lower_error = "{:.0f}".format(n_bh_median_lower_error)
        
        label = r"$\tilde{{\mu}} = {0}^{{+{1}}}_{{-{2}}}$".format(formatted_median, formatted_upper_error, formatted_lower_error)

        plt.figure(figsize = single_column_fig_size)
        plt.bar(bins_centre, hist, width = bins_width, color = color_darkblue, yerr = hist_err, ecolor = color_lightblue, edgecolor = bar_edge_color, linewidth = bar_edge_width)
        # plt.vlines(n_mean, ymin = 0, ymax = np.max(hist) + np.max(hist_err), color = color_yellow, linestyle = "dashed", label = r"$\mu$ = {0:.2f} $\pm$ {1:.2f}".format(n_mean, n_mean_error))
        # plt.vlines(n_median, ymin = 0, ymax = np.max(hist) + np.max(hist_err), color = color_yellow, linestyle = "solid", label = r"median = {0:.2f}".format(n_median))
        ymin, ymax = plt.ylim()
        plt.vlines(n_bh_median, ymin = ymin, ymax = ymax, color = color_red, linestyle = "solid", label = label)
        plt.axvspan(lower_percentile, upper_percentile, alpha = 0.25, facecolor = color_red, edgecolor = "None")
        plt.xlabel(r"$N_\mathrm{BH}$")
        # plt.ylabel(r"Number of galaxies with $N_\mathrm{BH}$")
        plt.ylabel(r"$N_\mathrm{g}$")
        plt.ylim(ymin, ymax)
        # plt.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc = "lower left", mode = "expand", borderaxespad = 0, ncol = 1)
        plt.legend(loc = "upper right")
        plt.tight_layout()
        plt.savefig(path + "number_dist.pdf", dpi = 500)
        plt.close()


class FluxPlotter:
    def __init__(self, flux_catalogue):
        self.flux_catalogue = flux_catalogue

    @staticmethod
    def flux_thresholds(flux):
        flux_th = np.logspace(np.log10(np.min(flux.value)), np.log10(np.max(flux.value)), 30) * flux.unit
        return(flux_th)

    @staticmethod
    def integrated_luminosity(flux, flux_th):
        int_lum = []
        for threshold in flux_th:
            int_lum.append(len(flux[flux >= threshold]))
        return(int_lum)
    
    def integrated_luminosity_mean(self, flux_th):
        int_lum_list = []
        galaxy_ids = np.unique(self.flux_catalogue["galaxy_id"].values)
        for galaxy_id in galaxy_ids:
            flux_catalogue_id = self.flux_catalogue[self.flux_catalogue["galaxy_id"] == galaxy_id]
            flux_id = flux_catalogue_id["flux [cm-2 s-1]"].values * u.Unit("cm-2 s-1")
            int_lum_id = self.integrated_luminosity(flux_id, flux_th)
            int_lum_list.append(int_lum_id)
        int_lum_mean = np.mean(int_lum_list, axis = 0)
        int_lum_std = np.std(int_lum_list, ddof = 1, axis = 0)
        int_lum_mean_error = int_lum_std / np.sqrt(len(int_lum_list))
        # int_lum_error = np.sqrt(int_lum_mean) / np.sqrt(len(int_lum_list))
        # return(int_lum_mean, int_lum_error)
        return(int_lum_mean, int_lum_mean_error)
 
    def plot_integrated_luminosity(self, flux_th, m_dm, color):
        int_lum_mean, int_lum_error = self.integrated_luminosity_mean(flux_th)
        plt.errorbar(flux_th, int_lum_mean, yerr = int_lum_error, label = f"$m_{{\chi}}$ = {np.round(m_dm.to(u.TeV).value, 1)} TeV", linestyle = "", marker = ".", capsize = 3, color = color, markersize = 3)

    def plot_integrated_luminosity_comparison(self, flux_th, label, color):
        int_lum_mean, int_lum_error = self.integrated_luminosity_mean(flux_th)
        plt.errorbar(flux_th, int_lum_mean, yerr = int_lum_error, label = label, linestyle = "", marker = ".", capsize = 3, color = color, markersize = 3)

    def plot_cuttoff_radius_dist(self, path):
        r_cut = self.flux_catalogue["r_cut [pc]"].values
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

        bins = np.logspace(np.log10(np.min(r_cut)), np.log10(np.max(r_cut)), number_bins)
        bins_centre = (bins[1:] + bins[:-1]) / 2
        bins_width = bins[1:] - bins[:-1]

        error_x_position = np.sqrt(bins[1:] * bins[:-1])

        plt.figure(figsize = single_column_fig_size)
        hist_mean, hist_mean_error = parameter_distr_mean(table = self.flux_catalogue, parameter = "r_cut [pc]", bins = bins)
        plt.bar(bins_centre, hist_mean, width = bins_width, color = color_darkblue, edgecolor = bar_edge_color, linewidth = bar_edge_width)
        plt.errorbar(error_x_position, hist_mean, yerr = hist_mean_error, color = color_lightblue, linestyle = "")
        # plt.vlines(r_cut_mean, ymin = 0, ymax = np.max(hist_mean) + np.max(hist_mean_error), color = color_yellow, linestyle = "dashed", label = r"$\mu$ = {0:.2e} $\pm$ {1:.2e} {2}".format(r_cut_mean, r_cut_mean_error, "pc"))
        # plt.vlines(r_cut_median, ymin = 0, ymax = np.max(hist_mean) + np.max(hist_mean_error), color = color_yellow, linestyle = "solid", label = r"median = {0:.2e} {1}".format(r_cut_median, "pc"))
        ymin, ymax = plt.ylim()
        ymax = ymax * 2
        plt.vlines(r_cut_median, ymin = ymin, ymax = ymax, color = color_red, linestyle = "solid", label = label)
        plt.axvspan(lower_percentile, upper_percentile, alpha = 0.25, facecolor = color_red, edgecolor = "None")
        plt.ylim(1e-3, ymax)
        plt.xlabel("$r_\mathrm{cut}$ [pc]")
        plt.ylabel("$N_\mathrm{BH}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(bbox_to_anchor = (0, 1.02, 1, 0.2), loc = "lower left", mode = "expand", borderaxespad = 0, ncol = 1)
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()
        