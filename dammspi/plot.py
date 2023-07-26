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
from dammspi.utils import nfw_profile, nfw_integral, M_bh_2
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.stats import gaussian_kde

# set matplotlib parameters for nice looking plots
plt.rcParams.update({'font.size': 8}) # 8 (paper), 10 (poster)
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman')#, weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'
cm_conversion_factor = 1/2.54  # centimeters in inches
single_column_fig_size = (8.85679 * cm_conversion_factor, 8.85679 * 3/4 * cm_conversion_factor)
single_column_fig_size_legend = (8.85679 * cm_conversion_factor, 8.85679 * 3/4 * 7/6 * cm_conversion_factor)
double_column_fig_size = (18.34621 * cm_conversion_factor, 18.34621 * 3/4 * cm_conversion_factor)
double_column_squeezed_fig_size = (18.34621 * cm_conversion_factor, 18.34621 * 1/2 * cm_conversion_factor)
markersize = 4

cmap = LinearSegmentedColormap.from_list("", ['#FFF275', '#E83151', "#003049", "#171B1D"])
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
            plt.ylabel("Number of BHs")
            plt.tight_layout()
            plt.savefig(path + f"{filename}.pdf", dpi = 500)
            plt.close()

    @staticmethod
    def plot_nfw(r, rho, rho_0, r_s, galaxy_mass, z, path):
        lablel_fit = f"Fit\n$\\rho_0$ = {rho_0.value:.2e} M$_{{\\odot}}$ / kpc$^3$\n$r_s$ = {r_s.value:.2f} kpc"
        plt.figure(figsize = single_column_fig_size)
        plt.title(f"M$_{{halo}}$ = {galaxy_mass.value:.2e} M$_{{\\odot}}$, z = {z:.2f}")
        plt.plot(r, rho, color = color_black, label = "Data")
        plt.plot(r, nfw_profile((rho_0, r_s), r), color = color_red, linestyle = "dashed", label = lablel_fit)
        plt.xlabel(r"$r$ [kpc]")
        plt.ylabel(r"$\rho$ [M$_{\odot}$ / kpc$^3$]")
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.legend()
        plt.savefig(path + f"nfw.pdf", dpi = 300)
        # plt.show()
        plt.close()

    @staticmethod
    def plot_radius_gravitational_influence(r_h, rho_0, r_s, M_bh, galaxy_mass, z, path):
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
        plt.savefig(path + f"r_h.pdf", dpi = 300)
        # plt.show()
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
            plt.ylabel("Number of BHs")
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
                plt.ylabel("Number of BHs")
                plt.tight_layout()
                plt.savefig(path + f"{filename}.pdf", dpi = 500)
                plt.close()

    def plot_2d_map(self, lat, long, path):
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')

        fig = plt.figure(figsize  = single_column_fig_size)
        ax = fig.add_subplot(111, projection='aitoff')
        ax.grid(True, alpha = 0.5)
        ax.scatter(coords.l.wrap_at('180d').radian, coords.b.radian, color = color_darkblue, marker = 'o', s = markersize)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()


    def plot_2d_map_contours(self, lat, long, path):
        # Extract coordinates
        coords = SkyCoord(long, lat, frame='galactic', unit='rad')

        # Calculate kernel density estimation
        kde = gaussian_kde([coords.l.wrap_at('180d').radian, coords.b.radian])

        # Define grid for evaluating KDE
        x = np.linspace(-np.pi, np.pi, 100)
        y = np.linspace(-np.pi / 2, np.pi / 2, 100)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Evaluate KDE at grid points
        Z = np.reshape(kde(positions).T, X.shape)

        # Plot the coordinates using Aitoff projection
        fig = plt.figure(figsize  = double_column_fig_size)
        ax = fig.add_subplot(111, projection='aitoff')
        ax.grid(True, alpha = 0.5)

        # Plot the density contours
        levels = np.linspace(Z.min(), Z.max(), 10)
        cont = ax.contour(X, Y, Z, levels=levels, cmap = cmap)

        # Add colorbar
        cbar = fig.colorbar(cont)

        # Plot the points
        ax.scatter(coords.l.wrap_at('180d').radian, coords.b.radian, color = color_darkblue, marker='o', s = markersize)

        # Set labels and title
        ax.set_xlabel('Galactic Longitude')
        ax.set_ylabel('Galactic Latitude')
        plt.tight_layout()
        plt.savefig(path, dpi = 500)

    def plot_cumulative_radial_distribution(self, distance, path):
        plt.figure(figsize = single_column_fig_size)
        plt.hist(distance, bins = 20, cumulative = True, density = True, color = color_darkblue)
        plt.xlabel(r"$d_\mathrm{GC}$ [kpc]")
        plt.ylabel(r"$N(<r) / N_\mathrm{tot}$")
        plt.tight_layout()
        plt.savefig(path, dpi = 500)
        plt.close()

    def cumulative_radial_distribution(self, distance, bins):
        cumulative_hist = []
        for bin in bins:
            cumulative_hist.append(np.sum(distance < bin) / len(distance))
        return cumulative_hist

    def plot_cumulative_radial_distribution_mean(self, path):
        distance = self.table_bh["d_GC [kpc]"].values
        d_min, d_max = np.min(distance), np.max(distance)
        bins = np.logspace(np.log10(d_min), np.log10(d_max), 20)

        cumulative_hist_list = []
        for galaxy_id in np.unique(self.table_bh["galaxy_id"].values):
            distance_galaxy_id = distance[self.table_bh["galaxy_id"].values == galaxy_id]
            cumulative_hist = self.cumulative_radial_distribution(distance_galaxy_id, bins)
            cumulative_hist_list.append(cumulative_hist)
        
        cumulative_hist_mean = np.mean(cumulative_hist_list, axis = 0)
        cumulative_hist_std = np.std(cumulative_hist_list, ddof = 1, axis = 0)
        cumulative_hist_mean_error = cumulative_hist_std / np.sqrt(len(cumulative_hist_list))

        plt.figure(figsize = single_column_fig_size)
        plt.errorbar(bins, cumulative_hist_mean, yerr = cumulative_hist_mean_error, color = color_darkblue, linestyle = "", marker = ".")
        plt.xlabel(r"$d_\mathrm{GC}$ [kpc]")
        plt.ylabel(r"$N(<r) / N_\mathrm{tot}$")
        plt.xscale("log")
        plt.yscale("log")
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
        parameters = ["m [M_solar]", "z_f", "d_GC [kpc]", "lat_GC [rad]", "long_GC [rad]", "d_Sun [kpc]", "lat_Sun [rad]", "long_Sun [rad]", "r_sp [pc]", "rho(r_sp) [GeV/cm3]"]
        units = ["$M_{\odot}$", "", "kpc", "rad", "rad", "kpc", "rad", "rad", "pc", "GeV/cm$^3$"]
        log_parameters = ["m [M_solar]", "r_sp [pc]", "rho(r_sp) [GeV/cm3]"]
        x_labels = [r"$m_\mathrm{BH}$ [$M_{\odot}$]", r"$z_f$", r"$d_\mathrm{GC}$ [kpc]", r"$b_\mathrm{GC}$ [rad]", r"$l_\mathrm{GC}$ [rad]", r"$d_\mathrm{Sun}$ [kpc]", r"$b_\mathrm{Sun}$ [rad]", r"$l_\mathrm{Sun}$ [rad]", "r_sp [pc]", r"$\rho(r_\mathrm{sp})$ [GeV/cm$^3$]"]
        filenames = ["mass_mean", "redshift_mean", "distance_gc_mean", "latitude_GC_mean", "longitude_GC_mean", "distance_sun_mean", "latitude_Sun_mean", "longitude_Sun_mean", "r_sp_mean", "rho_sp_mean"]
        
        for parameter, unit, x_label, filename in zip(parameters, units, x_labels, filenames):
            data = self.table_bh[parameter].values
            data_mean = np.mean(data)
            data_mean_error = np.std(data, ddof = 1) / np.sqrt(len(data))
            data_median = np.median(data)
            
            plt.figure(figsize = single_column_fig_size)
            if parameter in log_parameters:
                bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 9)
                # bins_centre = np.sqrt(bins[1:] * bins[:-1])
                # bin_width_left = bins_centre - bins[:-1]
                # bin_width_right = bins[1:] - bins_centre    
                # bins_width = np.vstack([bin_width_left, bin_width_right])
                # print("bins_width", bins_width)
                error_x_position = np.sqrt(bins[1:] * bins[:-1])
                plt.xscale("log")
                plt.yscale("log")
            else:
                bins = np.linspace(np.min(data), np.max(data), 9)
                error_x_position = (bins[1:] + bins[:-1]) / 2

            bins_centre = (bins[1:] + bins[:-1]) / 2
            bins_width = bins[1:] - bins[:-1]
            
            hist_mean, hist_mean_error = self.parameter_distr_mean(parameter, bins)
            plt.bar(bins_centre, hist_mean, width = bins_width, color = color_darkblue)
            plt.errorbar(error_x_position, hist_mean, yerr = hist_mean_error, color = color_lightblue, linestyle = "")
            plt.vlines(data_mean, ymin = 0, ymax = np.max(hist_mean) + np.max(hist_mean_error), color = color_yellow, linestyle = "dashed", label = r"$\mu$ = {0:.2e} $\pm$ {1:.2e} {2}".format(data_mean, data_mean_error, unit))
            plt.vlines(data_median, ymin = 0, ymax = np.max(hist_mean) + np.max(hist_mean_error), color = color_yellow, linestyle = "solid", label = r"median = {0:.2e} {1}".format(data_median, unit))
            plt.xlabel(x_label)
            plt.ylabel("Number of BHs")
            plt.legend()
            plt.tight_layout()
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
            #     plt.ylabel("Number of BHs")
            #     plt.tight_layout()
            #     plt.savefig(path + f"{filename}.pdf", dpi = 500)
            #     plt.close()

    def plot_number_dist(self, path):
        n = np.unique(self.table_bh["galaxy_id"].values, return_counts = True)[1]
        hist, bins_edges = np.histogram(n, bins = 10)
        bins_width = bins_edges[1:] - bins_edges[:-1]
        bins_centre = (bins_edges[1:] + bins_edges[:-1]) / 2
        hist_err = np.sqrt(hist)
        n_mean = np.mean(n)
        n_median = np.median(n)
        n_mean_error = np.std(n, ddof = 1) / np.sqrt(len(n))
        plt.figure(figsize = single_column_fig_size)
        plt.bar(bins_centre, hist, width = bins_width, color = color_darkblue, yerr = hist_err, ecolor = color_lightblue)
        plt.vlines(n_mean, ymin = 0, ymax = np.max(hist) + np.max(hist_err), color = color_yellow, linestyle = "dashed", label = r"$\mu$ = {0:.2f} $\pm$ {1:.2f}".format(n_mean, n_mean_error))
        plt.vlines(n_median, ymin = 0, ymax = np.max(hist) + np.max(hist_err), color = color_yellow, linestyle = "solid", label = r"median = {0:.2f}".format(n_median))
        plt.xlabel(r"$N_\mathrm{BH}$")
        plt.ylabel(r"Number of galaxies with $N_\mathrm{BH}$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path + "number_dist.pdf", dpi = 500)
        plt.close()



