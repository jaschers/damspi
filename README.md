# DAMSPI
<img src="https://github.com/jaschers/damspi/blob/main/logo/logo_text.png" width="750">

DArk Matter SPIkes (DAMSPI) is a fully Python-based software for the analysis of dark matter spikes around Intermediate Mass Black Holes (IMBHs) in the Milky Way. It allows to extract an IMBH mock catalogue and their corresponding dark matter spike parameters from the EAGLE simulations in order to probe a potential gamma-ray signal from dark matter self-annihilation. For more details, see our published article at TBA. The IMBH catalogue can be downloaded from TBA.

# Installation
Clone the repository to your prefered directory:
```
git clone git@github.com:jaschers/damspi.git
```
Create the conda environment and install the required packages:
```
conda create -n damspi python=3.9.15
```
```
conda activate damspi
```
```
pip install -r requirements.txt
```
Download the dark matter spectra from gammapy via:
```
gammapy download datasets
```
Add the following line to your ``.bashrc`` file:
```
export GAMMAPY_DATA=gammapy-datasets/1.0
```


# Usage
The IMBH catalogue is available at TBA and provides the IMBH coordinates, dark matter spike parameters and the expected gamma-ray flux for a variety of dark matter masses and dark matter cross sections. If you would like to calculate the gamma-ray flux for a different dark matter mass or cross section, you have two options:
1. Follow the instructions below to create the IMBH catalogue completely from scratch. This requires to download the EAGLE data, which, depending on the EAGLE dataset you choose, can take several days.
2. Download the ``catalogue_nfw.h5`` file from TBA, put it into the ``catalogue/RefL0100N1504/imbh/`` directory and use the ``calculate_flux.py`` script to calculate the gamma-ray flux for your desired dark matter masses or cross sections. See Section "Gamma-ray flux from dark matter self-annihilation" below for more details.

Every script has a help option ``-h`` or ``--help`` in order to get basic instructions on how to use the script. Some details will be discussed in the following.


# Download EAGLE data
First, an account on VirgoDB needs to be requested to get access to the EAGLE data, [here](https://virgodb.dur.ac.uk/). After sucessfully receiving your username and password, add the credentials to your ``.bashrc`` file: 
```
export VIRGODB_USERNAME=<your_username>
export VIRGODB_PASSWORD=<your_password>
```
Then, the EAGLE particle data can be downloaded with the following command:
```
python scripts/download_data.py
```
By default, this will download all the 29 redshift snapshots available for the reference dataset ``RefL0100N1504`` with a box size of 100 cMpc. The data is stored in the HDF5 format in the ``data`` directory. Only the black hole particle data, i.e. ``PartType5``, is downloaded to save storage space. The download may take a while, depending on your internet connection and the specific EAGLE dataset you choose. Other dataset can be downloaded by specifying the ``--sim_name`` (``-sn``) and ``--number_files`` (``-nf``) option:
```
python scripts/download_data.py -sn RefL0100N1504 -nf 256
```
The ``--number_files`` option describes the number of files that are present for each snapshot and are fixed by the EAGLE simulation. E.g. ``RefL0100N1504``, ``RefL0050N0752`` and ``RefL0025N0376`` have a number of 256, 128 and 16 files per snapshot, respectively.

# IMBH catalogue
The IMBH catalogue for a given EAGLE dataset can be extracted with the following command:
```
python scripts/create_catalogue.py -n nfw
```
By default, this will create the IMBH catalogue for the reference dataset ``RefL0100N1504``. The catalogue is stored in the HDF5 format in ``catalogue/<sim_name>/imbh/catalogue_<name>.h5``. It is required to provide the ``--name`` (``-n``) option, which specifies the suffix of the output filenames. Additional options are:
* ``--dark_matter_profile`` (``-dmp``): Dark matter profile to be used for the dark matter spike calculation. Options are ``nfw`` (default) and ``cored``.
* if ``-dmp cored`` is chosen, the core index ``--core_index`` (``-ci``) can be specified. Default is ``None``, which means that the the best fit core index is determined from the EAGLE data. Otherwise, the core index is fixed to the specified value.
* ``--load_temporary_catalogue`` (``-ltc``): If this option is specified, the IMBH catalogue is loaded from the temporary catalogue file ``catalogue/<sim_name>/imbh_temp/catalogue_temp_<name>.h5``. This is useful if the catalogue has already been created and the dark matter spike parameters should be extracted again with different options. In this case, the ``--name`` option should be the same as for the temporary catalogue file.
* ``--plot`` (``-pl``): If this option is specified, some cross-check plots are extracted and stored in ``plots/<sim_name>/galaxy_id_<galaxy_id>/black_holes/coordinates``, ``plots/<sim_name>/galaxy_id_<galaxy_id>/black_holes/distributions`` and ``plots/sim_name/galaxy_id_<galaxy_id>/black_holes/spikes/id_<bh_id>``. This option should be treated with great care, since it will create a lot of plots and therefore requires a lot of storage space. The plots are not required for the analysis and are only used for cross-checks.
* ``--save_animation`` (``-sa``): If this option is specified, an animation IMBH coordinates is stored in ``plots/<sim_name>/galaxy_id_<galaxy_id>/black_holes/coordinates``. Again, this option should be treated with great care, since it will create a lot of animations and therefore requires a lot of storage space. The animations are not required for the analysis and are only used for cross-checks.

The catalogue contains the following information for each IMBH:

| Field           | Unit         | Description                                                                                                                                                                  |
|-----------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `galaxy_id`     | -            | Unique identifier of a galaxy                                                                                                                                                |
| `bh_id`         | -            | Unique identifier of a black hole                                                                                                                                            |
| `m`             | M☉           | Black hole mass                                                                                                                                                              |
| `z_f`           | -            | Black hole formation redshift                                                                                                                                                |
| `z_c`           | -            | Closest redshift for which a snapshot in the EAGLE simulations is available (see text for details)                                                                            |
| `nsnap_c`       | -            | Closest snapshot for which a snapshot in the EAGLE simulations is available (see text for details)                                                                            |
| `d_GC`          | kpc          | Distance of the black hole to the centre of potential of the host galaxy                                                                                                     |
| `lat_GC`        | rad          | Galactic latitude of the black hole with the centre of potential of the host galaxy being the origin of the coordinate system                                                |
| `long_GC`       | rad          | Galactic longitude of the black hole with the centre of potential of the host galaxy being the origin of the coordinate system                                               |
| `d_Sun`         | kpc          | Distance of the black hole to the Sun                                                                                                                                        |
| `lat_Sun`       | rad          | Galactic latitude of the black hole with the Sun being the origin of the coordinate system                                                                                   |
| `long_Sun`      | rad          | Galactic longitude of the black hole with the Sun being the origin of the coordinate system                                                                                  |
| `gamma_sp`      | -            | Spike index                                                                                                                                                                  |
| `r_sp`          | pc           | Spike radius                                                                                                                                                                 |
| `rho(r_sp)`     | GeV/cm³      | Dark matter density at the spike radius                                                                                                                                      |
| `satellite`     | -            | `True` if the black hole is located in one of the satellite galaxies of the host galaxy                                                                                      |
| `no_host`       | -            | `True` if black hole was not assigned to any host at its formation redshift                                                                                                  |
| `r_c`           | -            | Core radius (only available for cored dark matter profile)                                                                                                                   |

## Create plots for IMBH catalogue
A bunch of usueful plots for the IMBH catalogue can be created with the following command:
```
python scripts/plot_catalogue.py -n nfw
```
The ``--name`` option should be the same as for the IMBH catalogue. The plots are stored in ``plots/<sim_name>/black_hole_dist/<name>``. The following plots are created:
* black hole number distribution
* 2D map of the black holes in galactic coordinates
* latitude and longitude distribution of the black holes
* cumulative radial distribution of the black holes
* distance distribution of the black holes to the Sun and to the galactic centre
* mass distribution of the black holes
* redshift distribution of the black holes
* core index distribution of the dark matter profile
* dark matter spike distribution
* spike radius distribution
* dark matter density at the spike radius distribution

# Gamma-ray flux from dark matter self-annihilation
The expected gamma-ray flux from dark matter self-annihilation around IMBHs can be calculated with the following command:
```
python scripts/calculate_flux.py -n nfw
```
The ``--name`` option should specifies for which catalogue the flux should be calculated. The flux catalogues are stored in the HDF5 format in ``catalogue/<sim_name>/flux/<name>/<channel>_channel/m_dm_<mass>GeV.h5``. By default, this will calculate the flux for a dark matter mass of 500, 1000 and 1500 GeV and a cross section of $3 \times 10^{-26}$ cm3/s assuming the b-channel.
The following options are available:
* ``--m_dm`` (``-mdm``): Mass of dark matter particle in GeV. Can be single input or mass range + number of masses (three inputs). If mass range is given, scaling can be specified by the mass_dm_scaling argument. Default: 500.
* ``--mass_dm_scaling`` (``-mdms``): Scaling of dark matter particle mass. Can be linear or log. Default: linear.
* ``--sigma_v`` (``-sv``):  Dark matter (velocity weighted) annihilation cross section in cm3/s. Default: 3e-26
* ``--channel`` (``-c``): Dark matter annihilation channel. Can be: 'V->e', 'V->mu', 'V->tau', 'W', 'WL', 'WT', 'Z', 'ZL', 'ZT', 'b', 'c', 'e', 'eL', 'eR', 'g', 'gamma', 'h', 'mu', 'muL', 'muR', 'nu_e', 'nu_mu', 'nu_tau', 'q', 't', 'tau', 'tauL', 'tauR'. Default: b.
* ``--E_th`` (``-eth``): Lower energy threshold to calculate number of gamma rays per dark matter annihilation in GeV. Default: 100
* ``--plot`` (``-pl``): Bool if plots are saved [y, n], default: n. If y, plots are saved in ``plots/<sim_name>/flux/<name>/<channel>_channel/``.

The following example extracts flux catalogues based on the ``catalogue/<sim_name>/imbh/catalogue_<name>.h5`` file for dark matter masses of 1000, 1500 and 2000 GeV for a cross section of $10^{28}$ cm3/s assuming the tau channel:
```
python scripts/calculate_flux.py -n nfw -mdm 100 2000 3 -mdms linear -sv 1e-28 -c tau
```
If you would like to calculate the flux for a single dark matter mass, you can simply specify the mass as a single input:
```
python scripts/calculate_flux.py -n nfw -mdm 1000 -mdms linear -sv 1e-28 -c tau
```
The flux catalogue contains the following information for each IMBH:

Each catalogue is extracted for a designated dark matter mass.

| Field       | Unit               | Description                                                                                         |
|-------------|--------------------|-----------------------------------------------------------------------------------------------------|
| `galaxy_id` | -                  | Unique identifier of a galaxy                                                                       |
| `bh_id`     | -                  | Unique identifier of a black hole                                                                   |
| `sigma_v`   | cm³/s              | Dark matter cross section times the relative velocity                                               |
| `r_cut`     | pc                 | Cutoff radius                                                                                       |
| `flux`      | 1/cm²/s            | Gamma-ray flux                                                                                      |

# Other useful scripts

## Halo dark matter profile
The dark matter profile of an example IMBH formation galaxy can be extracted with the following command:
```
python scripts/plot_halo_profile.py -n nfw
```
It will show the dark matter density versus the distance to the galactic centre and the best fit NFW profile and cored profile. The plots are stored in ``plots/<sim_name>/dm_profiles/``. It requires that the ``catalogue_temp/<sim_name>/catalogue_temp_<name>.h5`` is available.

## Spike dark matter profile
The dark matter profile of an typical IMBH can be plotted with the following command:
```
python scripts/plot_spike_profile.py
```
It uses the dark matter spike parameters from the ``config/config.yaml`` file. The plots are stored in ``plots/<sim_name>/dm_profiles/``.

## Comparison of gamma-ray fluxes for different dark matter profiles
The gamma-ray flux under the assumption of different dark matter profiles can be compared with the following command:
```
plot_flux_comparison.py -n nfw cored -mdm 500 -sv 1e-28 -c tau -l nfw cored
```
It is mandotary to run the ``create_catalogue.py`` and ``calculate_flux.py`` scripts for the different dark matter profiles before running this script. The plots are stored in ``plots/<name>/flux/comparison/<channel>_channel/``.
