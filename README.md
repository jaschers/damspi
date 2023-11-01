# DAMSPI
DArk Matter SPIkes (DAMSPI) is a python package for the analysis of dark matter spikes around Intermediate Mass Black Holes (IMBHs) in the Milky Way. It allows to extract an IMBH catalogue and their corresponding dark matter spikes parameters from the EAGLE simulations in order to probe a potential gamma-ray signal from dark matter self-annihilation. For more details, see our published article at TBA

# Installation
```
git clone git@github.com:jaschers/dammspi.git
```
```
conda env create -f environment.yml
```
```
conda activate damspi
```
```
pip install dampsi
```

# Usage
Every script has a help option ``-h`` or ``--help`` in order to get basic instructions on how to use the script. Some details will be discussed in the following.

# Download data
First, an account on VirgoDB needs to be requested to get access to the EAGLE data, [here](https://virgodb.dur.ac.uk/). After sucessfully receiving your username and password, open the ``config/config.yaml`` file and enter your credentials under ``User_input``. Then, the EAGLE particle data can be downloaded with the following command:
```
python download_data.py
```
By default, this will download all the 29 redshift snapshots available for the reference dataset ``RefL0100N1504`` with a box size of 100 cMpc. The data is stored in the HDF5 format in the ``data`` directory. Only the black hole particle data, i.e. ``PartType5``, is downloaded to save storage space. The download may take a while, depending on your internet connection and the specific EAGLE dataset you choose. Other dataset can be downloaded by specifying the ``--sim_name`` (``-sn``) and ``--number_files`` (``-nf``) option:
```
python download_data.py -sn RefL0050N0752 -nf 128
```
The ``--number_files`` option describes the number of files that are present for each snapshot and are fixed by the EAGLE simulation. E.g. ``RefL0100N1504``, ``RefL0050N0752`` and ``RefL0025N0376`` have a number of 256, 128 and 16 files per snapshot, respectively.

# Create IMBH catalogue
The IMBH catalogue for a given EAGLE dataset can be extracted with the following command:
```
python create_catalogue.py
```
By default, this will create the IMBH catalogue for the reference dataset ``RefL0100N1504``. The catalogue is stored in the HDF5 format in the ``catalogues`` directory. The catalogue contains the following information for each IMBH:
