# explore_script_1.py
# This script loads the NWB file and prints the keys available in
# nwb.acquisition and nwb.stimulus. It also prints the shape of the
# first data series in each to get an idea of the data size.

import pynwb
import h5py
import remfile
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Add mode='r'
nwb = io.read()

print("\nNWB file loaded successfully.")

print("\nKeys in nwb.acquisition:")
acquisition_keys = list(nwb.acquisition.keys())
print(acquisition_keys)
if acquisition_keys:
    first_acq_key = acquisition_keys[0]
    first_acq_series = nwb.acquisition[first_acq_key]
    print(f"\nShape of data in '{first_acq_key}': {first_acq_series.data.shape}")
    print(f"Description of '{first_acq_key}': {first_acq_series.description}")
    print(f"Unit of '{first_acq_key}': {first_acq_series.unit}")
    print(f"Rate of '{first_acq_key}': {first_acq_series.rate} Hz")
    print(f"Starting time of '{first_acq_key}': {first_acq_series.starting_time} {first_acq_series.starting_time_unit}")

print("\nKeys in nwb.stimulus:")
stimulus_keys = list(nwb.stimulus.keys())
print(stimulus_keys)
if stimulus_keys:
    first_stim_key = stimulus_keys[0]
    first_stim_series = nwb.stimulus[first_stim_key]
    print(f"\nShape of data in '{first_stim_key}': {first_stim_series.data.shape}")
    print(f"Description of '{first_stim_key}': {first_stim_series.description}")
    print(f"Unit of '{first_stim_key}': {first_stim_series.unit}")
    print(f"Rate of '{first_stim_key}': {first_stim_series.rate} Hz")
    print(f"Starting time of '{first_stim_key}': {first_stim_series.starting_time} {first_stim_series.starting_time_unit}")

# It's good practice to close the IO object, though not strictly necessary for remfile
# as it doesn't hold system resources in the same way a local file would.
# However, PyNWB might have its own internal state.
try:
    io.close()
except Exception as e:
    print(f"Error closing NWBHDF5IO: {e}")

# h5_file and remote_file will be closed when they go out of scope
# or by the garbage collector. For remfile, explicit close is not critical.
print("\nExploration script finished.")