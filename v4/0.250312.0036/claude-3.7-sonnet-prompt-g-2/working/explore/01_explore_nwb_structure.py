#!/usr/bin/env python
# This script explores the structure of the NWB file to understand the data organization

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots directory if it doesn't exist
os.makedirs("explore", exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the NWB file
print(f"NWB file identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject: {nwb.subject.subject_id}, Sex: {nwb.subject.sex}, Species: {nwb.subject.species}, Age reference: {nwb.subject.age__reference}")
print(f"Lab metadata - Cell ID: {nwb.lab_meta_data['DandiIcephysMetadata'].cell_id}")
print(f"Lab metadata - Slice ID: {nwb.lab_meta_data['DandiIcephysMetadata'].slice_id}")
print(f"Lab metadata - Targeted Layer: {nwb.lab_meta_data['DandiIcephysMetadata'].targeted_layer}")

# Get the number of recordings
print(f"\nNumber of intracellular recordings: {len(nwb.intracellular_recordings)}")
print(f"Number of simultaneous recordings: {len(nwb.icephys_simultaneous_recordings)}")
print(f"Number of sequential recordings: {len(nwb.icephys_sequential_recordings)}")

# Print the stimulus types
if hasattr(nwb.icephys_sequential_recordings, 'stimulus_type'):
    print(f"\nStimulus types: {nwb.icephys_sequential_recordings.stimulus_type[:]}")

# Look at what acquisition data is available
print("\nAcquisition data:")
acquisition_keys = list(nwb.acquisition.keys())
print(f"Number of acquisition series: {len(acquisition_keys)}")
print(f"First 5 acquisition keys: {acquisition_keys[:5]}")

# Look at what stimulus data is available
print("\nStimulus data:")
stimulus_keys = list(nwb.stimulus.keys())
print(f"Number of stimulus series: {len(stimulus_keys)}")
print(f"First 5 stimulus keys: {stimulus_keys[:5]}")

# Examine the structure of one response and one stimulus
print("\nExample response data structure:")
example_response = nwb.acquisition[acquisition_keys[0]]
print(f"Type: {type(example_response)}")
print(f"Description: {example_response.description}")
print(f"Starting time: {example_response.starting_time}")
print(f"Rate: {example_response.rate}")
print(f"Unit: {example_response.unit}")
print(f"Data shape: {example_response.data.shape}")
print(f"Conversion: {example_response.conversion}")

print("\nExample stimulus data structure:")
example_stimulus = nwb.stimulus[stimulus_keys[0]]
print(f"Type: {type(example_stimulus)}")
print(f"Description: {example_stimulus.description}")
print(f"Starting time: {example_stimulus.starting_time}")
print(f"Rate: {example_stimulus.rate}")
print(f"Unit: {example_stimulus.unit}")
print(f"Data shape: {example_stimulus.data.shape}")
print(f"Conversion: {example_stimulus.conversion}")

# Plot example stimulus and response data
plt.figure(figsize=(12, 6))

# Get a subset of the data to avoid loading too much
sample_size = 10000  # 0.5 seconds at 20kHz

# Plot stimulus
stimulus_data = example_stimulus.data[:sample_size]
stimulus_time = np.arange(len(stimulus_data)) / example_stimulus.rate
plt.subplot(2, 1, 1)
plt.plot(stimulus_time, stimulus_data * example_stimulus.conversion * 1e12)  # convert to pA
plt.title(f"Stimulus: {example_stimulus.description}")
plt.ylabel("Current (pA)")
plt.grid(True)

# Plot response
response_data = example_response.data[:sample_size]
response_time = np.arange(len(response_data)) / example_response.rate
plt.subplot(2, 1, 2)
plt.plot(response_time, response_data * example_response.conversion * 1e3)  # convert to mV
plt.title(f"Response: {example_response.description}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.tight_layout()
plt.savefig("explore/nwb_data_example.png")
print("Plot saved to explore/nwb_data_example.png")

# Print electrode information
print("\nElectrode information:")
for electrode_id, electrode in nwb.icephys_electrodes.items():
    print(f"Electrode {electrode_id}:")
    print(f"  Description: {electrode.description}")
    print(f"  Device: {electrode.device}")

# Plot histogram of data values for a response to check distribution
plt.figure(figsize=(8, 4))
plt.hist(response_data * example_response.conversion * 1e3, bins=50, alpha=0.7)
plt.title("Histogram of Response Values")
plt.xlabel("Voltage (mV)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.savefig("explore/response_histogram.png")
print("Histogram saved to explore/response_histogram.png")