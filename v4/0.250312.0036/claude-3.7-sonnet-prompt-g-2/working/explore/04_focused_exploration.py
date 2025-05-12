#!/usr/bin/env python
# This script provides a focused exploration of the data for creating the notebook

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a function to load an NWB file
def load_nwb_file(asset_id):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    print(f"Loading NWB file from {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    return io.read(), url

# Let's list the subjects and select a few to examine
subjects = {
    "sub-PK-109": "8609ffee-a79e-498c-8dfa-da46cffef135",  # First file we examined
    "sub-PK-110": "fb5d0a75-4e94-4174-a8b3-538cb88ff72c",  # Another subject
    "sub-PK-113": "46b31d08-c72a-4fef-aac7-032d4ca9530c"   # Third subject
}

# Examine basic information about each subject
for subject_id, asset_id in subjects.items():
    nwb, url = load_nwb_file(asset_id)
    
    print(f"\nSubject: {subject_id}")
    print(f"NWB file identifier: {nwb.identifier}")
    print(f"Session start time: {nwb.session_start_time}")
    print(f"Subject info: Sex={nwb.subject.sex}, Species={nwb.subject.species}")
    if hasattr(nwb.lab_meta_data['DandiIcephysMetadata'], 'targeted_layer'):
        print(f"Targeted layer: {nwb.lab_meta_data['DandiIcephysMetadata'].targeted_layer}")
    
    # Look at the experimental structure
    print(f"Number of intracellular recordings: {len(nwb.intracellular_recordings)}")
    print(f"Number of simultaneous recordings: {len(nwb.icephys_simultaneous_recordings)}")
    print(f"Number of sequential recordings: {len(nwb.icephys_sequential_recordings)}")
    
    # Get an example stimulus-response pair
    acquisition_keys = list(nwb.acquisition.keys())
    stimulus_keys = list(nwb.stimulus.keys())
    
    if len(acquisition_keys) > 0 and len(stimulus_keys) > 0:
        # Choose an interesting response (e.g., around the middle of the recording)
        idx = min(100, len(acquisition_keys) - 1) 
        response_key = acquisition_keys[idx]
        stimulus_key = stimulus_keys[idx]
        
        response = nwb.acquisition[response_key]
        stimulus = nwb.stimulus[stimulus_key]
        
        # Check sample sizes and determine how much to load
        sample_size = min(20000, response.data.shape[0])
        
        # Get the data
        response_data = response.data[:sample_size]
        stimulus_data = stimulus.data[:sample_size]
        
        # Convert to more convenient units
        response_data_mv = response_data * response.conversion * 1e3  # convert to mV
        stimulus_data_pa = stimulus_data * stimulus.conversion * 1e12  # convert to pA
        
        # Create time array
        time = np.arange(sample_size) / response.rate
        
        # Plot stimulus and response
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(time, stimulus_data_pa)
        plt.title(f"{subject_id}: Stimulus ({stimulus.description})")
        plt.ylabel("Current (pA)")
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, response_data_mv)
        plt.title(f"{subject_id}: Response ({response.description})")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"explore/{subject_id}_example.png")
        plt.close()
        
        print(f"Example plot saved to explore/{subject_id}_example.png")
        
        # Calculate basic statistics for the response
        baseline = np.median(response_data_mv[:int(sample_size/10)])  # first 10% as baseline
        mean_val = np.mean(response_data_mv)
        std_val = np.std(response_data_mv)
        min_val = np.min(response_data_mv)
        max_val = np.max(response_data_mv)
        range_val = max_val - min_val
        
        print(f"Response statistics:")
        print(f"  Baseline: {baseline:.2f} mV")
        print(f"  Mean: {mean_val:.2f} mV")
        print(f"  Std: {std_val:.2f} mV")
        print(f"  Range: {range_val:.2f} mV ({min_val:.2f} to {max_val:.2f} mV)")
        
        # Save URL for neurosift link
        with open(f"explore/{subject_id}_neurosift_url.txt", "w") as f:
            neurosift_url = f"https://neurosift.app/nwb?url={url}&dandisetId=001354&dandisetVersion=0.250312.0036"
            f.write(neurosift_url)
            print(f"Neurosift URL: {neurosift_url}")
    else:
        print("No acquisition or stimulus data found")

# For one subject, examine multiple trials to understand response patterns
subject_id = "sub-PK-109"
asset_id = subjects[subject_id]
nwb, _ = load_nwb_file(asset_id)

acquisition_keys = list(nwb.acquisition.keys())
stimulus_keys = list(nwb.stimulus.keys())

# Sample trials across the recording
sample_indices = [10, 50, 100, 150, 200]
sample_indices = [i for i in sample_indices if i < len(acquisition_keys)]

plt.figure(figsize=(12, 8))
for i, idx in enumerate(sample_indices):
    response_key = acquisition_keys[idx]
    stimulus_key = stimulus_keys[idx]
    
    response = nwb.acquisition[response_key]
    stimulus = nwb.stimulus[stimulus_key]
    
    sample_size = min(20000, response.data.shape[0])
    
    # Get the data
    response_data = response.data[:sample_size]
    stimulus_data = stimulus.data[:sample_size]
    
    # Convert to convenient units
    response_data_mv = response_data * response.conversion * 1e3  # convert to mV
    stimulus_data_pa = stimulus_data * stimulus.conversion * 1e12  # convert to pA
    
    # Create time array
    time = np.arange(sample_size) / response.rate
    
    # Plot in a grid
    plt.subplot(len(sample_indices), 2, 2*i+1)
    plt.plot(time, stimulus_data_pa)
    plt.title(f"Trial {idx}: Stimulus")
    plt.ylabel("Current (pA)")
    plt.grid(True)
    
    plt.subplot(len(sample_indices), 2, 2*i+2)
    plt.plot(time, response_data_mv)
    plt.title(f"Trial {idx}: Response")
    plt.ylabel("Voltage (mV)")
    plt.grid(True)

plt.tight_layout()
plt.savefig(f"explore/{subject_id}_multiple_trials.png")
plt.close()
print(f"\nMultiple trials plot saved to explore/{subject_id}_multiple_trials.png")

# Examine some metadata about the experiment
print("\nAdditional metadata from NWB file:")
for subject_id, asset_id in subjects.items():
    nwb, _ = load_nwb_file(asset_id)
    print(f"\nSubject: {subject_id}")
    
    # Get information about recording electrodes
    if hasattr(nwb, 'icephys_electrodes'):
        print("Recording electrodes:")
        for electrode_id, electrode in nwb.icephys_electrodes.items():
            print(f"  Electrode {electrode_id}: {electrode.description}")
    
    # Check if there are any notes or descriptions about PAGER receptor
    if hasattr(nwb, 'notes'):
        print(f"Notes: {nwb.notes}")
    
    # Check for any available hardware or other experimental details
    if hasattr(nwb.subject, 'description'):
        print(f"Subject description: {nwb.subject.description}")