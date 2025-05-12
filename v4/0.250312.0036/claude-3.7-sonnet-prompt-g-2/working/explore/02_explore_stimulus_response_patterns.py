#!/usr/bin/env python
# This script explores the relationship between stimulus and response in the NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the acquisition and stimulus keys
acquisition_keys = list(nwb.acquisition.keys())
stimulus_keys = list(nwb.stimulus.keys())

# Look at multiple pairs of stimulus-response to better understand the patterns
def examine_stimulus_response_pair(index, sample_size=20000):
    """Examine a specific stimulus-response pair"""
    # Get the stimulus and response data
    resp_key = acquisition_keys[index]
    stim_key = stimulus_keys[index]
    
    print(f"\nExamining pair {index}: Response: {resp_key}, Stimulus: {stim_key}")
    
    response = nwb.acquisition[resp_key]
    stimulus = nwb.stimulus[stim_key]
    
    # Print information about this pair
    print(f"  Response description: {response.description}")
    print(f"  Response starting time: {response.starting_time} seconds")
    print(f"  Stimulus description: {stimulus.description}")
    print(f"  Stimulus starting time: {stimulus.starting_time} seconds")
    
    # Get the data
    response_data = response.data[:sample_size]
    stimulus_data = stimulus.data[:sample_size]
    
    # Convert to appropriate units
    response_data_mv = response_data * response.conversion * 1e3  # convert to mV
    stimulus_data_pa = stimulus_data * stimulus.conversion * 1e12  # convert to pA
    
    # Create time arrays
    time = np.arange(sample_size) / response.rate
    
    # Plot the data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot stimulus
    ax1.plot(time, stimulus_data_pa)
    ax1.set_title(f"Stimulus: {stimulus.description}")
    ax1.set_ylabel("Current (pA)")
    ax1.grid(True)
    
    # Plot response
    ax2.plot(time, response_data_mv)
    ax2.set_title(f"Response: {response.description}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Voltage (mV)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"explore/pair_{index}_data.png")
    
    return stimulus_data_pa, response_data_mv, time

# Examine a few different pairs
pairs_to_examine = [0, 10, 20, 50, 100]
results = []

for idx in pairs_to_examine:
    stimulus, response, time = examine_stimulus_response_pair(idx)
    results.append((stimulus, response, time, idx))

# Create a plot comparing different stimulus patterns
plt.figure(figsize=(12, 8))

# Plot different stimuli
plt.subplot(2, 1, 1)
for stimulus, _, time, idx in results:
    plt.plot(time, stimulus, label=f"Stim {idx}")
plt.title("Comparison of Different Stimuli")
plt.ylabel("Current (pA)")
plt.grid(True)
plt.legend()

# Plot different responses
plt.subplot(2, 1, 2)
for _, response, time, idx in results:
    plt.plot(time, response, label=f"Resp {idx}")
plt.title("Corresponding Responses")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("explore/stimulus_response_comparison.png")
print("\nComparison plot saved to explore/stimulus_response_comparison.png")

# Create a more detailed view of one stimulus-response pair
# with zoomed in sections to see the details
detailed_idx = 50  # Choose a representative example
stimulus, response, time = examine_stimulus_response_pair(detailed_idx, sample_size=100000)

# Find regions of interest - look for the largest response changes
response_diff = np.abs(np.diff(response))
interest_points = np.argsort(response_diff)[-5:]  # Top 5 points with largest changes

fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 2]})

# Plot full stimulus
axes[0].plot(time, stimulus)
axes[0].set_title(f"Detailed Stimulus (Full 5 seconds)")
axes[0].set_ylabel("Current (pA)")
axes[0].grid(True)

# Plot full response
axes[1].plot(time, response)
axes[1].set_title(f"Detailed Response (Full 5 seconds)")
axes[1].set_ylabel("Voltage (mV)")
axes[1].grid(True)

# Plot zoomed in region of interest
# Find a region with interesting activity
# Look for the largest change in the response
max_diff_idx = np.argmax(response_diff)
zoom_start = max(0, max_diff_idx - 2000)
zoom_end = min(len(time), max_diff_idx + 8000)

axes[2].plot(time[zoom_start:zoom_end], response[zoom_start:zoom_end])
axes[2].set_title(f"Zoomed Response - Region of Interest")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Voltage (mV)")
axes[2].grid(True)

# Add a marker for the point of maximum change
axes[2].axvline(x=time[max_diff_idx], color='r', linestyle='--', 
               label=f"Max change at t={time[max_diff_idx]:.3f}s")
axes[2].legend()

plt.tight_layout()
plt.savefig("explore/detailed_stimulus_response.png")
print("\nDetailed plot saved to explore/detailed_stimulus_response.png")

# Calculate basic statistics for the responses
print("\nResponse Statistics:")
for _, response, _, idx in results:
    mean = np.mean(response)
    std = np.std(response)
    min_val = np.min(response)
    max_val = np.max(response)
    range_val = max_val - min_val
    print(f"Pair {idx}: Mean = {mean:.2f} mV, Std = {std:.2f} mV, Range = {range_val:.2f} mV ({min_val:.2f} to {max_val:.2f} mV)")