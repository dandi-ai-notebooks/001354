#!/usr/bin/env python
# This script analyzes the response of CA1 neurons to PAGER receptor activation

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Create a function to load an NWB file
def load_nwb_file(asset_id):
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    print(f"Loading NWB file from {url}")
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    return io.read()

# Define a function to analyze response characteristics
def analyze_response(nwb, response_key, stimulus_key, sample_size=20000):
    """Analyze response characteristics for a given stimulus-response pair"""
    response = nwb.acquisition[response_key]
    stimulus = nwb.stimulus[stimulus_key]
    
    # Get the data
    response_data = response.data[:sample_size]
    stimulus_data = stimulus.data[:sample_size]
    
    # Convert to appropriate units
    response_data_mv = response_data * response.conversion * 1e3  # convert to mV
    stimulus_data_pa = stimulus_data * stimulus.conversion * 1e12  # convert to pA
    
    # Create time array
    time = np.arange(sample_size) / response.rate
    
    # Calculate response characteristics
    baseline = np.median(response_data_mv[:int(sample_size/10)])  # first 10% as baseline
    peak_depolarization = np.max(response_data_mv)
    peak_hyperpolarization = np.min(response_data_mv)
    response_range = peak_depolarization - peak_hyperpolarization
    
    # Find if there are action potentials (simplified check for spikes)
    # We'll consider a rapid rise over a threshold as an action potential
    diff_data = np.diff(response_data_mv)
    threshold = 10  # mV/sample threshold for spike detection
    spike_indices = np.where(diff_data > threshold)[0]
    spike_count = len(spike_indices)
    
    # Calculate metrics for the stimulus
    stim_amplitude = np.max(stimulus_data_pa) - np.min(stimulus_data_pa)
    
    return {
        'baseline_mv': baseline,
        'peak_depolarization_mv': peak_depolarization,
        'peak_hyperpolarization_mv': peak_hyperpolarization,
        'response_range_mv': response_range,
        'spike_count': spike_count,
        'stimulus_amplitude_pa': stim_amplitude,
        'time': time,
        'response_data': response_data_mv,
        'stimulus_data': stimulus_data_pa,
    }

# Load the first NWB file (from previous explorations)
asset_id = "8609ffee-a79e-498c-8dfa-da46cffef135"  # sub-PK-109
nwb = load_nwb_file(asset_id)

# Print basic info about the recording
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Cell ID: {nwb.lab_meta_data['DandiIcephysMetadata'].cell_id}")
print(f"Targeted layer: {nwb.lab_meta_data['DandiIcephysMetadata'].targeted_layer}")

# Get the acquisition and stimulus keys
acquisition_keys = list(nwb.acquisition.keys())
stimulus_keys = list(nwb.stimulus.keys())

# Analyze multiple response-stimulus pairs
num_pairs_to_analyze = 10
results = []

for i in range(0, min(len(acquisition_keys), 200), 20):  # Sample pairs from throughout the recording
    if i // 2 >= len(acquisition_keys):
        break
        
    response_key = acquisition_keys[i]
    stimulus_key = stimulus_keys[i]
    
    print(f"\nAnalyzing pair {i}: Response: {response_key}, Stimulus: {stimulus_key}")
    
    result = analyze_response(nwb, response_key, stimulus_key)
    results.append({
        'index': i,
        'response_key': response_key,
        'stimulus_key': stimulus_key,
        'analysis': result
    })
    
    # Plot the stimulus-response pair
    plt.figure(figsize=(8, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(result['time'], result['stimulus_data'])
    plt.title(f"Stimulus {i}")
    plt.ylabel("Current (pA)")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(result['time'], result['response_data'])
    plt.title(f"Response {i} (Baseline: {result['baseline_mv']:.2f} mV, Spikes: {result['spike_count']})")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"explore/pair_{i}_analysis.png")
    plt.close()

# Create a summary plot of response characteristics
plt.figure(figsize=(12, 8))

# Extract features for plotting
indices = [r['index'] for r in results]
baselines = [r['analysis']['baseline_mv'] for r in results]
peak_depol = [r['analysis']['peak_depolarization_mv'] for r in results]
peak_hyperpol = [r['analysis']['peak_hyperpolarization_mv'] for r in results]
response_ranges = [r['analysis']['response_range_mv'] for r in results]
spike_counts = [r['analysis']['spike_count'] for r in results]
stim_amplitudes = [r['analysis']['stimulus_amplitude_pa'] for r in results]

# Create plots
plt.subplot(3, 2, 1)
plt.plot(indices, baselines, 'o-')
plt.title("Baseline Membrane Potential")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(indices, peak_depol, 'o-')
plt.title("Peak Depolarization")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(indices, peak_hyperpol, 'o-')
plt.title("Peak Hyperpolarization")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(indices, response_ranges, 'o-')
plt.title("Response Range")
plt.ylabel("Voltage (mV)")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(indices, spike_counts, 'o-')
plt.title("Spike Count")
plt.xlabel("Trial Index")
plt.ylabel("Count")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(indices, stim_amplitudes, 'o-')
plt.title("Stimulus Amplitude")
plt.xlabel("Trial Index")
plt.ylabel("Current (pA)")
plt.grid(True)

plt.tight_layout()
plt.savefig("explore/response_characteristics_summary.png")
print("Summary plot saved to explore/response_characteristics_summary.png")

# Try loading and analyzing a different cell to compare responses
# Using one of the other subjects: sub-PK-110
asset_id_2 = "fb5d0a75-4e94-4174-a8b3-538cb88ff72c"  # sub-PK-110
try:
    nwb2 = load_nwb_file(asset_id_2)
    
    print(f"\nComparing with second cell:")
    print(f"Subject ID: {nwb2.subject.subject_id}")
    print(f"Cell ID: {nwb2.lab_meta_data['DandiIcephysMetadata'].cell_id}")
    print(f"Targeted layer: {nwb2.lab_meta_data['DandiIcephysMetadata'].targeted_layer}")
    
    # Get the acquisition and stimulus keys
    acquisition_keys2 = list(nwb2.acquisition.keys())
    stimulus_keys2 = list(nwb2.stimulus.keys())
    
    # Analyze a few pairs from the second cell
    results2 = []
    for i in range(0, min(len(acquisition_keys2), 100), 20):
        if i >= len(acquisition_keys2) or i >= len(stimulus_keys2):
            break
            
        response_key = acquisition_keys2[i]
        stimulus_key = stimulus_keys2[i]
        
        print(f"\nAnalyzing pair {i} from second cell: Response: {response_key}, Stimulus: {stimulus_key}")
        
        result = analyze_response(nwb2, response_key, stimulus_key)
        results2.append({
            'index': i,
            'response_key': response_key,
            'stimulus_key': stimulus_key,
            'analysis': result
        })
    
    # Create a comparison plot of the two cells
    plt.figure(figsize=(12, 6))
    
    # Compare baselines
    plt.subplot(1, 2, 1)
    plt.plot([r['index'] for r in results], [r['analysis']['baseline_mv'] for r in results], 'o-', label=f"Cell 1 ({nwb.subject.subject_id})")
    plt.plot([r['index'] for r in results2], [r['analysis']['baseline_mv'] for r in results2], 'o-', label=f"Cell 2 ({nwb2.subject.subject_id})")
    plt.title("Baseline Membrane Potential Comparison")
    plt.xlabel("Trial Index")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    
    # Compare response ranges
    plt.subplot(1, 2, 2)
    plt.plot([r['index'] for r in results], [r['analysis']['response_range_mv'] for r in results], 'o-', label=f"Cell 1 ({nwb.subject.subject_id})")
    plt.plot([r['index'] for r in results2], [r['analysis']['response_range_mv'] for r in results2], 'o-', label=f"Cell 2 ({nwb2.subject.subject_id})")
    plt.title("Response Range Comparison")
    plt.xlabel("Trial Index")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("explore/cell_comparison.png")
    print("Cell comparison plot saved to explore/cell_comparison.png")
    
except Exception as e:
    print(f"Error analyzing second cell: {e}")
    print("Continuing with analysis of just the first cell")

# Print a statistical summary of the responses
print("\nStatistical Summary of Responses:")
print(f"Mean baseline: {np.mean(baselines):.2f} ± {np.std(baselines):.2f} mV")
print(f"Mean peak depolarization: {np.mean(peak_depol):.2f} ± {np.std(peak_depol):.2f} mV")
print(f"Mean peak hyperpolarization: {np.mean(peak_hyperpol):.2f} ± {np.std(peak_hyperpol):.2f} mV")
print(f"Mean response range: {np.mean(response_ranges):.2f} ± {np.std(response_ranges):.2f} mV")
print(f"Mean spike count: {np.mean(spike_counts):.2f} ± {np.std(spike_counts):.2f}")