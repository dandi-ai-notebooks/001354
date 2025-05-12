# Explore the relationship between stimulus and response in the PAGER recordings
# This script will examine how neurons respond to stimuli, particularly focusing on 
# the ramp patterns mentioned in the metadata.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up the output directory for plots
os.makedirs('explore', exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print the first few keys to verify format
acquisition_keys = list(nwb.acquisition.keys())
print("First 5 acquisition keys:", acquisition_keys[:5])

# Let's examine the relationship between stimulus and response for several recordings
# We'll look at recordings 1, 20, 50, 100, and 200 to get a spread (if available)
recording_indices = [1, 20, 50, 100]

plt.figure(figsize=(15, 12))

for i, idx in enumerate(recording_indices):
    # Get the response and stimulus data
    response_key = f"current_clamp-response-{idx:02d}-ch-0"
    stimulus_key = f"stimulus-{idx:02d}-ch-0"
    
    try:
        response = nwb.acquisition[response_key]
        stimulus = nwb.stimulus[stimulus_key]
    except KeyError:
        print(f"Skipping index {idx}, key not found")
        continue
    
    # Get the full data
    response_data = response.data[:] * response.conversion
    stimulus_data = stimulus.data[:] * stimulus.conversion * 1e12  # Convert to pA
    time_array = np.arange(len(response_data)) / response.rate
    
    # Plot response and stimulus together
    plt.subplot(len(recording_indices), 1, i+1)
    
    ax1 = plt.gca()
    line1 = ax1.plot(time_array, response_data, 'b-', label='Voltage')
    ax1.set_ylabel(f"Voltage ({response.unit})")
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(time_array, stimulus_data, 'r-', label='Current')
    ax2.set_ylabel("Current (pA)")
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title(f"Recording {idx} (Response and Stimulus)")
    plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig("explore/stimulus_response_comparison.png")

# Let's look at the action potential patterns more closely
# For recording #1, which had clear action potentials
plt.figure(figsize=(15, 10))

# Get recording 1 data
response_key = "current_clamp-response-01-ch-0"
stimulus_key = "stimulus-01-ch-0"
    
response = nwb.acquisition[response_key]
stimulus = nwb.stimulus[stimulus_key]

# Get the full data
response_data = response.data[:] * response.conversion
stimulus_data = stimulus.data[:] * stimulus.conversion * 1e12  # Convert to pA
time_array = np.arange(len(response_data)) / response.rate

# Plot full response and stimulus
plt.subplot(2, 1, 1)
ax1 = plt.gca()
line1 = ax1.plot(time_array, response_data, 'b-', label='Voltage')
ax1.set_ylabel(f"Voltage ({response.unit})")

ax2 = ax1.twinx()
line2 = ax2.plot(time_array, stimulus_data, 'r-', label='Current')
ax2.set_ylabel("Current (pA)")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Recording 1: Full Response and Stimulus")

# Plot zoomed in on action potentials (around 1.2-2.0 seconds)
plt.subplot(2, 1, 2)
t_start = 1.2
t_end = 2.0
mask = (time_array >= t_start) & (time_array <= t_end)

ax1 = plt.gca()
line1 = ax1.plot(time_array[mask], response_data[mask], 'b-', label='Voltage')
ax1.set_ylabel(f"Voltage ({response.unit})")

ax2 = ax1.twinx()
line2 = ax2.plot(time_array[mask], stimulus_data[mask], 'r-', label='Current')
ax2.set_ylabel("Current (pA)")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

plt.title("Recording 1: Zoomed in on Action Potentials")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig("explore/action_potentials.png")

# Let's compare stimulus shapes to see if there's variation
plt.figure(figsize=(15, 8))

# Use the correct format for indices
indices_to_check = [1, 50, 100, 150]
valid_indices = []
for i, idx in enumerate(indices_to_check):
    stimulus_key = f"stimulus-{idx:02d}-ch-0"
    
    try:
        stimulus = nwb.stimulus[stimulus_key]
        valid_indices.append(idx)
    except KeyError:
        print(f"Skipping stimulus {idx}, key not found")
        continue
    
    if len(valid_indices) > 6:  # Limit to 6 plots
        break

for i, idx in enumerate(valid_indices[:6]):  # Use only valid indices, limit to 6
    stimulus_key = f"stimulus-{idx:02d}-ch-0"
    stimulus = nwb.stimulus[stimulus_key]
    stimulus_data = stimulus.data[:] * stimulus.conversion * 1e12  # Convert to pA
    time_array = np.arange(len(stimulus_data)) / stimulus.rate
    
    plt.subplot(2, 3, i+1)
    plt.plot(time_array, stimulus_data, 'r-')
    plt.title(f"Stimulus {idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (pA)")

plt.tight_layout()
plt.savefig("explore/stimulus_comparisons.png")

# Let's examine the change in neuronal response after prolonged stimulation
# Compare the first few and last few recordings
plt.figure(figsize=(15, 12))

# First, early responses
plt.subplot(2, 1, 1)
for idx in range(1, 6):
    response_key = f"current_clamp-response-{idx:02d}-ch-0"
    
    try:
        response = nwb.acquisition[response_key]
    except KeyError:
        print(f"Skipping response {idx}, key not found")
        continue
        
    response_data = response.data[:] * response.conversion
    time_array = np.arange(len(response_data)) / response.rate
    
    plt.plot(time_array, response_data, label=f"Recording {idx}")

plt.title("Early Responses (1-5)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Then, later responses
plt.subplot(2, 1, 2)
# Determine the highest recording index
acquisition_keys = list(nwb.acquisition.keys())
response_keys = [k for k in acquisition_keys if 'current_clamp-response' in k and '-ch-0' in k]
# Extract indices
indices = []
for key in response_keys:
    parts = key.split("-")
    if len(parts) > 2 and parts[2].isdigit():
        indices.append(int(parts[2]))

if indices:
    max_idx = max(indices)
    # Use the 5 highest available indices under 312 (since we saw many high indices in the metadata)
    last_indices = sorted([idx for idx in indices if idx <= 312])[-5:]
    
    for idx in last_indices:
        response_key = f"current_clamp-response-{idx:02d}-ch-0"
        
        try:
            response = nwb.acquisition[response_key]
        except KeyError:
            print(f"Skipping response {idx}, key not found")
            continue
            
        response_data = response.data[:] * response.conversion
        time_array = np.arange(len(response_data)) / response.rate
        
        plt.plot(time_array, response_data, label=f"Recording {idx}")

    plt.title(f"Late Responses ({last_indices[0]}-{last_indices[-1]})")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()

plt.tight_layout()
plt.savefig("explore/early_vs_late_responses.png")

# Print information about the stimulus patterns
print("\nStimulus Analysis:")
for idx in [1, 20, 50, 100]:
    stimulus_key = f"stimulus-{idx:02d}-ch-0"
    
    try:
        stimulus = nwb.stimulus[stimulus_key]
    except KeyError:
        print(f"Skipping stimulus {idx}, key not found")
        continue
        
    stimulus_data = stimulus.data[:] * stimulus.conversion * 1e12  # Convert to pA
    
    min_val = np.min(stimulus_data)
    max_val = np.max(stimulus_data)
    mean_val = np.mean(stimulus_data)
    
    print(f"Stimulus {idx}:")
    print(f"  Min: {min_val:.2f} pA")
    print(f"  Max: {max_val:.2f} pA")
    print(f"  Mean: {mean_val:.2f} pA")
    print(f"  Description: {stimulus.description}")