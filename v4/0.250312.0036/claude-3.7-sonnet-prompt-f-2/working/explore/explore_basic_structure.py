# Explore the basic structure of the NWB file and visualize some current clamp recordings
# This script examines the basic structure of the NWB file and plots a few sample recordings
# to understand the nature of the data and experimental conditions.

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

# Print basic metadata
print("NWB File Overview:")
print(f"Session ID: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Subject Sex: {nwb.subject.sex}")
print(f"Subject Age Reference: {nwb.subject.age__reference}")

# Look at the acquisition data structure
print("\nAcquisition Data:")
acquisition_keys = list(nwb.acquisition.keys())
print(f"Number of acquisition series: {len(acquisition_keys)}")
print(f"First 10 acquisition keys: {acquisition_keys[:10]}")

# Look at the stimulus data structure
print("\nStimulus Data:")
stimulus_keys = list(nwb.stimulus.keys())
print(f"Number of stimulus series: {len(stimulus_keys)}")
print(f"First 10 stimulus keys: {stimulus_keys[:10]}")

# Look at the metadata from DandiIcephysMetadata
print("\nDandi Icephys Metadata:")
dandi_metadata = nwb.lab_meta_data['DandiIcephysMetadata']
print(f"Cell ID: {dandi_metadata.cell_id}")
print(f"Slice ID: {dandi_metadata.slice_id}")
print(f"Targeted Layer: {dandi_metadata.targeted_layer}")

# Plot a few example current clamp responses
response_keys = [k for k in acquisition_keys if 'current_clamp-response' in k]
stimulus_keys = [k for k in nwb.stimulus.keys() if 'stimulus' in k]

# Let's plot the first 3 response and stimulus pairs
plt.figure(figsize=(15, 10))

for i in range(3):
    response_key = response_keys[i]
    stimulus_key = stimulus_keys[i]
    
    response = nwb.acquisition[response_key]
    stimulus = nwb.stimulus[stimulus_key]
    
    # Get a subset of data to plot (first 10000 points)
    time_subset = 10000
    time_array = np.arange(time_subset) / response.rate
    response_data = response.data[:time_subset] * response.conversion
    stimulus_data = stimulus.data[:time_subset] * stimulus.conversion
    
    # Plot response
    plt.subplot(3, 2, i*2 + 1)
    plt.plot(time_array, response_data)
    plt.title(f"Response: {response_key}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"Voltage ({response.unit})")
    
    # Plot stimulus
    plt.subplot(3, 2, i*2 + 2)
    plt.plot(time_array, stimulus_data * 1e12)  # Convert A to pA
    plt.title(f"Stimulus: {stimulus_key}")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (pA)")

plt.tight_layout()
plt.savefig("explore/basic_current_clamp_recordings.png")

# Plot response to different stimuli to compare
# Let's look at the first set of responses (channel 0) across different time periods
plt.figure(figsize=(15, 12))

for i in range(6):
    response_key = f"current_clamp-response-{i+1:02d}-ch-0"
    stimulus_key = f"stimulus-{i+1:02d}-ch-0"
    
    response = nwb.acquisition[response_key]
    stimulus = nwb.stimulus[stimulus_key]
    
    # Get a subset of data to plot
    time_subset = 20000  # 1 second worth of data
    time_array = np.arange(time_subset) / response.rate
    response_data = response.data[:time_subset] * response.conversion
    stimulus_data = stimulus.data[:time_subset] * stimulus.conversion
    
    # Plot response and stimulus together
    plt.subplot(6, 1, i+1)
    
    ax1 = plt.gca()
    line1 = ax1.plot(time_array, response_data, 'b-', label='Voltage')
    ax1.set_ylabel(f"Voltage ({response.unit})")
    
    ax2 = ax1.twinx()
    line2 = ax2.plot(time_array, stimulus_data * 1e12, 'r-', label='Current')
    ax2.set_ylabel("Current (pA)")
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)
    
    plt.title(f"Recording {i+1}: {response.starting_time}s (Response and Stimulus)")
    plt.xlabel("Time (s)")

plt.tight_layout()
plt.savefig("explore/compare_recordings_ch0.png")

# Let's also explore the full dataset for one recording to understand the time span
response_full = nwb.acquisition["current_clamp-response-01-ch-0"]
full_time_array = np.arange(len(response_full.data)) / response_full.rate
print(f"\nFull time span of a recording: {full_time_array[-1]} seconds")
print(f"Number of data points: {len(response_full.data)}")
print(f"Sampling rate: {response_full.rate} Hz")

# Plot the full first recording to see the overall pattern
plt.figure(figsize=(15, 5))
plt.plot(full_time_array, response_full.data[:] * response_full.conversion)
plt.title("Full Recording: current_clamp-response-01-ch-0")
plt.xlabel("Time (s)")
plt.ylabel(f"Voltage ({response_full.unit})")
plt.savefig("explore/full_recording_01.png")

# Print some info about the stimulus type and experiment
print("\nStimulus Information:")
print(f"Stimulus description: {nwb.stimulus['stimulus-01-ch-0'].description}")

# Check if we have sequential recordings table data
if hasattr(nwb, 'icephys_sequential_recordings'):
    seq_df = nwb.icephys_sequential_recordings.to_dataframe()
    print("\nSequential Recordings Table:")
    print(seq_df)
    
    if 'stimulus_type' in seq_df.columns:
        print(f"\nStimulus type: {seq_df['stimulus_type'].values}")

# Print information about the electrodes
if hasattr(nwb, 'icephys_electrodes'):
    print("\nIcephys Electrodes:")
    for key, electrode in nwb.icephys_electrodes.items():
        print(f"{key}: {electrode}")