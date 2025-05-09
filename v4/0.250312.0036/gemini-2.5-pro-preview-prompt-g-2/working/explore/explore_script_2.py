# explore_script_2.py
# This script loads the NWB file and plots the second current clamp response
# (current_clamp-response-02-ch-0) and its corresponding stimulus
# (stimulus-02-ch-0).

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the second current clamp response and stimulus
response_series_name = "current_clamp-response-02-ch-0"
stimulus_series_name = "stimulus-02-ch-0"

if response_series_name not in nwb.acquisition:
    print(f"Response series {response_series_name} not found in acquisition.")
    exit()
if stimulus_series_name not in nwb.stimulus:
    print(f"Stimulus series {stimulus_series_name} not found in stimulus.")
    exit()

response_series = nwb.acquisition[response_series_name]
stimulus_series = nwb.stimulus[stimulus_series_name]

# Get data (load a subset)
num_points_to_plot = 20000  # Plot 1 second of data
response_data = response_series.data[:num_points_to_plot]
stimulus_data = stimulus_series.data[:num_points_to_plot]

# Create timestamps
response_timestamps = np.arange(len(response_data)) / response_series.rate + response_series.starting_time
stimulus_timestamps = np.arange(len(stimulus_data)) / stimulus_series.rate + stimulus_series.starting_time

# Create plot
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

# Plot response
axes[0].plot(response_timestamps, response_data)
axes[0].set_title(f"Response: {response_series.name}")
axes[0].set_ylabel(f"Voltage ({response_series.unit})")

# Plot stimulus
axes[1].plot(stimulus_timestamps, stimulus_data)
axes[1].set_title(f"Stimulus: {stimulus_series.name}")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel(f"Current ({stimulus_series.unit})")

plt.tight_layout()
plt.savefig("response_stimulus_plot_2.png")

print(f"Plotted {response_series_name} and {stimulus_series_name} to response_stimulus_plot_2.png")

io.close()