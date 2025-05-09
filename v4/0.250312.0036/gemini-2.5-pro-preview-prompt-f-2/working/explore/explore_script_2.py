# explore_script_2.py
# This script loads the NWB file and plots the first stimulus
# and its corresponding response for both channels (ch-0 and ch-1)
# for the first available sweep (e.g., 'stimulus-01-ch-0' and 'current_clamp-response-01-ch-0').
# It plots a subset of the data (first 1 second).

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apply seaborn theme
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()
print("NWB file loaded successfully.")

# --- Data extraction and plotting for sweep 01, channels 0 and 1 ---
stim_key_ch0 = 'stimulus-01-ch-0'
resp_key_ch0 = 'current_clamp-response-01-ch-0'
stim_key_ch1 = 'stimulus-01-ch-1'
resp_key_ch1 = 'current_clamp-response-01-ch-1'

num_points_to_plot = 20000 # 1 second of data at 20000 Hz

if stim_key_ch0 in nwb.stimulus and resp_key_ch0 in nwb.acquisition and \
   stim_key_ch1 in nwb.stimulus and resp_key_ch1 in nwb.acquisition:

    stim_series_ch0 = nwb.stimulus[stim_key_ch0]
    resp_series_ch0 = nwb.acquisition[resp_key_ch0]
    stim_series_ch1 = nwb.stimulus[stim_key_ch1]
    resp_series_ch1 = nwb.acquisition[resp_key_ch1]

    # Load data subsets
    stim_data_ch0 = stim_series_ch0.data[:num_points_to_plot]
    resp_data_ch0 = resp_series_ch0.data[:num_points_to_plot]
    stim_data_ch1 = stim_series_ch1.data[:num_points_to_plot]
    resp_data_ch1 = resp_series_ch1.data[:num_points_to_plot]
    
    # Time vector
    time_vector = np.arange(num_points_to_plot) / stim_series_ch0.rate

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Stimulus and Response for Sweep 01 (First {num_points_to_plot/stim_series_ch0.rate:.1f}s)", fontsize=16)

    # Channel 0 Stimulus
    axs[0, 0].plot(time_vector, stim_data_ch0)
    axs[0, 0].set_title(f'{stim_key_ch0} ({stim_series_ch0.description})')
    axs[0, 0].set_ylabel(f'Stimulus ({stim_series_ch0.unit})')

    # Channel 0 Response
    axs[1, 0].plot(time_vector, resp_data_ch0)
    axs[1, 0].set_title(f'{resp_key_ch0} ({resp_series_ch0.description})')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel(f'Response ({resp_series_ch0.unit})')
    
    # Channel 1 Stimulus
    axs[0, 1].plot(time_vector, stim_data_ch1)
    axs[0, 1].set_title(f'{stim_key_ch1} ({stim_series_ch1.description})')
    axs[0, 1].set_ylabel(f'Stimulus ({stim_series_ch1.unit})')

    # Channel 1 Response
    axs[1, 1].plot(time_vector, resp_data_ch1)
    axs[1, 1].set_title(f'{resp_key_ch1} ({resp_series_ch1.description})')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel(f'Response ({resp_series_ch1.unit})')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plot_filename = "explore/stim_resp_sweep01_channels_0_1.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)
else:
    print(f"Could not find all required series for sweep 01: {stim_key_ch0}, {resp_key_ch0}, {stim_key_ch1}, {resp_key_ch1}")

try:
    io.close()
except Exception as e:
    print(f"Error closing NWBHDF5IO: {e}")

print("Exploration script 2 finished.")