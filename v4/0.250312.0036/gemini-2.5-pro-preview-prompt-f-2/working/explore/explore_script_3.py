# explore_script_3.py
# This script loads the NWB file and plots stimulus and response
# for channel 0 of sweeps 02, 03, and 04 to look for action potentials.
# It plots a subset of the data (first 1 second or full trace if shorter).

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

sweeps_to_check = ['02', '03', '04', '05'] # Check a few more
num_points_to_plot = 20000 # Max 1 second of data at 20000 Hz

for sweep_num_str in sweeps_to_check:
    stim_key = f'stimulus-{sweep_num_str}-ch-0'
    resp_key = f'current_clamp-response-{sweep_num_str}-ch-0'
    
    print(f"\nChecking sweep: {sweep_num_str}")

    if stim_key in nwb.stimulus and resp_key in nwb.acquisition:
        stim_series = nwb.stimulus[stim_key]
        resp_series = nwb.acquisition[resp_key]

        # Determine number of points to plot for this sweep (max 1s)
        current_num_points = min(num_points_to_plot, stim_series.data.shape[0])
        
        stim_data = stim_series.data[:current_num_points] * stim_series.conversion
        resp_data = resp_series.data[:current_num_points] * resp_series.conversion
        
        time_vector = np.arange(current_num_points) / stim_series.rate

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Sweep {sweep_num_str}-ch-0 (First {current_num_points/stim_series.rate:.2f}s)", fontsize=16)

        # Stimulus
        axs[0].plot(time_vector, stim_data)
        axs[0].set_title(f'{stim_key} ({stim_series.description})')
        axs[0].set_ylabel(f'Stimulus ({stim_series.unit})')

        # Response
        axs[1].plot(time_vector, resp_data)
        axs[1].set_title(f'{resp_key} ({resp_series.description})')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel(f'Response ({resp_series.unit})')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_filename = f"explore/stim_resp_sweep{sweep_num_str}_ch0.png"
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close(fig)
    else:
        print(f"Could not find series for sweep {sweep_num_str}: {stim_key} or {resp_key}")

try:
    io.close()
except Exception as e:
    print(f"Error closing NWBHDF5IO: {e}")

print("\nExploration script 3 finished.")