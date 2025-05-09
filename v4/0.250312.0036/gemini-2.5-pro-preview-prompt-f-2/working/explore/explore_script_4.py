# explore_script_4.py
# Scan later sweeps (e.g., 50-60) for depolarizing stimuli on ch-0
# and plot the first one found.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
print(f"Loading NWB file from: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()
print("NWB file loaded successfully.")

found_depolarizing_stim = False
# Let's check a range of sweeps, e.g., from 10 up to 312 (max observed in nwb-file-info)
# We'll check some specific ones that might be different, like around common current steps.
# Based on output of nwb-file-info, there are many series like current_clamp-response-XXX-ch-Y
# up to XXX=312. Let's try some in the middle and some higher ones.
# For example, sweeps often come in families.
sweep_indices_to_try = list(range(10, 30)) + list(range(50,70)) + list(range(100, 120))


for i in sweep_indices_to_try:
    sweep_num_str = f"{i:02d}" # Format as two digits, e.g., 01, 10, etc.
    # The tool output shows series like 'current_clamp-response-10-ch-0' not 'current_clamp-response-010-ch-0'
    # The naming seems to be 'current_clamp-response-X-ch-Y' where X is the number without leading zeros for X < 10,
    # and with leading zeros if needed for consistent naming by some tools, or just the number.
    # Let's try to match based on the nwb_file_info output format precisely. Sweep numbers are 1-312.
    sweep_num_str_nwb_format = str(i) # Original number as string

    stim_key = f'stimulus-{sweep_num_str_nwb_format}-ch-0'
    resp_key = f'current_clamp-response-{sweep_num_str_nwb_format}-ch-0'
    
    print(f"\nChecking sweep: {sweep_num_str_nwb_format} (key: {stim_key})")

    if stim_key in nwb.stimulus and resp_key in nwb.acquisition:
        stim_series = nwb.stimulus[stim_key]
        
        # Load a small chunk of stimulus data to check for depolarizing current
        # Check first 1000 points, or fewer if series is shorter
        check_points = min(1000, stim_series.data.shape[0])
        stim_data_chunk = stim_series.data[:check_points] * stim_series.conversion
        
        if np.any(stim_data_chunk > 0):
            print(f"Found depolarizing stimulus in {stim_key}!")
            found_depolarizing_stim = True
            
            resp_series = nwb.acquisition[resp_key]
            
            num_points_to_plot = min(20000, stim_series.data.shape[0]) # Plot up to 1s
            
            stim_data_plot = stim_series.data[:num_points_to_plot] * stim_series.conversion
            resp_data_plot = resp_series.data[:num_points_to_plot] * resp_series.conversion
            time_vector = np.arange(num_points_to_plot) / stim_series.rate

            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(f"Sweep {sweep_num_str_nwb_format}-ch-0 (Depolarizing Stimulus)", fontsize=16)

            axs[0].plot(time_vector, stim_data_plot)
            axs[0].set_title(f'{stim_key} ({stim_series.description})')
            axs[0].set_ylabel(f'Stimulus ({stim_series.unit})')

            axs[1].plot(time_vector, resp_data_plot)
            axs[1].set_title(f'{resp_key} ({resp_series.description})')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel(f'Response ({resp_series.unit})')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plot_filename = f"explore/depolarizing_stim_resp_sweep{sweep_num_str_nwb_format}_ch0.png"
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            plt.close(fig)
            break # Stop after finding and plotting the first one
    else:
        # print(f"Series not found for sweep {sweep_num_str_nwb_format}: {stim_key} or {resp_key}")
        pass # Reduce noise, many keys won't exist in this sparse search

if not found_depolarizing_stim:
    print("No depolarizing stimulus found in the checked sweeps for ch-0.")

try:
    io.close()
except Exception as e:
    print(f"Error closing NWBHDF5IO: {e}")

print("\nExploration script 4 finished.")