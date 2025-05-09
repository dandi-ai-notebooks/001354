# explore_script_5.py
# Scan a wider range of sweeps for depolarizing stimuli on ch-0.
# If not found, check ch-1 for a few representative sweeps.
# Plot the first depolarizing stimulus and response found.

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

found_plot_worthy_sweep = False
plot_filename_gen = ""

# --- Check Channel 0 first with a wider range ---
print("\n--- Searching for depolarizing stimulus on Channel 0 ---")
# Sweep numbers in this file go up to 312. Let's check a good portion.
# The nwb-file-info implies 'stimulus-X-ch-Y' where X is the number.
sweep_indices_ch0 = list(range(10, 200)) 

for i in sweep_indices_ch0:
    sweep_num_str = str(i)
    stim_key = f'stimulus-{sweep_num_str}-ch-0'
    resp_key = f'current_clamp-response-{sweep_num_str}-ch-0'
    
    if stim_key in nwb.stimulus and resp_key in nwb.acquisition:
        # print(f"Checking ch-0 sweep: {sweep_num_str} (key: {stim_key})") # Reduce verbosity
        stim_series = nwb.stimulus[stim_key]
        check_points = min(1000, stim_series.data.shape[0])
        stim_data_chunk = stim_series.data[:check_points] * stim_series.conversion
        
        if np.any(stim_data_chunk > 1e-12): # Check for positive current (allowing for small noise around zero)
            print(f"Found depolarizing stimulus in {stim_key}!")
            found_plot_worthy_sweep = True
            
            resp_series = nwb.acquisition[resp_key]
            num_points_to_plot = min(20000, stim_series.data.shape[0]) # Plot up to 1s
            
            stim_data_plot = stim_series.data[:num_points_to_plot] * stim_series.conversion
            resp_data_plot = resp_series.data[:num_points_to_plot] * resp_series.conversion
            time_vector = np.arange(num_points_to_plot) / stim_series.rate

            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(f"Sweep {sweep_num_str}-ch-0 (Depolarizing Stimulus)", fontsize=16)
            axs[0].plot(time_vector, stim_data_plot)
            axs[0].set_title(f'{stim_key} ({stim_series.description})')
            axs[0].set_ylabel(f'Stimulus ({stim_series.unit})')
            axs[1].plot(time_vector, resp_data_plot)
            axs[1].set_title(f'{resp_key} ({resp_series.description})')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel(f'Response ({resp_series.unit})')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plot_filename_gen = f"explore/depolarizing_stim_resp_sweep{sweep_num_str}_ch0.png"
            plt.savefig(plot_filename_gen)
            print(f"Plot saved to {plot_filename_gen}")
            plt.close(fig)
            break 
    # else:
        # print(f"Series not found for ch-0 sweep {sweep_num_str}: {stim_key} or {resp_key}")


if not found_plot_worthy_sweep:
    print("\n--- No depolarizing stimulus found on Channel 0 in the extended range. ---")
    # Add search for Channel 1 if needed, but for now focus on ch-0 as primary.

if not found_plot_worthy_sweep:
    print("No suitable depolarizing stimulus found in the checked sweeps to generate a plot.")

try:
    io.close()
except Exception as e:
    print(f"Error closing NWBHDF5IO: {e}")

print("\nExploration script 5 finished.")