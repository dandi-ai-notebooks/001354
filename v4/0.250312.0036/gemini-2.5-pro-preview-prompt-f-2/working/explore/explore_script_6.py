# explore_script_6.py
# Scan remaining sweeps on ch-0 (200-312), then check ch-1 for depolarizing stimuli.
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
plotted_channel = -1
plotted_sweep_num = ""

# --- Check Channel 0 first for remaining sweeps ---
print("\n--- Searching for depolarizing stimulus on Channel 0 (sweeps 200-312) ---")
sweep_indices_ch0_late = list(range(200, 313)) # Up to 312

for i in sweep_indices_ch0_late:
    sweep_num_str = str(i)
    stim_key = f'stimulus-{sweep_num_str}-ch-0'
    resp_key = f'current_clamp-response-{sweep_num_str}-ch-0'
    
    if stim_key in nwb.stimulus and resp_key in nwb.acquisition:
        stim_series = nwb.stimulus[stim_key]
        check_points = min(1000, stim_series.data.shape[0])
        stim_data_chunk = stim_series.data[:check_points] * stim_series.conversion
        
        if np.any(stim_data_chunk > 1e-12): 
            print(f"Found depolarizing stimulus in {stim_key}!")
            found_plot_worthy_sweep = True
            plotted_channel = 0
            plotted_sweep_num = sweep_num_str
            
            resp_series = nwb.acquisition[resp_key]
            num_points_to_plot = min(20000, stim_series.data.shape[0]) 
            
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
    
if not found_plot_worthy_sweep:
    print("\n--- No depolarizing stimulus found on Channel 0. Now checking Channel 1 for select sweeps. ---")
    sweep_indices_ch1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250, 300]
    for i in sweep_indices_ch1:
        sweep_num_str = str(i)
        stim_key = f'stimulus-{sweep_num_str}-ch-1'
        resp_key = f'current_clamp-response-{sweep_num_str}-ch-1'

        if stim_key in nwb.stimulus and resp_key in nwb.acquisition:
            print(f"Checking ch-1 sweep: {sweep_num_str} (key: {stim_key})")
            stim_series = nwb.stimulus[stim_key]
            check_points = min(1000, stim_series.data.shape[0])
            stim_data_chunk = stim_series.data[:check_points] * stim_series.conversion

            if np.any(stim_data_chunk > 1e-12):
                print(f"Found depolarizing stimulus in {stim_key} on Channel 1!")
                found_plot_worthy_sweep = True
                plotted_channel = 1
                plotted_sweep_num = sweep_num_str

                resp_series = nwb.acquisition[resp_key]
                num_points_to_plot = min(20000, stim_series.data.shape[0])
                
                stim_data_plot = stim_series.data[:num_points_to_plot] * stim_series.conversion
                resp_data_plot = resp_series.data[:num_points_to_plot] * resp_series.conversion
                time_vector = np.arange(num_points_to_plot) / stim_series.rate

                fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                fig.suptitle(f"Sweep {sweep_num_str}-ch-1 (Depolarizing Stimulus)", fontsize=16)
                axs[0].plot(time_vector, stim_data_plot)
                axs[0].set_title(f'{stim_key} ({stim_series.description})')
                axs[0].set_ylabel(f'Stimulus ({stim_series.unit})')
                axs[1].plot(time_vector, resp_data_plot)
                axs[1].set_title(f'{resp_key} ({resp_series.description})')
                axs[1].set_xlabel('Time (s)')
                axs[1].set_ylabel(f'Response ({resp_series.unit})')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plot_filename_gen = f"explore/depolarizing_stim_resp_sweep{sweep_num_str}_ch1.png"
                plt.savefig(plot_filename_gen)
                print(f"Plot saved to {plot_filename_gen}")
                plt.close(fig)
                break

if not found_plot_worthy_sweep:
    print("No suitable depolarizing stimulus found on ch-0 or ch-1 in the checked sweeps to generate a plot.")

try:
    io.close()
except Exception as e:
    print(f"Error closing NWBHDF5IO: {e}")

print("\nExploration script 6 finished.")