# This script searches for the first response/stimulus pair in the NWB file where the stimulus is nonzero (for either channel).
# It plots the first 1000 samples of both response and stimulus for that pair and channel(s). Plot is saved in explore/.
# If no nonzero pair is found, script will not produce a plot.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

def plot_traces(resp, stim, channel_label, epoch_idx):
    data_resp = resp.data[0:1000] * resp.conversion
    data_stim = stim.data[0:1000] * stim.conversion
    time = np.arange(1000) / resp.rate

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, data_resp)
    plt.title(f"Response (channel {channel_label}): {resp.name}")
    plt.ylabel("Membrane potential (V)")
    plt.subplot(2, 1, 2)
    plt.plot(time, data_stim)
    plt.ylabel("Current stimulus (A)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plot_fname = f"explore/first_nonzero_stimulus_epoch{epoch_idx}_ch{channel_label}.png"
    plt.savefig(plot_fname, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_fname}")

found = False
for i in range(1, 20):  # Check up to 20 epochs for performance
    for ch in [0, 1]:
        resp_key = f"current_clamp-response-{i:02d}-ch-{ch}"
        stim_key = f"stimulus-{i:02d}-ch-{ch}"
        if resp_key in nwb.acquisition and stim_key in nwb.stimulus:
            stim = nwb.stimulus[stim_key]
            stim_data = stim.data[0:1000] * stim.conversion
            if np.any(stim_data != 0):
                resp = nwb.acquisition[resp_key]
                plot_traces(resp, stim, ch, i)
                found = True
    if found:
        break
if not found:
    print("No nonzero stimulus found in first 20 epochs for either channel.")