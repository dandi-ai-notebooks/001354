# This script loads the first current clamp stimulus series and plots a subset of the data.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the first current clamp stimulus series
stimulus_series = nwb.stimulus["stimulus-01-ch-0"]

# Get a subset of the data and the corresponding time
data = stimulus_series.data[0:50000]
rate = stimulus_series.rate
starting_time = stimulus_series.starting_time
timestamps = starting_time + np.arange(len(data)) / rate

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel(f"Current ({stimulus_series.unit})")
plt.title("Current Clamp Stimulus (first 2.5 seconds)")
plt.grid(True)
plt.savefig('explore/stimulus_01_ch_0.png')
plt.close()

io.close()