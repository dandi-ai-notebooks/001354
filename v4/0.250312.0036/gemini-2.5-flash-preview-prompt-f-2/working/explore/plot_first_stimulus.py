# This script loads and plots the first few data points from the first current clamp stimulus.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the first current clamp stimulus data
stimulus = nwb.stimulus
data_series = stimulus["stimulus-01-ch-0"]

# Load the same number of data points as the response and convert to pA
num_points = 10000
data = data_series.data[0:num_points]
conversion = data_series.conversion
data_pA = data * conversion * 1e12 # convert to pA (assuming base unit is A)

# Get the corresponding time points in seconds
rate = data_series.rate
starting_time = data_series.starting_time
time = starting_time + np.arange(num_points) / rate

# Plot the data
plt.figure(figsize=(10, 4))
plt.plot(time, data_pA)
plt.xlabel('Time (s)')
plt.ylabel('Stimulus (pA)')
plt.title('First 0.5s of Current Clamp Stimulus 01 - Channel 0')
plt.grid(True)
plt.savefig('explore/first_stimulus_plot.png')
plt.close()

io.close()