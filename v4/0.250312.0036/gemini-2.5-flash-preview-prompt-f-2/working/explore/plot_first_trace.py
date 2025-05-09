# This script loads and plots the first few data points from the first current clamp response.

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

# Get the first current clamp response data
acquisition = nwb.acquisition
data_series = acquisition["current_clamp-response-01-ch-0"]

# Load a small segment of the data and convert to mV
# Get the first 10000 data points (0.5 seconds at 20000 Hz)
num_points = 10000
data = data_series.data[0:num_points]
conversion = data_series.conversion
data_mV = data * conversion * 1000 # convert to mV

# Get the corresponding time points in seconds
rate = data_series.rate
starting_time = data_series.starting_time
time = starting_time + np.arange(num_points) / rate

# Plot the data
plt.figure(figsize=(10, 4))
plt.plot(time, data_mV)
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)')
plt.title('First 0.5s of Current Clamp Response 01 - Channel 0')
plt.grid(True)
plt.savefig('explore/first_trace_plot.png')
plt.close()

io.close()