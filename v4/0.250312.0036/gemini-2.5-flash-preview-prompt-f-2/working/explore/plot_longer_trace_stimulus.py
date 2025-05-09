# This script loads and plots a longer segment of the first current clamp response and stimulus.

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

# Get the first current clamp response and stimulus data
response_series = nwb.acquisition["current_clamp-response-01-ch-0"]
stimulus_series = nwb.stimulus["stimulus-01-ch-0"]

# Load a longer segment of the data (e.g., first 2 seconds at 20000 Hz)
num_points = 2 * 20000
response_data = response_series.data[0:num_points]
stimulus_data = stimulus_series.data[0:num_points]

# Get conversion factors and convert to appropriate units (mV and pA)
response_conversion = response_series.conversion
response_data_mV = response_data * response_conversion * 1000

stimulus_conversion = stimulus_series.conversion
stimulus_data_pA = stimulus_data * stimulus_conversion * 1e12

# Get the corresponding time points in seconds
rate = response_series.rate
starting_time = response_series.starting_time
time = starting_time + np.arange(num_points) / rate

# Plot the response data
plt.figure(figsize=(10, 4))
plt.plot(time, response_data_mV)
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)')
plt.title('First 2s of Current Clamp Response 01 - Channel 0')
plt.grid(True)
plt.savefig('explore/longer_trace_plot.png')
plt.close()

# Plot the stimulus data
plt.figure(figsize=(10, 4))
plt.plot(time, stimulus_data_pA)
plt.xlabel('Time (s)')
plt.ylabel('Stimulus (pA)')
plt.title('First 2s of Current Clamp Stimulus 01 - Channel 0')
plt.grid(True)
plt.savefig('explore/longer_stimulus_plot.png')
plt.close()

io.close()