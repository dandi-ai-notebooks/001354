# This script loads the first response and stimulus trace from the selected NWB file and plots the first 1000 samples for both available channels (ch0 and ch1).
# All plots are saved to PNG files in the explore/ directory.

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

# Plot first 1000 samples for current_clamp-response-01-ch-0 and stimulus-01-ch-0
resp = nwb.acquisition["current_clamp-response-01-ch-0"]
stim = nwb.stimulus["stimulus-01-ch-0"]
data_resp = resp.data[0:1000] * resp.conversion  # convert to volts
data_stim = stim.data[0:1000] * stim.conversion  # convert to amperes

time = np.arange(1000) / resp.rate

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time, data_resp)
plt.title("Response (channel 0): current_clamp-response-01-ch-0")
plt.ylabel("Membrane potential (V)")
plt.subplot(2, 1, 2)
plt.plot(time, data_stim)
plt.ylabel("Current stimulus (A)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig("explore/first_1000_ch0.png", dpi=150)
plt.close()

# Plot for channel 1
resp1 = nwb.acquisition["current_clamp-response-01-ch-1"]
stim1 = nwb.stimulus["stimulus-01-ch-1"]
data_resp1 = resp1.data[0:1000] * resp1.conversion
data_stim1 = stim1.data[0:1000] * stim1.conversion

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time, data_resp1)
plt.title("Response (channel 1): current_clamp-response-01-ch-1")
plt.ylabel("Membrane potential (V)")
plt.subplot(2, 1, 2)
plt.plot(time, data_stim1)
plt.ylabel("Current stimulus (A)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.savefig("explore/first_1000_ch1.png", dpi=150)
plt.close()