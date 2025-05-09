"""
Explore: Plot the response and stimulus for a segment where current is injected in an NWB file from Dandiset 001354.
This script uses "current_clamp-response-02-ch-0" (acquisition) and "stimulus-02-ch-0" (stimulus)
for cell sub-PK-109, plotting the first 1000 samples from both.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Hard-coded URL as per info
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

resp_name = "current_clamp-response-02-ch-0"
stim_name = "stimulus-02-ch-0"

resp = nwb.acquisition[resp_name]
stim = nwb.stimulus[stim_name]

n = 1000  # samples to plot
resp_data = resp.data[:n] * resp.conversion  # volts
stim_data = stim.data[:n] * stim.conversion  # amperes
rate = resp.rate
t = np.arange(n) / rate + resp.starting_time  # Use the correct starting time

plt.figure(figsize=(8, 5))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t, resp_data, label='Membrane Potential')
ax1.set_ylabel('Membrane potential (V)')
ax1.set_title('Evoked Current Clamp Response')
ax2 = plt.subplot(2, 1, 2)
ax2.plot(t, stim_data, color='tab:orange', label='Stimulus')
ax2.set_ylabel('Current (A)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Applied Stimulus')
plt.tight_layout()
plt.savefig("explore/evoked_response.png")
print("Plot saved: explore/evoked_response.png")