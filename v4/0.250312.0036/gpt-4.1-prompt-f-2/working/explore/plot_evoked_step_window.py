"""
Plot membrane potential and applied stimulus from sample 20000 to 65000 for 'current_clamp-response-01-ch-0' and 'stimulus-01-ch-0'.
Goal: illustrate the neuronal voltage response to a strong sustained current pulse.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

resp = nwb.acquisition["current_clamp-response-01-ch-0"]
stim = nwb.stimulus["stimulus-01-ch-0"]

ix1 = 20000
ix2 = 65000

resp_data = resp.data[ix1:ix2] * resp.conversion  # volts
stim_data = stim.data[ix1:ix2] * stim.conversion  # amperes
rate = resp.rate
t = np.arange(ix1, ix2) / rate  # in seconds

plt.figure(figsize=(9, 5))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t, resp_data, label='Membrane Potential')
ax1.set_ylabel('Membrane potential (V)')
ax1.set_title('Membrane Potential Response to Sustained Current Step')
ax2 = plt.subplot(2, 1, 2)
ax2.plot(t, stim_data, color='tab:orange', label='Stimulus')
ax2.set_ylabel('Current (A)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Applied Current Step')
plt.tight_layout()
plt.savefig("explore/evoked_step_response.png")
print("Plot saved: explore/evoked_step_response.png")