"""
Print and plot the full range of stimulus-01-ch-0 for all 100,000 samples.
Goal: discover if/when a non-zero current is injected and at what sample range.
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

stim = nwb.stimulus["stimulus-01-ch-0"]
d = stim.data[:]
conversion = stim.conversion
n_nonzero = np.count_nonzero(d)
print(f"Stimulus samples: {d.shape[0]}, min={d.min()}, max={d.max()}, mean={d.mean()}, nonzero count={n_nonzero}")

plt.figure(figsize=(10,4))
plt.plot(d, label='Stimulus (raw)')
plt.ylabel("Stimulus (raw units)")
plt.xlabel("Sample")
plt.title("Raw Stimulus Trace for All Samples: stimulus-01-ch-0")
plt.tight_layout()
plt.savefig("explore/stimulus_01_ch0_full.png")
print("Plot saved: explore/stimulus_01_ch0_full.png")