"""
Plot the raw stimulus (prior to conversion) for 'stimulus-01-ch-0' in the selected NWB file 
to verify the waveform, data type, and values for troubleshooting stimulus visibility.
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
d = stim.data[:1000]  # raw values, likely float64
conversion = stim.conversion
print(f"Stimulus dtype: {d.dtype}, min={d.min()}, max={d.max()}, conversion={conversion}")

plt.figure(figsize=(8,3))
plt.plot(d, label='Raw Stimulus Data')
plt.ylabel("Stimulus (raw units)")
plt.xlabel("Sample")
plt.title("Raw Stimulus Trace: stimulus-01-ch-0")
plt.tight_layout()
plt.savefig("explore/stimulus_01_ch0_raw.png")
print("Plot saved: explore/stimulus_01_ch0_raw.png")