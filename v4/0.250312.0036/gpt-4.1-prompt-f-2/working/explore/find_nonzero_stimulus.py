"""
Scan through several stimulus-XX-ch-0 channels to identify any with a nonzero stimulus in sub-PK-109_ses-20240717T150830_slice-2024-07-17-0001_cell-2024-07-17-0001_icephys.nwb.
Print min/max for the first 300 'stimulus-XX-ch-0' series.
"""

import pynwb
import h5py
import remfile
import numpy as np

url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

for i in range(1, 301):
    stim_name = f"stimulus-{i:02d}-ch-0"
    try:
        stim = nwb.stimulus[stim_name]
        d = stim.data[:]
        minimum = d.min()
        maximum = d.max()
        mean = d.mean()
        print(f"{stim_name}: min={minimum:.3g}, max={maximum:.3g}, mean={mean:.3g}")
    except Exception as e:
        print(f"{stim_name}: Error or missing.")