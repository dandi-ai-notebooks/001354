# %% [markdown]
# # Exploring Dandiset 001354: Hippocampal neuronal responses to programmable antigen-gated G-protein-coupled engineered receptor activation
#
# This notebook helps explore the contents of Dandiset 001354 version 0.250312.0036.
#
# **Disclaimer:** This notebook was AI-generated and has not been fully verified. Users should be cautious when interpreting the code or results and are encouraged to verify the outputs independently.

# %% [markdown]
# ## Dandiset Overview
#
# This dataset contains single cell electrophysiological recordings of mouse hippocampal CA1 neurons in response to activation of programmable antigen-gated G-protein-coupled engineered receptors.
#
# Dandiset Link: https://dandiarchive.org/dandiset/001354/0.250312.0036

# %% [markdown]
# ## Notebook Contents
#
# This notebook will cover:
# - Loading the Dandiset and listing its assets.
# - Loading an NWB file from the Dandiset.
# - Exploring the structure and metadata of the NWB file.
# - Visualizing sample current clamp response and stimulus data.

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following Python packages:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `itertools`

# %% [markdown]
# ## Loading the Dandiset
#
# We can connect to the DANDI archive and access the Dandiset using the `dandi` Python client.

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001354", "0.250312.0036")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB File
#
# We will load one of the NWB files from the Dandiset to explore its contents. For this example, we will use the following NWB file:
# `sub-PK-109/sub-PK-109_ses-20240717T150830_slice-2024-07-17-0001_cell-2024-07-17-0001_icephys.nwb`
#
# The URL for this asset is: `https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/`

# %%
import pynwb
import h5py
import remfile

# Load the NWB file directly from the DANDI archive using its URL
nwb_url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## NWB File Metadata
#
# We can access various metadata about the recording from the loaded NWB file.

# %%
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Targeted layer: {nwb.lab_meta_data['DandiIcephysMetadata'].targeted_layer}")

# %% [markdown]
# ## Exploring Acquisition and Stimulus Data
#
# This NWB file contains current clamp recordings and stimuli in the `acquisition` and `stimulus` sections, respectively. We can see the available data series.

# %%
print("Acquisition data keys:")
for key in nwb.acquisition.keys():
    print(f"- {key}")

print("\nStimulus data keys:")
for key in nwb.stimulus.keys():
    print(f"- {key}")

# %% [markdown]
# ## Visualizing Sample Data
#
# Let's visualize a segment of the first current clamp response and its corresponding stimulus to see an example of the recorded data and the applied stimulus. We will plot the `current_clamp-response-01-ch-0` and `stimulus-01-ch-0` series.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Get the first current clamp response and stimulus data series
response_series = nwb.acquisition["current_clamp-response-01-ch-0"]
stimulus_series = nwb.stimulus["stimulus-01-ch-0"]

# Define the time segment to visualize (e.g., from 0 to 2 seconds)
start_time = 0
end_time = 2
rate = response_series.rate
start_index = int(start_time * rate)
end_index = int(end_time * rate)
num_points = end_index - start_index

# Load the data for the specified time segment
response_data = response_series.data[start_index:end_index]
stimulus_data = stimulus_series.data[start_index:end_index]

# Get conversion factors and convert to appropriate units (mV and pA)
response_conversion = response_series.conversion
response_data_mV = response_data * response_conversion * 1000

stimulus_conversion = stimulus_series.conversion
stimulus_data_pA = stimulus_data * stimulus_conversion * 1e12

# Create a time vector
time = response_series.starting_time + np.arange(num_points) / rate

# Plot the response data
plt.figure(figsize=(10, 4))
plt.plot(time, response_data_mV)
plt.xlabel('Time (s)')
plt.ylabel(f'Membrane potential ({response_series.unit})')
plt.title('Current Clamp Response 01 - Channel 0')
plt.grid(True)
plt.show()

# Plot the stimulus data
plt.figure(figsize=(10, 4))
plt.plot(time, stimulus_data_pA)
plt.xlabel('Time (s)')
plt.ylabel(f'Stimulus ({stimulus_series.unit})')
plt.title('Current Clamp Stimulus 01 - Channel 0')
plt.grid(True)
plt.show()

# %% [markdown]
# As seen in the plots, the recording starts with a baseline period followed by a step stimulus that elicits a series of action potentials in the neuron.

# %% [markdown]
# ## Further Exploration
#
# Researchers can further explore this Dandiset by:
# - Loading other NWB files to analyze recordings from different cells or sessions.
# - Examining other data interfaces within the NWB files if available (though in this case the primary data is current clamp).
# - Performing quantitative analysis on the current clamp data, such as calculating input resistance, membrane time constant, or analyzing action potential properties.
# - Correlating electrophysiological data with experimental conditions (e.g., presence of DCZ or mCherry).

# %%
# Close the NWB file
io.close()