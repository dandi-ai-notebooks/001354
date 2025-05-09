# %% [markdown]
# # Exploring Dandiset 001354: Hippocampal neuronal responses to programmable antigen-gated G-protein-coupled engineered receptor activation

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset, [DANDI:001354 (version 0.250312.0036)](https://dandiarchive.org/dandiset/001354/0.250312.0036), contains single cell electrophysiological recordings of mouse hippocampal CA1 neurons. These recordings were made in response to activation of programmable antigen-gated G-protein-coupled engineered receptors (PAGERs). Neurons were transfected with an AAV1/2-hSyn-a-mCherry-PAGER-Gi-P2A-mEGFP, and responses were recorded during the application of DCZ (100 nM) or DCZ + soluble mCherry (1 uM).
#
# The study aims to understand neuronal responses under these specific chemogenetic manipulations.
#
# **Citation:** Klein, Peter (2025) Hippocampal neuronal responses to programmable antigen-gated G-protein-coupled engineered receptor activation (Version 0.250312.0036) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.001354/0.250312.0036

# %% [markdown]
# ## What this notebook covers
#
# This notebook will guide you through:
# 1. Listing required Python packages.
# 2. Connecting to the DANDI archive and loading basic information about Dandiset 001354.
# 3. Listing some of the assets (NWB files) available in this Dandiset.
# 4. Loading one of the NWB files and exploring its basic metadata.
# 5. Providing a link to explore the NWB file on Neurosift.
# 6. Demonstrating how to load and visualize some example data from the NWB file.
#
# **Note:** The step to get detailed NWB file information using `tools_cli.py nwb-file-info` timed out during the generation of this notebook. Therefore, the sections on loading specific data paths from the NWB file will be more generic and might require adjustments based on the actual file structure.

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following Python packages. It is assumed they are already installed on your system.
#
# * `dandi` (for interacting with the DANDI API)
# * `pynwb` (for working with NWB files)
# * `h5py` (as a backend for pynwb)
# * `numpy` (for numerical operations)
# * `matplotlib` (for plotting)
# * `itertools` (used in the example DANDI API code)
# * `requests` and `remfile` (often dependencies for streaming NWB data)
# * `seaborn` (for enhanced plotting styles)

# %% [markdown]
# ## Loading the Dandiset using the DANDI API

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
import h5py # For direct HDF5 access if needed
import requests # For fetching S3 URLs
import tempfile # If downloading is chosen
import os # For file operations

# Apply seaborn styling for plots (except for images)
sns.set_theme()

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "001354"
dandiset_version = "0.250312.0036"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}") # This should be the API URL
print(f"Dandiset Web URL: https://dandiarchive.org/dandiset/{dandiset_id}/{dandiset_version}")
print(f"Dandiset description: {metadata.get('description', 'N/A')}")


# %% [markdown]
# ### List Assets in the Dandiset

# %%
# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.asset_id}, Size: {asset.size} bytes)") # Corrected to asset.asset_id

# %% [markdown]
# ## Loading an NWB file
#
# We will now load one of the NWB files from the Dandiset and examine its metadata. We'll choose the first asset listed for this demonstration.
#
# **Path of the NWB file being loaded:** `sub-PK-109/sub-PK-109_ses-20240717T150830_slice-2024-07-17-0001_cell-2024-07-17-0001_icephys.nwb`
#
# **Asset ID:** `8609ffee-a79e-498c-8dfa-da46cffef135`
#
# To load the NWB file, we first need its download URL. The DANDI client can provide S3 URLs for direct streaming.

# %%
# Let's pick the first asset from the previous list (or re-fetch if necessary)
# Ensure assets are fetched if this cell is run independently
assets_iterator = dandiset.get_assets()
try:
    first_asset = next(assets_iterator)
    print(f"Selected asset: {first_asset.path} with ID: {first_asset.asset_id}")

    # Get the S3 download URL for streaming
    # For published versions, we can often construct the URL directly or use asset.get_content_url()
    # The asset.get_content_url(follow_redirects=True, strip_redirects=True) method is robust
    # For simplicity here, and if streaming directly via pynwb with an S3 URL, this is a common pattern:
    nwb_s3_url = first_asset.get_content_url(follow_redirects=True, strip_redirects=True)
    if not nwb_s3_url: # Fallback if direct S3 URL is not immediately available or requires specific flags
        # A more general way to get a download URL:
        nwb_api_download_url = f"https://api.dandiarchive.org/api/assets/{first_asset.asset_id}/download/"
        print(f"Using API download URL: {nwb_api_download_url}")
        # pynwb can sometimes handle this URL directly if it redirects to an S3 URL,
        # or we might need to resolve it. Let's assume pynwb handles remfile usage.
        # For direct streaming, we usually need the underlying S3 URL.
        # The `tools_cli.py` script constructs this:
        # https://api.dandiarchive.org/api/assets/<ASSET_ID>/download/
        # We will use this form for the Neurosift link and attempt to use it for pynwb.
        # pynwb supports http/https URLs, especially if they point to S3 storage, using remfile.
        file_url_for_pynwb = nwb_api_download_url
    else:
        print(f"S3 URL for streaming: {nwb_s3_url}")
        file_url_for_pynwb = nwb_s3_url


    print(f"\nAttempting to load NWB file from: {file_url_for_pynwb}")
    # Note: The `tools_cli.py nwb-file-info` command timed out.
    # The following code attempts generic NWB loading. Specific data paths may need adjustment.
    # For streaming, ensure 'remfile' is installed. pynwb uses it automatically for S3 URLs.
    # Set environment variable for S3 anonymous access by h5py/remfile if needed (often not necessary for public DANDI)
    os.environ['HDF5_USE_ROS3'] = '1' # For direct H5PY S3 driver, pynwb's remfile is usually preferred for http

    # Using 'r' mode and `load_namespaces=True` is standard.
    # For remote files, pynwb typically uses `driver='ros3'` implicitly if path is S3, or `remfile` for http(s)
    # If `file_url_for_pynwb` is the api.dandiarchive.org URL, `remfile` should handle it.
    try:
        io = NWBHDF5IO(path=file_url_for_pynwb, mode='r', load_namespaces=True) # Removed driver='ros3' to let pynwb decide
        nwbfile = io.read()
        print("\nNWB file loaded successfully.")
        print("NWB File Info:")
        print(f"  Identifier: {nwbfile.identifier}")
        print(f"  Session description: {nwbfile.session_description}")
        print(f"  Session start time: {nwbfile.session_start_time}")
        print(f"  Experimenter: {nwbfile.experimenter}")
        print(f"  Institution: {nwbfile.institution}")
        print(f"  Lab: {nwbfile.lab}")
        
        # Do not display the full nwbfile object here to avoid excessive output.
    except Exception as e:
        print(f"Error loading NWB file: {e}")
        print("This might be due to network issues, or the specific URL format not being directly streamable by pynwb without further tools.")
        print("You might need to ensure 'remfile' is installed and working correctly, or try downloading the file first.")
        nwbfile = None # Ensure nwbfile is None if loading fails

except StopIteration:
    print("No assets found in the Dandiset to load.")
    nwbfile = None
except Exception as e:
    print(f"An unexpected error occurred while trying to get asset information or load NWB: {e}")
    nwbfile = None


# %% [markdown]
# ### Explore NWB file on Neurosift
#
# You can explore the NWB file interactively on Neurosift using the following link:
#
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/&dandisetId=001354&dandisetVersion=0.250312.0036](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/&dandisetId=001354&dandisetVersion=0.250312.0036)
# (Note: The Neurosift link uses the version `0.250312.0036` as specified in the setup).

# %% [markdown]
# ## Summarizing NWB File Contents
#
# If the NWB file was loaded successfully, we can inspect its contents.
# **Reminder:** The `tools_cli.py nwb-file-info` command, which provides detailed information about data paths within the NWB file, timed out. The following examples are general and will attempt to access common NWB data structures. If these paths do not exist in this specific file, an error will occur or no data will be found.

# %%
if nwbfile:
    print("Exploring NWB file structure:")
    
    print("\nAvailable acquisition objects (potential raw data):")
    if nwbfile.acquisition:
        for name, series in nwbfile.acquisition.items():
            print(f"- {name} (type: {type(series).__name__})")
            if hasattr(series, 'data'):
                 print(f"  Shape: {series.data.shape}, Dtype: {series.data.dtype}")
            if hasattr(series, 'timestamps') and series.timestamps is not None:
                 print(f"  Timestamps length: {len(series.timestamps)}")
    else:
        print("  No acquisition objects found.")

    print("\nAvailable stimulus objects:")
    if nwbfile.stimulus:
        for name, series in nwbfile.stimulus.items():
            print(f"- {name} (type: {type(series).__name__})")
            if hasattr(series, 'data'):
                 print(f"  Shape: {series.data.shape}, Dtype: {series.data.dtype}")
    else:
        print("  No stimulus objects found.")

    print("\nAvailable processing modules:")
    if nwbfile.processing:
        for module_name, processing_module in nwbfile.processing.items():
            print(f"- Processing module: {module_name}")
            for data_interface_name, data_interface in processing_module.data_interfaces.items():
                print(f"  - {data_interface_name} (type: {type(data_interface).__name__})")
    else:
        print("  No processing modules found.")
        
    print("\nAvailable icephys (intracellular electrophysiology) data:")
    # According to metadata: variableMeasured: ["CurrentClampSeries", "CurrentClampStimulusSeries"]
    # These are typically found under `nwbfile.icephys` or in `nwbfile.acquisition`
    if nwbfile.icephys:
        print("Icephys metadata:")
        for series_name, series_data in nwbfile.icephys.items():
            print(f"- Sweep {series_name}: (type: {type(series_data).__name__})")
            # Example based on common CurrentClampSeries attributes
            if hasattr(series_data, 'data') and series_data.data is not None:
                print(f"  Data shape: {series_data.data.shape}, Unit: {series_data.conversion} {series_data.unit}")
            if hasattr(series_data, 'stimulus_description'):
                 print(f"  Stimulus: {series_data.stimulus_description}")
            if hasattr(series_data, 'sweep_number'):
                 print(f"  Sweep Number: {series_data.sweep_number}")
    else:
        # Check acquisition if not directly in icephys (less standard for processed icephys but possible)
        found_icephys_in_acq = False
        if nwbfile.acquisition:
            for name, series in nwbfile.acquisition.items():
                if isinstance(series, (pynwb.icephys.CurrentClampSeries, pynwb.icephys.VoltageClampSeries, pynwb.icephys.CurrentClampStimulusSeries, pynwb.icephys.VoltageClampStimulusSeries)):
                    if not found_icephys_in_acq:
                        print("Icephys series found in nwbfile.acquisition:")
                        found_icephys_in_acq = True
                    print(f"- {name} (type: {type(series).__name__})")
                    if hasattr(series, 'data'):
                        print(f"  Data shape: {series.data.shape}, Unit: {series.conversion} {series.unit if hasattr(series, 'unit') else 'N/A'}")
        if not found_icephys_in_acq:
            print("  No dedicated icephys objects found in nwbfile.icephys or common icephys types in nwbfile.acquisition.")

else:
    print("NWB file not loaded, skipping content summary.")

# %% [markdown]
# ## Loading and Visualizing Data from the NWB File
#
# We will now attempt to load and visualize some data. As mentioned, specific paths for `CurrentClampSeries` or other data types are not confirmed due to the earlier timeout. We will try to access data typically found in `icephys` experiments.
#
# **Note on remote data access:** Accessing data from remote NWB files involves streaming. For large datasets, it's crucial to load only necessary subsets to avoid long loading times and high memory usage.

# %%
if nwbfile:
    # Try to find CurrentClampSeries, often in acquisition or icephys
    ccs_series = None
    series_name_to_plot = None

    # First check nwbfile.acquisition
    if nwbfile.acquisition:
        for name, series_obj in nwbfile.acquisition.items():
            if isinstance(series_obj, pynwb.icephys.CurrentClampSeries):
                ccs_series = series_obj
                series_name_to_plot = f"acquisition['{name}']"
                print(f"Found CurrentClampSeries in acquisition: {name}")
                break
    
    # If not in acquisition, check nwbfile.icephys (less common for the series itself, but possible)
    if not ccs_series and nwbfile.icephys:
         for name, series_obj in nwbfile.icephys.items():
            if isinstance(series_obj, pynwb.icephys.CurrentClampSeries):
                ccs_series = series_obj
                series_name_to_plot = f"icephys['{name}']"
                print(f"Found CurrentClampSeries in icephys: {name}")
                break
    
    if ccs_series:
        print(f"Plotting data from: {series_name_to_plot}")
        
        data = ccs_series.data
        timestamps = ccs_series.timestamps

        # Determine how much data to plot to keep it manageable
        num_points_to_plot = min(len(data), 2000) # Plot up to 2000 points
        
        # Load a subset of data and corresponding timestamps
        # Using direct slicing on HDF5 dataset
        data_subset = data[:num_points_to_plot]
        if timestamps is not None and len(timestamps) == len(data): # Timestamps might be regularly sampled
            time_subset = timestamps[:num_points_to_plot]
            time_unit = "s" # Assuming timestamps are in seconds
        elif hasattr(ccs_series, 'starting_time') and hasattr(ccs_series, 'rate'):
            # Generate time axis if timestamps are not explicitly stored but rate and starting_time are
            time_subset = ccs_series.starting_time + np.arange(num_points_to_plot) / ccs_series.rate
            time_unit = "s"
        else:
            # Fallback to sample numbers if no time information
            time_subset = np.arange(num_points_to_plot)
            time_unit = "samples"

        print(f"Plotting the first {num_points_to_plot} data points.")

        plt.figure(figsize=(12, 6))
        # Seaborn styling is already applied globally
        plt.plot(time_subset, data_subset)
        plt.title(f"Example Current Clamp Data ({series_name_to_plot}) - First {num_points_to_plot} points")
        plt.xlabel(f"Time ({time_unit})")
        plt.ylabel(f"Voltage ({ccs_series.conversion if ccs_series.conversion else 1.0} {ccs_series.unit})")
        plt.grid(True)
        plt.show()
        
        # Also try to plot stimulus if available
        # Let's assume stimulus would be a CurrentClampStimulusSeries
        # and try to find one, perhaps related by name or structure
        stim_series = None
        if nwbfile.stimulus:
            for name, series_obj in nwbfile.stimulus.items():
                if isinstance(series_obj, pynwb.icephys.CurrentClampStimulusSeries):
                    # Heuristic: match by name if possible, or just take the first one
                    # This is highly speculative without `nwb-file-info`
                    stim_series = series_obj
                    print(f"Found CurrentClampStimulusSeries in stimulus: {name}")
                    break # Take the first one for now
        
        if stim_series:
            stim_data = stim_series.data
            stim_num_points_to_plot = min(len(stim_data), num_points_to_plot) # Match length if possible
            stim_data_subset = stim_data[:stim_num_points_to_plot]
            
            if hasattr(stim_series, 'starting_time') and hasattr(stim_series, 'rate'):
                stim_time_subset = stim_series.starting_time + np.arange(stim_num_points_to_plot) / stim_series.rate
                stim_time_unit = "s"
            else:
                stim_time_subset = np.arange(stim_num_points_to_plot)
                stim_time_unit = "samples"

            plt.figure(figsize=(12, 4))
            plt.plot(stim_time_subset, stim_data_subset, color='orange')
            plt.title(f"Example Stimulus Data ({name}) - First {stim_num_points_to_plot} points")
            plt.xlabel(f"Time ({stim_time_unit})")
            plt.ylabel(f"Current ({stim_series.conversion if stim_series.conversion else 1.0} {stim_series.unit})")
            plt.grid(True)
            plt.show()
            
    else:
        print("Could not find a suitable CurrentClampSeries to plot.")
        print("This might be because the NWB file structure is different than assumed,")
        print("or the data is not present under common paths like 'acquisition' or 'icephys'.")
        print("Consult the Neurosift link or use H5Py to explore the NWB file structure manually.")

else:
    print("NWB file not loaded, skipping data visualization.")

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to:
# - Connect to the DANDI archive and retrieve information about Dandiset 001354.
# - List assets within the Dandiset.
# - Attempt to load an NWB file from the Dandiset for analysis, along with a link to explore it on Neurosift.
# - Show a generic example of how to access and plot time series data (CurrentClampSeries) that might be present in such NWB files.
#
# **Limitations & Challenges:**
# - The `tools_cli.py nwb-file-info` command, which is crucial for understanding the detailed internal structure and specific data paths within NWB files, timed out during the preparation of this notebook. This means that the data loading and visualization examples are based on common NWB conventions (e.g., for CurrentClampSeries) and may not perfectly match the structure of the files in this specific Dandiset. Users may need to use tools like H5Py or Neurosift to identify the exact paths to their data of interest.
# - Visualizing raw electrophysiology data often requires careful selection of time ranges and potentially downsampling or averaging, especially for remote files, to manage performance.
#
# **Possible Future Directions:**
# 1.  **Detailed Exploration:** Use H5Py (e.g., `h5py.File(nwb_file_url, 'r', driver='ros3')`) or the Neurosift link to thoroughly explore the NWB file structure and identify all relevant data groups and datasets (e.g., specific sweep numbers, stimulus protocols, cell metadata).
# 2.  **Targeted Analysis:** Once specific data paths are known, perform targeted analyses. For example:
#     *   Compare neuronal responses across different experimental conditions (e.g., DCZ vs. DCZ + mCherry application) if stimulus information is clearly segregated.
#     *   Extract features from responses, such as spike times (if applicable and sufficiently clear in current clamp), action potential shapes, or changes in membrane potential.
#     *   Analyze data from multiple cells or multiple subjects if the Dandiset contains such data.
# 3.  **Advanced Visualizations:** Create more sophisticated visualizations, such as:
#     *   Overlaying responses from different sweeps or conditions.
#     *   Plotting I-V curves if data from multiple current injection steps are available.
#     *   Creating heatmaps of activity across many trials or cells.
# 4.  **Integrate with Other Tools:** Utilize other Python libraries for neurophysiology data analysis (e.g., `spikeinterface` for spike sorting if extracellular data were present, or custom analysis scripts) to process the data further.
#
# Users are encouraged to adapt the code provided, especially the data access parts, based on the actual structure of the NWB files in this Dandiset. Exploring documentation associated with the Dandiset or the NWB standard can also provide valuable context.

# %%
# Final cleanup if an NWB file was opened (optional, Python's garbage collector usually handles it)
if 'io' in locals() and io is not None:
    try:
        io.close()
        print("NWB IO closed.")
    except Exception as e:
        print(f"Error closing NWB IO: {e}")