# %% [markdown]
# # Exploring Dandiset 001354: Hippocampal neuronal responses to programmable antigen-gated G-protein-coupled engineered receptor activation

# %% [markdown]
# > **DISCLAIMER**: This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results. Always validate findings with additional research and analysis.

# %% [markdown]
# ## Overview
# 
# This notebook explores Dandiset 001354, which contains single cell electrophysiological recordings from mouse hippocampal CA1 neurons. These recordings capture neuronal responses to the activation of programmable antigen-gated G-protein-coupled engineered receptors (PAGERs).
# 
# According to the dataset description, recorded neurons were transfected with AAV1/2-hSyn-a-mCherry-PAGER-Gi-P2A-mEGFP, and responses were recorded after the application of either DCZ (100 nM) or DCZ + soluble mCherry (1 uM).
# 
# The dataset is available at: [https://dandiarchive.org/dandiset/001354/0.250312.0036](https://dandiarchive.org/dandiset/001354/0.250312.0036)

# %% [markdown]
# ## Required Packages
# 
# The following packages are needed for this analysis:

# %%
# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import islice
import h5py
import remfile
import pynwb
from dandi.dandiapi import DandiAPIClient

# Set seaborn theme for plots, avoiding deprecated style setting
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset
# 
# We'll first connect to the DANDI archive and load basic information about the Dandiset:

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("001354", "0.250312.0036")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Access requirements: {metadata['access'][0]['status']}")
print(f"License: {metadata['license'][0]}")
# Handle contributor field which might contain dictionaries
contributors = []
for contributor in metadata['contributor']:
    if isinstance(contributor, dict) and 'name' in contributor:
        contributors.append(contributor['name'])
    elif isinstance(contributor, str):
        contributors.append(contributor)
    else:
        contributors.append(str(contributor))

print(f"Contributors: {', '.join(contributors)}")
print(f"\nKeywords: {', '.join(metadata['keywords'])}")

# %% [markdown]
# ### Description
# 
# The detailed description of this Dandiset:

# %%
print(metadata['description'])

# %% [markdown]
# ## Listing Assets in the Dandiset
# 
# Let's list some of the NWB files available in this Dandiset:

# %%
# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier}, Size: {asset.size} bytes)")

# %% [markdown]
# ## Loading an NWB File
# 
# For our analysis, we'll focus on a specific NWB file from this Dandiset. We'll load the file using PyNWB and explore its content.
# 
# We'll be examining the file: `sub-PK-109/sub-PK-109_ses-20240717T150830_slice-2024-07-17-0001_cell-2024-07-17-0001_icephys.nwb`

# %%
# Load the NWB file using PyNWB
url = "https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the NWB file
print(f"NWB file identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")

# %% [markdown]
# ### Subject Information
# 
# Let's look at the subject information contained in this NWB file:

# %%
# Print subject information
subject = nwb.subject
print(f"Subject ID: {subject.subject_id}")
print(f"Species: {subject.species}")
print(f"Sex: {subject.sex}")
print(f"Date of birth: {subject.date_of_birth}")
print(f"Age reference: {subject.age__reference}")

# %% [markdown]
# ## NWB File Structure
# 
# The NWB file contains electrophysiological recordings from CA1 neurons. Let's examine the structure of the file in more detail.
# 
# ### Electrodes & Devices

# %%
# Print information about electrodes and devices
print("Electrodes:")
for electrode_id, electrode in nwb.icephys_electrodes.items():
    print(f"  - {electrode_id}")
    print(f"    Description: {electrode.description}")
    print(f"    Device: {electrode.device}")

# %% [markdown]
# ### Lab Metadata
# 
# This NWB file also includes some lab-specific metadata:

# %%
# Extract lab metadata
lab_meta = nwb.lab_meta_data.get('DandiIcephysMetadata')
print(f"Cell ID: {lab_meta.cell_id}")
print(f"Slice ID: {lab_meta.slice_id}")
print(f"Targeted layer: {lab_meta.targeted_layer}")
print(f"Inferred layer: {lab_meta.inferred_layer or 'Not specified'}")

# %% [markdown]
# ### Data Structure
# 
# The NWB file is organized in a hierarchical structure. Here are the main components of the data:
# 
# - **Acquisition**: Contains current clamp response data
# - **Stimulus**: Contains stimulus data
# - **Intracellular Recordings**: Table linking stimuli and responses
# - **ICEphys Simultaneous Recordings**: Table grouping related recordings
# - **ICEphys Sequential Recordings**: Table grouping simultaneous recordings
# 
# Let's examine the acquisition and stimulus data more closely:

# %%
# Count the number of acquisition items
print(f"Number of acquisition items: {len(nwb.acquisition)}")
print(f"Number of stimulus items: {len(nwb.stimulus)}")

# Look at the first few acquisition items
print("\nSample acquisition items:")
for i, (name, series) in enumerate(islice(nwb.acquisition.items(), 3)):
    print(f"  - {name}")
    print(f"    Type: {type(series).__name__}")
    print(f"    Description: {series.description}")
    print(f"    Starting time: {series.starting_time} {series.starting_time_unit}")
    print(f"    Data shape: {series.data.shape}")
    print(f"    Rate: {series.rate} Hz")
    print(f"    Unit: {series.unit}")

# Look at the first few stimulus items
print("\nSample stimulus items:")
for i, (name, series) in enumerate(islice(nwb.stimulus.items(), 3)):
    print(f"  - {name}")
    print(f"    Type: {type(series).__name__}")
    print(f"    Description: {series.description}")
    print(f"    Starting time: {series.starting_time} {series.starting_time_unit}")
    print(f"    Data shape: {series.data.shape}")
    print(f"    Rate: {series.rate} Hz")
    print(f"    Unit: {series.unit}")

# %% [markdown]
# ## Visualizing Data
# 
# Now let's visualize some of the data from the NWB file. We'll start by looking at the responses and stimuli for one of the recordings.

# %%
# Define a function to load and visualize a subset of the data
def visualize_recording(response_name, stimulus_name, sample_size=10000):
    """
    Visualize a recording by plotting the response and stimulus data.
    
    Parameters:
    -----------
    response_name : str
        Name of the response series in nwb.acquisition
    stimulus_name : str
        Name of the stimulus series in nwb.stimulus
    sample_size : int
        Number of data points to plot (default: 10000)
    """
    # Load response data
    response_series = nwb.acquisition[response_name]
    response_data = response_series.data[:sample_size] * response_series.conversion
    response_time = np.arange(len(response_data)) / response_series.rate + response_series.starting_time
    
    # Load stimulus data
    stimulus_series = nwb.stimulus[stimulus_name]
    stimulus_data = stimulus_series.data[:sample_size] * stimulus_series.conversion
    stimulus_time = np.arange(len(stimulus_data)) / stimulus_series.rate + stimulus_series.starting_time
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Plot response data
    ax1.plot(response_time, response_data, color='blue')
    ax1.set_ylabel(f'Membrane Potential ({response_series.unit})')
    ax1.set_title(f'Response: {response_name}')
    ax1.grid(True)
    
    # Plot stimulus data
    ax2.plot(stimulus_time, stimulus_data * 1e12, color='red')  # Convert to pA for better visibility
    ax2.set_ylabel('Current (pA)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title(f'Stimulus: {stimulus_name}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print additional information
    print(f"Response description: {response_series.description}")
    print(f"Stimulus description: {stimulus_series.description}")
    print(f"Recording duration: {sample_size/response_series.rate:.2f} seconds (showing first {sample_size} samples)")
    print(f"Sampling rate: {response_series.rate} Hz")

# %% [markdown]
# ### Visualizing a Single Recording
# 
# Let's visualize the first recording in the dataset:

# %%
# Visualize the first recording
visualize_recording('current_clamp-response-01-ch-0', 'stimulus-01-ch-0')

# %% [markdown]
# ### Comparing Multiple Recordings
# 
# Now let's compare multiple recordings to see how the responses change:

# %%
# Define a function to compare multiple recordings
def compare_recordings(recording_numbers, channel=0, sample_size=5000):
    """
    Compare multiple recordings by plotting their responses.
    
    Parameters:
    -----------
    recording_numbers : list
        List of recording numbers to compare
    channel : int
        Channel number (0 or 1)
    sample_size : int
        Number of data points to plot (default: 5000)
    """
    plt.figure(figsize=(12, 8))
    
    for rec_num in recording_numbers:
        # Format the recording number with leading zeros if needed
        rec_str = f"{rec_num:02d}" if rec_num < 100 else f"{rec_num}"
        
        # Get the response name
        response_name = f"current_clamp-response-{rec_str}-ch-{channel}"
        
        if response_name in nwb.acquisition:
            # Load response data
            response_series = nwb.acquisition[response_name]
            response_data = response_series.data[:sample_size] * response_series.conversion
            response_time = np.arange(len(response_data)) / response_series.rate + response_series.starting_time
            
            # Plot the response data
            plt.plot(response_time, response_data, label=f"Recording {rec_num}")
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'Membrane Potential (V)')
    plt.title(f'Comparison of Multiple Recordings (Channel {channel})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# Compare four sequential recordings
compare_recordings([1, 2, 3, 4])

# %% [markdown]
# ### Examining Ramp-Type Stimuli and Responses

# %%
# Let's look at a different set of recordings
compare_recordings([10, 20, 30, 40])

# %% [markdown]
# ### Visualizing Average Response
# 
# Now let's calculate and visualize the average response across multiple recordings:

# %%
def calculate_average_response(recording_numbers, channel=0, sample_size=5000):
    """
    Calculate the average response across multiple recordings.
    
    Parameters:
    -----------
    recording_numbers : list
        List of recording numbers to average
    channel : int
        Channel number (0 or 1)
    sample_size : int
        Number of data points to use (default: 5000)
    
    Returns:
    --------
    avg_time : ndarray
        Time points for the average response
    avg_response : ndarray
        Average response data
    std_response : ndarray
        Standard deviation of the response data
    """
    all_responses = []
    
    for rec_num in recording_numbers:
        # Format the recording number with leading zeros if needed
        rec_str = f"{rec_num:02d}" if rec_num < 100 else f"{rec_num}"
        
        # Get the response name
        response_name = f"current_clamp-response-{rec_str}-ch-{channel}"
        
        if response_name in nwb.acquisition:
            # Load response data
            response_series = nwb.acquisition[response_name]
            response_data = response_series.data[:sample_size] * response_series.conversion
            all_responses.append(response_data)
    
    # Stack all responses and calculate statistics
    responses_array = np.vstack(all_responses)
    avg_response = np.mean(responses_array, axis=0)
    std_response = np.std(responses_array, axis=0)
    
    # Generate time points
    response_series = nwb.acquisition[f"current_clamp-response-{recording_numbers[0]:02d}-ch-{channel}"]
    avg_time = np.arange(len(avg_response)) / response_series.rate + response_series.starting_time
    
    return avg_time, avg_response, std_response

# %%
# Calculate average response for recordings 1-5
avg_time, avg_response, std_response = calculate_average_response([1, 2, 3, 4, 5])

# Plot average response with standard deviation
plt.figure(figsize=(12, 6))
plt.plot(avg_time, avg_response, 'b-', label='Average Response')
plt.fill_between(avg_time, avg_response-std_response, avg_response+std_response, 
                 alpha=0.3, color='blue', label='±1 SD')
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('Average Response Across Multiple Recordings (Channel 0)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Examining Stimulus Patterns

# %%
# Let's examine the stimulus patterns
def compare_stimuli(stimulus_numbers, channel=0, sample_size=5000):
    """
    Compare multiple stimuli.
    
    Parameters:
    -----------
    stimulus_numbers : list
        List of stimulus numbers to compare
    channel : int
        Channel number (0 or 1)
    sample_size : int
        Number of data points to plot (default: 5000)
    """
    plt.figure(figsize=(12, 6))
    
    for stim_num in stimulus_numbers:
        # Format the stimulus number with leading zeros if needed
        stim_str = f"{stim_num:02d}" if stim_num < 100 else f"{stim_num}"
        
        # Get the stimulus name
        stimulus_name = f"stimulus-{stim_str}-ch-{channel}"
        
        if stimulus_name in nwb.stimulus:
            # Load stimulus data
            stimulus_series = nwb.stimulus[stimulus_name]
            stimulus_data = stimulus_series.data[:sample_size] * stimulus_series.conversion
            stimulus_time = np.arange(len(stimulus_data)) / stimulus_series.rate + stimulus_series.starting_time
            
            # Convert to pA for better visibility
            plt.plot(stimulus_time, stimulus_data * 1e12, label=f"Stimulus {stim_num}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)')
    plt.title(f'Comparison of Stimuli (Channel {channel})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# Compare the stimuli for recordings 1-5
compare_stimuli([1, 2, 3, 4, 5])

# %% [markdown]
# ## Analyzing Response Properties
# 
# Now let's analyze some properties of the neuronal responses like peak amplitude and time to peak:

# %%
def analyze_response_properties(recording_numbers, channel=0):
    """
    Analyze properties of neuronal responses.
    
    Parameters:
    -----------
    recording_numbers : list
        List of recording numbers to analyze
    channel : int
        Channel number (0 or 1)
    
    Returns:
    --------
    properties_df : pandas.DataFrame
        DataFrame containing the analyzed properties
    """
    properties = []
    
    for rec_num in recording_numbers:
        # Format the recording number with leading zeros if needed
        rec_str = f"{rec_num:02d}" if rec_num < 100 else f"{rec_num}"
        
        # Get the response name
        response_name = f"current_clamp-response-{rec_str}-ch-{channel}"
        
        if response_name in nwb.acquisition:
            # Load response data
            response_series = nwb.acquisition[response_name]
            response_data = response_series.data[:] * response_series.conversion
            
            # Calculate response properties
            baseline = np.mean(response_data[:1000])  # First 1000 points as baseline
            peak_amp = np.max(response_data[1000:]) - baseline
            peak_idx = np.argmax(response_data[1000:]) + 1000
            time_to_peak = peak_idx / response_series.rate
            
            # Store properties
            properties.append({
                'Recording': rec_num,
                'Baseline (V)': baseline,
                'Peak Amplitude (V)': peak_amp,
                'Time to Peak (s)': time_to_peak
            })
    
    # Create DataFrame
    properties_df = pd.DataFrame(properties)
    
    return properties_df

# %%
# Analyze properties for recordings 1-10
response_props = analyze_response_properties(range(1, 11))
print(response_props)

# %% [markdown]
# Let's visualize these response properties:

# %%
# Plot the peak amplitudes
plt.figure(figsize=(10, 5))
plt.bar(response_props['Recording'], response_props['Peak Amplitude (V)'])
plt.xlabel('Recording Number')
plt.ylabel('Peak Amplitude (V)')
plt.title('Peak Amplitude for Recordings 1-10')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# %%
# Plot the time to peak
plt.figure(figsize=(10, 5))
plt.bar(response_props['Recording'], response_props['Time to Peak (s)'])
plt.xlabel('Recording Number')
plt.ylabel('Time to Peak (s)')
plt.title('Time to Peak for Recordings 1-10')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Combined Stimulus-Response Analysis
# 
# Finally, let's create a more comprehensive visualization that shows both stimulus and response data together for a single recording, allowing us to better understand the stimulus-response relationship:

# %%
def visualize_stimulus_response_relationship(recording_number, channel=0, sample_size=10000):
    """
    Create a comprehensive visualization of stimulus and response for one recording.
    
    Parameters:
    -----------
    recording_number : int
        Recording number to visualize
    channel : int
        Channel number (0 or 1)
    sample_size : int
        Number of data points to plot (default: 10000)
    """
    # Format the recording number
    rec_str = f"{recording_number:02d}" if recording_number < 100 else f"{recording_number}"
    
    # Get the response and stimulus names
    response_name = f"current_clamp-response-{rec_str}-ch-{channel}"
    stimulus_name = f"stimulus-{rec_str}-ch-{channel}"
    
    if response_name in nwb.acquisition and stimulus_name in nwb.stimulus:
        # Load response data
        response_series = nwb.acquisition[response_name]
        response_data = response_series.data[:sample_size] * response_series.conversion
        response_time = np.arange(len(response_data)) / response_series.rate + response_series.starting_time
        
        # Load stimulus data
        stimulus_series = nwb.stimulus[stimulus_name]
        stimulus_data = stimulus_series.data[:sample_size] * stimulus_series.conversion
        stimulus_time = np.arange(len(stimulus_data)) / stimulus_series.rate + stimulus_series.starting_time
        
        # Create the figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot response data
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Membrane Potential (V)', color='blue')
        ax1.plot(response_time, response_data, color='blue', label='Response')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)
        
        # Create second y-axis for stimulus
        ax2 = ax1.twinx()
        ax2.set_ylabel('Current (pA)', color='red')
        ax2.plot(stimulus_time, stimulus_data * 1e12, color='red', label='Stimulus')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add title and legend
        plt.title(f'Stimulus-Response Relationship for Recording {recording_number}, Channel {channel}')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print additional information
        print(f"Response description: {response_series.description}")
        print(f"Stimulus description: {stimulus_series.description}")
        print(f"Recording duration: {sample_size/response_series.rate:.2f} seconds")
        print(f"Sampling rate: {response_series.rate} Hz")

# %%
# Visualize the stimulus-response relationship for recording 15
visualize_stimulus_response_relationship(15)

# %% [markdown]
# ## Examining Response to Different Treatment Conditions
# 
# The dataset description mentions that responses were recorded after application of either DCZ (100 nM) or DCZ + soluble mCherry (1 uM). While we don't have explicit labels in the dataset for these conditions, we can compare responses from different recordings to look for patterns that might correspond to these different treatments.

# %%
# Compare groups of recordings that might represent different treatment conditions
groups = {
    "Group 1": [1, 2, 3, 4, 5],
    "Group 2": [50, 51, 52, 53, 54],
    "Group 3": [100, 101, 102, 103, 104],
}

plt.figure(figsize=(12, 8))

for name, recordings in groups.items():
    try:
        avg_time, avg_response, _ = calculate_average_response(recordings)
        plt.plot(avg_time, avg_response, label=name)
    except Exception as e:
        print(f"Error processing {name}: {e}")

plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('Average Responses Across Different Recording Groups')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Explore this NWB File in Neurosift
# 
# You can explore this NWB file interactively using the Neurosift web application:
# 
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/&dandisetId=001354&dandisetVersion=0.250312.0036](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/8609ffee-a79e-498c-8dfa-da46cffef135/download/&dandisetId=001354&dandisetVersion=0.250312.0036)

# %% [markdown]
# ## Summary and Conclusions
# 
# In this notebook, we explored Dandiset 001354, which contains electrophysiological recordings of mouse hippocampal CA1 neurons in response to activation of programmable antigen-gated G-protein-coupled engineered receptors (PAGERs).
# 
# We learned:
# 
# 1. How to load and access data from the DANDI archive using the Python API
# 2. How to explore NWB file structure containing intracellular electrophysiology data
# 3. How to extract and visualize current clamp recordings and their associated stimuli
# 4. How to analyze response properties like peak amplitude and time to peak
# 
# The dataset contains a rich collection of recordings from neurons in the CA1 region of the hippocampus, with various stimulus protocols being applied. These recordings provide insights into how PAGERs can modulate neuronal activity.
# 
# ### Possible Future Directions
# 
# Future analyses of this dataset could include:
# 
# 1. More detailed characterization of neuronal response properties, such as input resistance and membrane time constant
# 2. Comparison of responses across different subject animals and experimental conditions
# 3. Quantitative analysis of how PAGER activation affects neuronal excitability and synaptic transmission
# 4. Correlation of electrophysiological properties with fluorescence imaging of the transfected neurons
# 5. Development of computational models to predict neuronal responses to PAGER activation

# %% [markdown]
# ## Acknowledgments
# 
# This dataset was contributed by Peter Klein and collaborators. The authors acknowledge support from several institutions, including the St Jude Children's Research Hospital Collaborative Research Consortium on GPCRs, the Chan Zuckerberg Biohub–San Francisco, Phil and Penny Knight Initiative for Brain Resilience, Stanford Cancer Institute, Wu Tsai Neurosciences Institute of Stanford University, and the NIH.