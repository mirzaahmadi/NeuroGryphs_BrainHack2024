import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# Define the preprocessing function
def preprocess_eeg(file_path, channels, fs, target_entry):

    # Load EEG data
    eeg_data = pd.read_csv(file_path)

    # Handle missing values
    eeg_data = eeg_data.interpolate(method='linear').dropna()

    # Bandpass filter function
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    # Notch filter function
    def notch_filter(data, freq, fs, quality=30):
        nyquist = 0.5 * fs
        w0 = freq / nyquist
        b, a = iirnotch(w0, quality)
        return filtfilt(b, a, data)

    # Normalize function
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12), 
        'Beta': (13, 30)
    }

    # Initialize preprocessed data
    preprocessed_data = pd.DataFrame()

    # Aggregate bands data
    aggregated_bands = {band: [] for band in bands.keys()}

    # Loop through channels for preprocessing
    for channel in channels:
        # Apply bandpass filter (0.5-50 Hz)
        filtered_data = bandpass_filter(eeg_data[channel], 0.5, 50, fs)

        # Apply notch filter (50 Hz)
        filtered_data = notch_filter(filtered_data, 50, fs)

        # Normalize the filtered signal
        normalized_data = normalize(filtered_data)

        # Extract frequency bands and aggregate
        for band, (low, high) in bands.items():
            band_filtered = bandpass_filter(filtered_data, low, high, fs)
            aggregated_bands[band].append(band_filtered)

    # Create aggregated columns for each band
    for band in bands.keys():
        preprocessed_data[f'Aggregate_{band}'] = np.mean(aggregated_bands[band], axis=0)

    # Create a normalized aggregate column
    preprocessed_data['Normalized_Aggregate'] = normalize(preprocessed_data[[f'Aggregate_{band}' for band in bands.keys()]].mean(axis=1))

    preprocessed_data['Target'] = target_entry

    return preprocessed_data

def plot_time_domain_signal(unfiltered_data, filtered_data, title, file_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(unfiltered_data)
    plt.title(f"Unfiltered EEG - {title}")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(filtered_data)
    plt.title(f"Filtered EEG - {title}")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    
    #Save plots to folder
    visualizations_folder_path = './Visualizations'
    folder_path1 = 'time_domain_signal_plots'
    os.makedirs(os.path.join(visualizations_folder_path, folder_path1), exist_ok=True)

    plt.savefig(os.path.join(visualizations_folder_path, folder_path1, file_name), format='png', dpi=300)

def plot_band_signal(filtered_band_data, band_name, title, file_name, aggregate):
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_band_data[aggregate])
    plt.title(f"Filtered {band_name} Band - {title}")
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    
    #Save plots to folder
    folder_path2 = './Visualizations/band_signal_plots'
    os.makedirs(folder_path2, exist_ok=True)
    plt.savefig(os.path.join(folder_path2, file_name), format='png', dpi=300)
