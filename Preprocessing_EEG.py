import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

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

# Define function to combine multiple preprocessed EEG datasets
def combine_preprocessed_eegs(preprocessed_eegs, output_file):

    combined_data = pd.concat(preprocessed_eegs, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
    return combined_data

# Set up variables to call functions
channels = ['TP9', 'AF7', 'AF8', 'TP10']
fs = 256  # Sampling frequency in Hz

file_path1 = 'Raw_EEG_Data/Face3_EEG.csv'
file_path2 = 'Raw_EEG_Data/Rest1_EEG.csv'
file_path3 = 'Raw_EEG_Data/Face2_EEG.csv'
file_path4 = 'Raw_EEG_Data/Rest2_EEG.csv'
file_path5 = 'Raw_EEG_Data/Face3_EEG.csv'
file_path6 = 'Raw_EEG_Data/Rest3_EEG.csv'
file_path7 = 'Raw_EEG_Data/Face4_EEG.csv'
file_path8 = 'Raw_EEG_Data/Rest4_EEG.csv'
file_path9 = 'Raw_EEG_Data/Face5_EEG.csv'
file_path10 = 'Raw_EEG_Data/Rest5_EEG.csv'
file_path11 = 'Raw_EEG_Data/Face6_EEG.csv'
file_path12 = 'Raw_EEG_Data/Rest6_EEG.csv'

# Call preprocessed_eeg function for 12 files 
preprocessed_eeg_1 = preprocess_eeg(file_path1, channels, fs, 'Face')
preprocessed_eeg_2 = preprocess_eeg(file_path2, channels, fs,'Resting')  
preprocessed_eeg_3 = preprocess_eeg(file_path3, channels, fs, 'Face')
preprocessed_eeg_4 = preprocess_eeg(file_path4, channels, fs, 'Resting')
preprocessed_eeg_5 = preprocess_eeg(file_path5, channels, fs, 'Face')
preprocessed_eeg_6 = preprocess_eeg(file_path6, channels, fs, 'Resting')
preprocessed_eeg_7 = preprocess_eeg(file_path7, channels, fs, 'Face')
preprocessed_eeg_8 = preprocess_eeg(file_path8, channels, fs, 'Resting')
preprocessed_eeg_9 = preprocess_eeg(file_path9, channels, fs, 'Face')
preprocessed_eeg_10 = preprocess_eeg(file_path10, channels, fs, 'Resting')
preprocessed_eeg_11 = preprocess_eeg(file_path11, channels, fs, 'Face')
preprocessed_eeg_12 = preprocess_eeg(file_path12, channels, fs, 'Resting')

# Split into testing and training data set
filtered_train = combine_preprocessed_eegs([preprocessed_eeg_1, preprocessed_eeg_2, preprocessed_eeg_3, preprocessed_eeg_4, preprocessed_eeg_5, preprocessed_eeg_6, preprocessed_eeg_9, preprocessed_eeg_10, preprocessed_eeg_11, preprocessed_eeg_12], 'filtered_training_combined.csv')

filtered_test = combine_preprocessed_eegs([preprocessed_eeg_7, preprocessed_eeg_8], 'filtered_test_combined.csv')
