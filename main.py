""" 
This script will serve as the central point of this project. It serves to execute the primary logic of this repository
"""
import sys
import subprocess
import pandas as pd
import os
from Preprocessing_EEG import preprocess_eeg, plot_time_domain_signal, plot_band_signal


def main():
    # Intake original training and testing directories
    original_training_directory = input("Input training directory name: ") # Takes in input DIR of 10 training data (5 resting, 5 faces)
    original_testing_directory = input("Input testing directory name: " )# Takes in input DIR of 2 testing data (1 resting, 1 faces)


    # Output filtered training and testing datasets as CSV files
    processed_training_list, processed_testing_list, file_for_visualization = process_real_data(original_training_directory, original_testing_directory)
    combine_preprocessed_eegs(processed_training_list, 'training_combined.csv')
    combine_preprocessed_eegs(processed_testing_list, 'testing_combined.csv')    
    
    
    # Plot unfiltered vs. filtered EEG data for both "Resting" and "Face" conditions
    file_path_1_for_visualization = 'Raw_EEG_Test/Face1_EEG.csv'
    unfiltered_eeg_data = pd.read_csv(file_path_1_for_visualization)
    unfiltered_eeg_data = unfiltered_eeg_data.interpolate(method='linear').dropna()
    
    plot_time_domain_signal(unfiltered_eeg_data['TP9'], file_for_visualization['Normalized_Aggregate'], 'Resting vs Face - TP9', 'TP9_plot.png')
    plot_time_domain_signal(unfiltered_eeg_data['AF7'], file_for_visualization['Normalized_Aggregate'], 'Resting vs Face - AF7', 'AF7_plot.png')
    plot_time_domain_signal(unfiltered_eeg_data['AF8'], file_for_visualization['Normalized_Aggregate'], 'Resting vs Face - AF8', 'AF8_plot.png')
    plot_time_domain_signal(unfiltered_eeg_data['TP10'], file_for_visualization['Normalized_Aggregate'], 'Resting vs Face - TP10', 'TP10_plot.png')
    
    # Plot the filtered band signals for "Resting" and "Face" conditions
    plot_band_signal(file_for_visualization, 'Delta', 'Face - TP9', "delta_plot.png", "Aggregate_Delta")
    plot_band_signal(file_for_visualization, 'Theta', 'Face - TP9', "theta_plot.png", "Aggregate_Theta")
    plot_band_signal(file_for_visualization, 'Alpha', 'Face - TP9', "alpha_plot.png", "Aggregate_Alpha")
    plot_band_signal(file_for_visualization, 'Beta', 'Face - TP9', "beta_plot.png", "Aggregate_Beta")
        
        
    # Train classifier using training dataset, and once trained, test classifier using unseen testing data - ALSO VISUALIZES DATA
    subprocess.run("python brain_wave_classifier.py training_combined.csv", shell=True)
    subprocess.run("python predict_brain_waves.py testing_combined.csv", shell=True)
    

    # Filter synthetic data
    filtered_syn_rest = preprocess_eeg("syn_raw_rest.csv", ['TP9', 'AF7', 'AF8', 'TP10'], 256, 'Rest')
    filtered_syn_face = preprocess_eeg("syn_raw_face.csv", ['TP9', 'AF7', 'AF8', 'TP10'], 256, 'Face')
    
    merged_synthetic_df = pd.concat([filtered_syn_rest, filtered_syn_face], ignore_index=True)
    merged_synthetic_df.to_csv("combined_synthetic_data.csv", index=False) 

    subprocess.run("python predict_brain_waves.py combined_synthetic_data.csv", shell=True)
    
def process_real_data(train, test):
    # Set up variables to call functions
    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    fs = 256  # Sampling frequency in Hz
    
    # Initialize lists to store all preprocessed training and testing files
    csv_files_training = [] 
    csv_files_testing = []

    # Loop through files in the directory, creating list of training csv files
    for file in os.listdir("./" + train):  # List files in directory
        file_path = os.path.join("./" + train, file)  # Construct full file path
        if "Rest" in file:  # Check if "Rest" is in the filename 
            preprocessed_eeg = preprocess_eeg(file_path, channels, fs, 'Rest')
        elif "Face" in file: #Check if "Face" is in the filename
            preprocessed_eeg = preprocess_eeg(file_path, channels, fs, 'Face')
        
        csv_files_training.append(preprocessed_eeg)
    
    # Loop through files in the directory, creating list of testing csv files
    for file in os.listdir("./" + test):  
        file_path = os.path.join("./" + test, file)  # Construct full file path
        if "Rest" in file: 
            preprocessed_eeg = preprocess_eeg(file_path, channels, fs, 'Rest')
        elif "Face" in file: 
            preprocessed_eeg_for_visualization = preprocess_eeg(file_path, channels, fs, 'Face')
        
        csv_files_testing.append(preprocessed_eeg)
        csv_files_testing.append(preprocessed_eeg_for_visualization)
    
    return csv_files_training, csv_files_testing, preprocessed_eeg_for_visualization

#This function produces a filtered training dataset and a filtered testing dataset in CSV format
def combine_preprocessed_eegs(preprocessed_eegs, output_file): 
    combined_data = pd.concat(preprocessed_eegs, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
    
    
    
if __name__ == "__main__":
    main()
    