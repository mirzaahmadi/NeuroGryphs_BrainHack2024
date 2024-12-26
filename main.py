""" 
This script will serve as the central point of this project. It serves to execute the primary logic of this repository
"""
import sys
import subprocess
import pandas as pd
import os
from Preprocessing_EEG import preprocess_eeg
import brain_wave_classifier


def main():
    # Intake original training and testing directories
    original_training_directory = input("Input training directory name: ") # Takes in input DIR of 10 training data (5 resting, 5 faces)
    original_testing_directory = input("Input testing directory name: " )# Takes in input DIR of 2 testing data (1 resting, 1 faces)

    # Output filtered training and testing datasets as CSV files
    processed_training_list, processed_testing_list = process(original_training_directory, original_testing_directory)
    
    combine_preprocessed_eegs(processed_training_list, 'training_combined.csv')
    combine_preprocessed_eegs(processed_testing_list, 'testing_combined.csv')    
    
    # Train classifier using training dataset, and once trained, test classifier using unseen testing data
    subprocess.run("python brain_wave_classifier.py training_combined.csv", shell=True)
    subprocess.run("python predict_brain_waves.py testing_combined.csv", shell=True)

    
    
def process(train, test):
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
            preprocessed_eeg = preprocess_eeg(file_path, channels, fs, 'Face')
        
        csv_files_testing.append(preprocessed_eeg)
    
    return csv_files_training, csv_files_testing

#This function produces a filtered training dataset and a filtered testing dataset in CSV format
def combine_preprocessed_eegs(preprocessed_eegs, output_file): 
    combined_data = pd.concat(preprocessed_eegs, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
    
if __name__ == "__main__":
    main()
    