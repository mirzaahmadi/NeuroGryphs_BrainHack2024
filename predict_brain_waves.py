""" 
This script will read in an UNSEEN data file (of which the model has not been trained on) which will then be used to test the model
"""
import pickle
from brain_wave_classifier import load_data
import re
import sys

def main():
    # Checks command line arguments to ensure arguments are correctly formatted
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict_new_data.py 'dataset.csv' \n Incorrect number of arguments")
    if not re.search(r".+\.csv$" ,sys.argv[1]):
        sys.exit("Usage: python brain_wave_classifier.py 'dataset.csv' \n Please make sure the dataset is in .CSV format")
    
    # Only features are needed, as the model will be predicting targets
    features, targets = load_data(sys.argv[1])  
    
    number_of_rests = 0
    number_of_faces = 0
    
    for pair in zip(features, targets):
        if pair[1] == 0.0:
            number_of_rests += 1
        if pair[1] == 1.0:
            number_of_faces += 1
            
       
    # Load the trained model from the pickle file - to use on the unseen data
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    
    # Use the loaded model for predictions
    predictions = trained_model.predict(features)
    
    resting = 0
    faces = 0
    
    for pred in predictions:
        print(pred)
        
        if pred == 0.0:
            resting += 1
        if pred == 1.0:
            faces += 1
    
    print("Accuracy: ", (resting / (resting + faces)) * 100)
        
        
if __name__ == "__main__":
    main()
    
