""" 
This script will read in an UNSEEN data file (of which the model has not been trained on) which will then be used to test the model
"""
import pickle
from brain_wave_classifier import load_data
import re
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # Checks command line arguments to ensure arguments are correctly formatted
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict_brain_waves.py 'dataset.csv' \n Incorrect number of arguments")
    if not re.search(r".+\.csv$" ,sys.argv[1]):
        sys.exit("Usage: python predict_brain_waves.py 'dataset.csv' \n Please make sure the dataset is in .CSV format")
    
    # Only features are needed, as the model will be predicting targets
    features, targets = load_data(sys.argv[1])  
    
    #Load the trained model
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    
    # Use the loaded model for predictions
    predictions = trained_model.predict(features)
    
    # Evaluate the model - multiple accuracy metrics employed
    accuracy = accuracy_score(targets, predictions)
    print("")
    print("")
    print('TESTING DATASET - ACCURACY METRICS')
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(targets, predictions))
    print("Classification Report:\n", classification_report(targets, predictions, target_names=["Rest", "Face"]))
        
        
if __name__ == "__main__":
    main()
    
