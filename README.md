# Augmenting EEG Classification with GAN-Generated Data

## Overview
This repository contains the work of the NeuroGryphs team for BrainHack 2024.
We aim to build an end-to-end EEG pipeline that includes data acquisition, preprocessing, predictive modeling, and GAN-based synthetic EEG generation. This comprehensive approach seeks to evaluate GAN-generated synthetic data as a proxy for limited real-world healthcare data.

## Demo Video

[![Watch the Demo Video](https://img.youtube.com/vi/d2m1RBaY948/0.jpg)](https://www.youtube.com/watch?v=d2m1RBaY948&t=104s)

Click the image above to watch the video.

## Project Description
Our project combines classical EEG analysis techniques with modern machine learning frameworks to:
1. Acquire and preprocess EEG data.
2. Train predictive models to classify brain states (e.g., "Face Visualization" vs. "Resting State").
3. Develop a Generative Adversarial Network (GAN) to generate synthetic EEG data.
4. Test and evaluate the GAN and predictive models on real and synthetic EEG datasets.

By integrating these components, we aim to improve EEG data analysis pipelines and the potential use of synthetic EEG data in neuroscience research.

## Repository Structure
```
NeuroGryphs_BrainHack2024/
├── Raw_EEG_Test/               # Raw EEG testing dataset
├── Raw_EEG_Train/              # Raw EEG training dataset
├── Preprocessing_EEG.py        # Filtering and preparing data for model with vizualizations
├── brain_wave_classifier.py    # Model script with visualizations for training data
├── main.py/                    # Main 
├── predict_brain_waves.py      # Model script with visualizations for testing data
└── README.md                   # Project documentation
```

## Installation

**Clone the repository**:
   ```bash
   git clone https://github.com/mirzaahmadi/NeuroGryphs_BrainHack2024.git
   cd NeuroGryphs_BrainHack2024
   ```
## Usage 
To execute the pipeline, use the command below. You will be prompted to specify the input directories for testing and training data.

 ```bash
  python main.py
```

## End-to-End Pipeline

### Step 1: Data Acquisition
Acquired EEG data of 6 individuals at a resting state and visualizing faces using NeuroFusion and Muse. 

### Step 2: Preprocessing
Prepare the EEG data by applying:
- Bandpass filtering to isolate relevant frequencies.
- Powerline noise removal (at 50Hz). 
- Segmentation into frequency bands for analysis.

Visualizations of filtered vs. unfiltered EEG signals for all channels and frequency bands will be saved in the `time_domain_signal_plots/`,`band_signal_plots/` folder, respectively.

### Step 3: Predictive Modeling
Train a model to classify brain states (e.g., "Face" vs. "Resting"):

Model performance metrics, such as AUC-ROC curve and learning curve, will be saved in the `Visualizations/Curves/` folder.

### Step 4: GAN Development
Train a GAN to generate synthetic EEG data using eeggan package. 

GAN parameters: 
- Sample Size = 1000
- Sequence Length = 100
- Parallel Generation = 50
- Training Rate = 0.0001

### Step 5: Testing and Integration
Test the predictive model with both real and synthetic EEG data. 

Results, including performance metrics and visualizations, are saved in the folders and visible in command line. 

## Results
- **Predictive Model Accuracy on Training Dataset**: [90%].
- **Predictive Model Accuracy on Testing Dataset**: [95%].

## Future Work
- Implement higher quality EEG data into GAN and model. 
- Expand the GAN framework to output higher quality synthetic EEG data. 
- Improve model to better predict cognitive activity for synthetic EEG data.
- Develop a model to distinguish real vs. synthetic EEG data, ideally with a low or 50% accuracy. 

## Collaborators 
- Mirza Ahmadi
- Isha Baxi
- Rebecca Choi
- Vivian Phung
- Moiz Syed
- Thomas Tekle
- Mani Setayesh

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- **Data Acquisition**: Collected using [NeuroFusion](https://usefusion.ai/research) and the Muse Headband. Special thanks to [Ore Ogundipe] for their support and guidance.
- **References**: [EEG-GAN](https://autoresearch.github.io/EEG-GAN/)
