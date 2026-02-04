
# HHAI-KEML-DCRF_BiLSTM_2025
The source code of the paper titled “A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering” is uploaded here. The paper is published on ceur-ws.org: https://ceur-ws.org/Vol-4074/paper4-1.pdf

# License
© 2025. This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
To view a copy of this license, visit: https://creativecommons.org/licenses/by/4.0/

# Summary
This work implements a Deep Conditional Random Field (DeepCRF) with BiLSTM/GRU framework for Speech Emotion Recognition (SER). It supports feature extraction, preprocessing, data augmentation, training, and N-fold cross-validation, and produces detailed evaluation reports, confusion matrices, and aggregated training curves. The code is designed for research experiments and reproducible evaluation across multiple(five) speech emotion datasets.

# Folder Structure
.
├── DeepCRF_BiLSTM/       # These python scripts are used for Emotion Detection using DeepCRF and BiLSTM, 80%-20% train-test aplit without PCA, 5-fold cross validation with PCA.
│   ├── drawing.py
│   ├── main.py
│   ├── main_fold.py
│   ├── main_PCA_Count.py
│   ├── train_DeepCRF.py
│   ├── train_DeepCRF_Fold.py
│   ├── train_DeepCRF_Normal_Paper.py
│   ├── train_DeepCRF_PCA_Count.py
│
├── SpeechEM/      # These python scripts used for feature extraction from audio-speach of 5 public dataset.
│   ├── AudioProcess.py        # Audio preprocessing
│   ├── DataAugmentation.py    # Data augmentation methods
│   ├── FeatureExtract.py      # Feature extraction (e.g., MFCC, etc.)
│   ├── ReadDatabase.py        # Dataset loader
│   └── main.py                # Pipeline entry point
│
├── datasetFeature/            # Extracted feature files (*.feat)
├── results/                   # Auto-generated results (metrics, plots, reports)
└── README.md


# Guidelines for Running the Code

- Set up the Python environment and install all required libraries and dependencies.
- Update the dataset path, feature set path, and any other necessary paths for saving data, scores, and images.
- Make sure all paths are correctly configured before running the code to avoid errors.

# Publicly avalable five datasets
