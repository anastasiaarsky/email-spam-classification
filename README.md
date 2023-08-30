# Spam Classification

My capstone project aims to create two models (one using traditional ML and one using DL) that are capable of detecting spam email messages.

## About

Spam is unwanted and unsolicited messages sent electronically. These spam messages often have malicious intent, and range from misleading advertising to phishing and malware spreads. Thus, spam is detrimental to both users and services, and creates mistrust and wariness between the two parties.

Furthermore, spam is rapidly on the rise, with the Federal Trade Commission reporting $8.8 billion in total reported losses in 2022, compared to the $6.1 billion in 2021 and the mere $1.2 billion in 2020 (FTC.gov). Therefore, it is increasingly important for companies and services to detect and filter spam messages.

## Features

1. **Text Preprocessing**: Automate text cleaning and tokenization to prepare raw text for analysis.
2. **Feature Extraction**: Extract meaningful features from text TF-IDF (Term Frequency-Inverse Document Frequency) for the ML model.
3. **Word Embedding**: Convert text into dense vector representations suitable for NLP tasks using pretrained FastText word embeddings for the DL model.
4. **Text Classification**: Implement text classification models (one ML and one DL) to categorize email text as spam or not spam.
   - ML model: Random Forest Classifier
   - DL model: Convolutional Neural Network (CNN)

## Usage

To collect and **preprocess the data**, run:  
Terminal: ```python src/data/make_dataset.py```  
The preprocessed data will be saved in as a zipped CSV file, located at data/processed_data.zip.

To train the **Random Forest** model (with **TF-IDF** feature extraction):  
Terminal: ```python -m src.models.rf_model --train```  
The saved model can be found at models/random_forest_model.joblib. Additionally, evaluation metrics and a confusion matrix will be printed for the validation set.

To predict once the **Random Forest** model has been trained and saved, run:  
Terminal: ```python -m src.models.rf_model```  
This will perform TF-IDF feature extraction on the testing set and use the saved trained RF model for predictions. Evaluation metrics and a confusion matrix will be printed for the testing set.

To train the **CNN** model (using **FastText** pretrained word embeddings), run:  
Terminal: ```python -m src.models.cnn_model --train```  
The saved model can be found at models/cnn_model.h5. To later load and reuse the trained model without having to retrain it, import the keras library and use the keras.models.load_model method. Additionally, evaluation metrics and a confusion matrix will be printed for the validation set.

To predict using the trained **CNN** model, run:  
Terminal: ```python -m src.models.cnn_model```  
This will utilize FastText word embeddings on the testing set and use the saved trained CNN model for predictions. Evaluation metrics and a confusion matrix will be printed for the testing set.
