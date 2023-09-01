# Spam Classification

This repository contains the code and resources for a spam classification capstone project that includes both a machine learning and a deep learning solution. 
The project aims to compare the performance of these two techniques for accurately classifying email messages as spam or not spam.

## Introduction

Spam messages are a common issue in communication platforms. This project explores both traditional machine learning and modern deep learning approaches to tackle the spam classification problem. By comparing the performance of these two different techniques, we aim to identify the most effective solution for accurate spam detection.

## Dataset

The dataset includes 39,763 entries, with 20,695 labeled as ham and 19,068 as spam.

It is made up of two publicly available datasets:
- [SpamAssassin dataset](https://spamassassin.apache.org/old/publiccorpus/)
- [Enron Spam dataset](https://github.com/MWiechmann/enron_spam_data)  

You can find the raw and preprocessed datasets [`data`](https://github.com/anastasiaarsky/production_repo/tree/main/data) directory. 
The code for data collection and preprocessing are located in the [`src/data`](https://github.com/anastasiaarsky/production_repo/tree/main/src/data) directory.


## Features

1. **Data Collection**: Collect and curate a labeled dataset of email messages for training and evaluation.
2. **Text Preprocessing**: Automate text cleaning and tokenization to prepare raw text for analysis.
2. **Feature Extraction**: Extract meaningful features from text TF-IDF (Term Frequency-Inverse Document Frequency) for the ML model.
3. **Word Embedding**: Convert text into dense vector representations suitable for NLP tasks using pretrained FastText word embeddings for the DL model.
4. **Text Classification**: Implement spam classification models (one ML and one DL) to categorize the text in email messages as spam or not spam.
   - ML model: Random Forest Classifier (RF)
   - DL model: Convolutional Neural Network (CNN)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/anastasiaarsky/production_repo.git
cd production_repo
```

2. Collect and preprocess the data:  
```bash 
python -m src.data.make_dataset
```
- The preprocessed data will be saved in as a zipped CSV file, located at data/processed_data.zip. 
- The raw data will also be saved in data/raw_data.zip.

3. Machine Learning:  
   a. Train the Random Forest model (with TF-IDF feature extraction):  
   ```bash
   python -m src.models.rf_model --train
   ```  
   - This will perform TF-IDF feature extraction on the training and validation sets and use the training set to train the model. 
   - The model will be saved in models/random_forest_model.joblib. 
   - Evaluation metrics and a confusion matrix will be printed for the validation set.

   b. Run predictions once the Random Forest model has been trained and saved:
   ```bash
   python -m src.models.rf_model
   ```  
   - This will perform TF-IDF feature extraction on the testing set and use the trained RF model that was previously saved for predictions. 
   - Evaluation metrics and a confusion matrix will be printed for the testing set.  
   

4. Deep Learning:  
   a. Train the CNN model (using FastText pretrained word embeddings):  
   ```bash 
   python -m src.models.cnn_model --train
   ```  
   - This will first vectorize the training and validation sets and use them to create an embedding layer for the CNN. 
   - Then the CNN will be trained and saved at models/cnn_model.h5.

   b. Run predictions using the trained CNN model:  
   ```bash 
   python -m src.models.cnn_model
   ```  
   - This will vectorize the testing set and use the saved trained CNN model for predictions. 
   - Evaluation metrics and a confusion matrix will be printed for the testing set.

## Results

Though my CNN model that leveraged FastText word embeddings took slightly more CPU time to train (3 min 23 s) compared to the simple Random Forest model (1 min 8 s), it boasted a higher accuracy (98.70% vs 98.24%), as well as a higher recall, precision, and f1 score.

Below is a comparison of the two models:

| Model         | Feature Extraction Method | Training Time (CPU) | Accuracy | F1 Score | Recall | Precision |  
|---------------|---------------------------|---------------------|----------|----------|--------|-----------|
| Random Forest | TF-IDF                    | 1min 8s             | 98.24%   | 98.24%   | 98.24% | 98.24%    |
| CNN           | FastText Word Embeddings  | 3min 30s            | 98.70%   | 98.70%   | 98.72% | 98.69%    |