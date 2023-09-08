# Results

This report will outline the results of my models throughout various stages in the model selection process. 

## Part 1: Results after the Reproduction of Available Solutions

My goal at this stage was to research publicly available solutions for both the Enron Spam and Spam Assassin datasets, and reproduce their results. I settled on experimenting with a **Random Forest Classifier** and a **Recurrent Neural Network** (LSTM) model as those two models seemed to outperform any others for both datasets.

### Results    

| Model	                          | Accuracy | F1 Score | 	Recall | Precision |  
|---------------------------------|----------|----------|---------|-----------|  
| Random Forest                   | 	0.981   | 0.981    | 	0.981  | 0.982     |  
| Recurrent Neural Network (LSTM) | 	0.981   | 0.981    | 	0.981  | 0.982     |  

Both models obtained identical accuracy, F1 score, recall, and precision metrics. The accuracy of both models was 98.1%, which is very high.

### Analysis
When considering the time taken train, the Random Forest model outperformed the RNN (LSTM) model. The Random Forest model took 1 minute and 37 seconds to run, while the RNN model took around 1 hour and 37 minutes. So the RNN model took rougly 100 times the amount of time to obtain the exact same results.

### Conclusion
Moving forward, I decided to first explore other ensemble methods, specifically boosting methods like XGBoost/AdaBoost or even CatBoost. Ensemble methods are less complex and faster to train compared to deep learning methods, so they are preferred if they are able to achieve comparable results.

Next, I wanted to experiment further with deep learning models. I first wanted to tune the hyperparameters for the RNN model I implemented, as well as look into adding additional hidden layers to improve its results. I also considered implementing a Convolutional Neural Network (CNN) as they are faster than RNNs and highly proficient in extracting meaningful patterns, which is ideal for spam detection.

Finally, I wanted to experiment with other forms of feature extraction. At this stage, I had tried TF-IDF Vectorization and a modified Bag of Words model, but I also wanted to consider Count Vectorization, Word2Vec, and possibly Word Embeddings.

## Part 2: Results after Experimentation on Various Ensemble Methods

In this stage, I wanted to experiment with other ensemble methods and forms of feature extraction.

So far, I had tried TF-IDF Vectorization (on the word level) and a modified Bag of Words model, but I also wanted to consider TF-IDF Vectorization on the ngram and character levels, as well as Count Vectorization.

I also wanted to explore boosting methods (specifically XGBoost), and compare the results to my previously used bagging method (Random Forest).

### Results

Before Hyperparameter Tuning:  

| Model	        | Feature Extraction Method | Accuracy | F1 Score | 	Recall | Precision |  
|---------------|---------------------------|----------|----------|---------|-----------|  
| Random Forest | Count Vectorization       | 	0.981   | 0.981    | 	0.981  | 0.981     |  
| Random Forest | Word Level TF-IDF         | 	0.983   | 0.983    | 	0.983  | 0.983     |  
| Random Forest | N-gram Level TF-IDF       | 	0.915   | 0.915    | 	0.917  | 0.922     |  
| Random Forest | Char Level TF-IDF         | 	0.971   | 0.971    | 	0.971  | 0.971     | 
| XGBoost       | Count Vectorization       | 	0.972   | 0.972    | 	0.973  | 0.972     |  
| XGBoost       | Word Level TF-IDF         | 	0.970   | 0.970    | 	0.971  | 0.970     |  
| XGBoost       | N-gram Level TF-IDF       | 	0.891   | 0.890    | 	0.893  | 0.898     |  
| XGBoost       | Char Level TF-IDF         | 	0.978   | 0.978    | 	0.978  | 0.978     | 
| XGBoost       | Char Level TF-IDF (Tuned) | 	0.982   | 0.982    | 	0.983  | 0.982     |

After Hyperparameter Tuning:

| Model	        | Feature Extraction Method | Accuracy | F1 Score | 	Recall | Precision |  
|---------------|---------------------------|----------|----------|---------|-----------|  
| Random Forest | Count Vectorization       | 	0.980   | 0.980    | 	0.980  | 0.980     |  
| Random Forest | Word Level TF-IDF         | 	0.980   | 0.980    | 	0.980  | 0.979     | 
| XGBoost       | Char Level TF-IDF         | 	0.982   | 0.982    | 	0.983  | 0.982     |


### Analysis

Overall, I found that the un-tuned Random Forest model using both TF-IDF (on the word level) and Count Vectorization, as well as tuned XGBoost model using TF-IDF (on the character level), produced approximately equal results - between 98.1-98.3% accuracy.

Furthermore, I found that TF-IDF feature extraction on the n-gram level resulted in a less accurate model for both Random Forest and XGBoost models.

### Conclusion

I decided to go ahead with the untuned Random Forest model using word-level TF-IDF as my final Machine Learning model as it had the highest accuracy (98.3%).


## Part 3: Results after Prototyping Deep Learning Models

My goal for this stage was to create a working implementations of a Spam Classification prototype using a Bidirectional Long-Term Short Memory (BiLSTM) model, which is a type of RNN (Recurrent Neural Network). 

Whereas simple RNNs only learn from the immediately preceding data, LSTMs keep track not just the immediately preceding data, but the earlier data too. This allows them to learn from data that is far away from its current position. Furthermore, the typical state in an LSTM relies on the past and the present events. However, there can be situations where a prediction depends on the past, present, and future events. So in the context of email spam detection, whether or not an email is a spam can depend on future words in the email. 

I also experimented with different feature extraction methods: a Continuous Bag of Words model, and the pretrained GloVe Word Embeddings. Leveraging pre-trained word embeddings is a form deep transfer learning, and is commonly used in NLP.

### Results

After experimenting with different variations of my Bi-LSTM models (adding dropout layers, additional Bi-LSTM layers, playing around with learning rate, adding more epochs, hyperparameter tuning), these were the results for the highest performing models for each type of feature extraction:

| Model   | Feature Extraction Method | Accuracy | Training Time (CPU) |
|---------|---------------------------|----------|---------------------|
| Bi-LSTM | GloVe Word Embeddings     | 0.9603   | 27m 34s             |   
| Bi-LSTM | Continuous Bag of Words   | 0.9782   | 16m 10s             | 	

### Analysis

The Bi-LSTM model that used a continuous bag of words for feature extraction outperformed the Bi-LSTM model that used GloVE word embeddings in both accuracy and CPU training time.  

However, this Bi-LSTM still underperformed compared to my previously used Random Forest model (97.82% vs 98.27% accuracy).


## Part 4: Results after Scaling 

My goal in this stage was to scale my prototype to handle a larger volume of data.

To do this, I decided to implement a Convolutional Neural Network (CNN) in place of a BiLSTM.
CNNs are built for processing images, allowing them the capability of handling large amounts of data. This allows them to be considerably faster than BiLSTMs. They are also highly proficient in extracting meaningful patterns, which is ideal for spam detection.

Once again, I analyzed a Continuous Bag of Words model and the pretrained GloVe Word Embeddings for my feature extraction. I also be used an additional deep transfer learning feature extraction approach by leveraging pretrained FastText word embeddings.

### Results:

After experimenting with different variations of my CNN models (adding dropout layers, additional Bi-LSTM layers, playing around with learning rate, adding more epochs, hyperparameter tuning), these were the results for the highest performing models for each type of feature extraction:

| Model | Feature Extraction Method | Accuracy | Training Time (CPU) |
|-------|---------------------------|----------|---------------------|
| CNN   | 	GloVe Word Embeddings    | 	95.66%  | 1min 49s            |
| CNN   | 	Continuous Bag of Words	 | 	97.96%  | 8min 46s            |
| CNN   | 	FastText Word Embeddings | 	98.70%  | 3min 30s            |

### Analysis

I found that my previously-used methods of feature extraction (GloVe word embeddings, Continuous Bag of Words) did not result in a higher accuracy than a simple Random Forest model. So, I did some research and decided to utilize FastText word embeddings instead, which resulted in a testing accuracy of 98.70% with my CNN model.


## Part 5: Final Results

The final models I settled on were a Random Forest Model (for the traditional ML model), and a CNN (for the modern DL model).

### Results:

| Model         | Feature Extraction Method | Training Time (CPU) | Accuracy | F1 Score | 	Recall | Precision |
|---------------|---------------------------|---------------------|----------|----------|---------|-----------|
| Random Forest | TF-IDF                    | 1min 8              | 98.27%   | 98.26%   | 98.27%  | 98.25%    | 
| CNN           | FastText Word Embeddings	 | 3min 30s	           | 98.70%   | 98.70%   | 98.72%  | 98.69%    |

### Analysis:

Though my CNN model that leveraged FastText word embeddings took slightly more CPU time to train (3 min 23 s) compared to the simple Random Forest model using TF-IDF (1 min 8 s), it boasted a higher accuracy (98.70% vs 98.27%), as well as a higher recall, precision, and f1 score.

