# Deployment Steps

### Part 1: Train and save the trained CNN (Keras model):
1. Clone the repository:
```bash
git clone https://github.com/anastasiaarsky/production_repo.git
cd production_repo
```

2. Collect and preprocess the data:  
```bash 
python -m src.data.make_dataset
```

3. Train the CNN model using the preprocessed data:
```bash 
python -m src.models.cnn_model --train
```  
The trained model will be saved at: models/cnn_model.h5.

### Part 2: Deploy my trained CNN using Amazon SageMaker

1. Upload the trained model into an S3 bucket
2. Deploy the trained model using SageMaker Serverless Endpoint

### Part 3: Connect the SageMaker Endpoint to a serverless function using AWS Lambda and API Gateway

1. Create a [Lambda function](../src/deployment/lambda_function.py) that:  
    a. Does some preprocessing to transform HTML input data into a CSV file that the endpoint is expecting  
   - Note: Upload the tokenizer saved in models/tokenizer.pickle to S3 since this tokenizer will be used to preprocess the input data by the Lambda function  
   
    b. Returns the inference output in a format expected by API Gateway  
2. Use API Gateway to build an HTTP endpoint that will allow our back-end Lambda function to talk to our front-end web app 
3. Host the app on an S3 bucket

### Resources:
[Deploy trained Keras or TensorFlow models using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/deploy-trained-keras-or-tensorflow-models-using-amazon-sagemaker/)

[Deploy an NLP classification model with Amazon SageMaker and Lambda](https://austinlasseter.medium.com/deploy-an-nlp-classification-model-with-amazon-sagemaker-and-lambda-cd5ea6339781)
