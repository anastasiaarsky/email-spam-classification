# This function triggers an endpoint to a trained model on SageMaker

import boto3
import json
import pickle
import re
import unicodedata
from keras.utils import pad_sequences
from nltk.corpus import stopwords

# Cache the stopwords object to speed up stopword removal in exclude_stopwords method
cached_stopwords = set(stopwords.words("english"))


# Replace URLs with the string 'url', emails with the string 'email', and numbers with the string 'number' (Step 3)
# Input: string, Output: string
def replace_url_email_number(text):
    # replaces URLs with the string 'url'
    text = re.sub(
        r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])',
        ' url ', text)
    # replaces emails with the string 'email'
    text = re.sub(r'\S*@\S*\s?', ' email ', text)
    # replaces numbers with the string 'number'
    text = re.sub(r'[0-9]+', ' number ', text)
    return text


# Exclude non-ASCII characters (Step 4)
# Input: string, Output: string
def exclude_non_ascii(text):
    words = text.split()
    words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
    return ' '.join(words)


# Exclude certain unwanted punctuation (Step 5)
# Unwanted punctuation: " # % & ( ) * + - / < = > @ [ \ ] ^ _ ` { | } ~
# Input: string, Output: string
def exclude_unwanted_punctuation(text):
    to_exclude = r'"#%&()*+,-/:;<=>@[\]^_`{|}~'
    translator = str.maketrans(' ', ' ', to_exclude)
    return text.translate(translator)


# Ensure that certain punctuation is recognized as its own token by added a space in front and behind of it (Step 6)
# Important punctuation = . ! ? $
# Input: string, Output: string
def add_punctuation_token(text):
    to_tokenize = '.!?$'
    text = re.sub(r'([' + to_tokenize + '])', r' \1 ', text)
    return text


# Exclude stopwords (Step 7)
# Input: string, Output: string
def exclude_stopwords(text):
    words = text.split()
    text = ' '.join([word for word in words if word not in cached_stopwords])
    return text


# Remove extra whitespaces and newlines (Step 8)
# Input: string, Output: string
def remove_extra_space(text):
    text = re.sub(r'\n|\r', '', text)
    text = re.sub(r' +', ' ', text)
    return text


# Calls all the above text preprocessing functions and returns a cleaned string
# Input: string, Output: string
def preprocess(text):
    # Step 1: Lowercase the data
    clean_text = text.str.lower()

    # Step 2: Replace URLs with the string 'url', emails with the string 'email', and numbers with the string 'number'
    clean_text = replace_url_email_number(clean_text)

    # Step 3: Exclude non-ASCII characters
    clean_text = exclude_non_ascii(clean_text)

    # Step 4: Exclude unwanted punctuation
    clean_text = exclude_unwanted_punctuation(clean_text)

    # Step 5: Ensure that certain punctuation is recognized as its own token
    clean_text = add_punctuation_token(clean_text)

    # Step 6: Exclude stopwords
    clean_text = exclude_stopwords(clean_text)

    # Step 7: Remove extra whitespaces and newlines
    clean_text = remove_extra_space(clean_text)

    # Step 8: Return the cleaned text
    return clean_text


def vectorize(text):
    # Load the tokenizer from S3
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='bucket_name', Key='tokenizer.pickle')
    file_content = response['Body'].read()
    tokenizer = pickle.loads(file_content)

    # Convert the text to a sequence
    text_seq = tokenizer.texts_to_sequences([text])

    # Pad the sequence so it is the appropriate length (500)
    text_pad = pad_sequences(text_seq, maxlen=500)

    return text_pad


# primary function
def lambda_handler(event, context):
    # lambda receives the input from the web app as an event in json format
    email = event['body']
    preprocessed_email = preprocess(email)
    vectorized_email = vectorize(preprocessed_email)
    payload = {"instances": [vectorized_email]}

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the text we were given
    response = runtime.invoke_endpoint(EndpointName='endpoint_name',
                                       # The name of the endpoint we created
                                       ContentType='application/json',
                                       # The data format that is expected by the Sagemaker model
                                       Body=json.dumps(payload))

    # The response is an HTTP response whose body contains the result of our inference
    output = json.loads(response['Body'].read().decode('utf-8'))
    prob = output[0]['prob'][0] * 100
    label = output[0]['label'][0].split('__label__')[1].map({'spam': 1.0, 'ham': 0.0})
    output = 'The predicted label is {} with a probability of {:.1f}%'.format(label, prob)

    # we return the output in a format expected by API Gateway
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'},
        'body': output
    }
