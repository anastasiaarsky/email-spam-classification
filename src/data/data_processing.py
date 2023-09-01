from src.data import data_collection

import os
import re
import unicodedata

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


def main():
    # Step 1: Get data from data_collection module
    # Full Text column
    df = data_collection.main()

    # Step 2: Lowercase the data
    df['Clean_Text'] = df['Text'].str.lower()

    # Step 3: Replace URLs with the string 'url', emails with the string 'email', and numbers with the string 'number'
    df['Clean_Text'] = df['Clean_Text'].apply(replace_url_email_number)

    # Step 4: Exclude non-ASCII characters
    df['Clean_Text'] = df['Clean_Text'].apply(exclude_non_ascii)

    # Step 5: Exclude unwanted punctuation
    df['Clean_Text'] = df['Clean_Text'].apply(exclude_unwanted_punctuation)

    # Step 6: Ensure that certain punctuation is recognized as its own token
    df['Clean_Text'] = df['Clean_Text'].apply(add_punctuation_token)

    # Step 7: Exclude stopwords
    df['Clean_Text'] = df['Clean_Text'].apply(exclude_stopwords)

    # Step 8: Remove extra whitespaces and newlines
    df['Clean_Text'] = df['Clean_Text'].apply(remove_extra_space)

    # Step 9: Create a new dataframe that just contains the Label and (clean) Text columns
    clean_df = df[['Label', 'Clean_Text']].copy()
    clean_df.rename(columns={'Clean_Text': 'Text'}, inplace=True)

    # Step 10: Save the data to a zipped CSV file
    output_path = os.path.join(os.path.dirname(__file__), '../../data/processed_data.zip')
    compression_opts = dict(method='zip', archive_name='processed_data.csv')
    clean_df.to_csv(output_path, index=False, escapechar='\\', compression=compression_opts)

    print('Finished data processing, processed data can be found at data/processed_data.csv.zip')

    # Step 11: Return the clean dataframe
    return clean_df


if __name__ == "__main__":
    main()
