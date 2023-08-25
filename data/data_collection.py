import logging
import pandas as pd
import zipfile


# Reads the raw email data
# Input: path and a boolean, Output: pandas datafram
def read_emails(path, spam):
    df = pd.DataFrame(columns=['Text', 'Label'])
    with zipfile.ZipFile(path, 'r') as zip:
        for file in zip.namelist():
            f = zip.open(file)
            subject_found = False
            text = ''
            for line in f.readlines():
                line = line.decode('latin-1')
                if subject_found:
                    text += line
                else:
                    if line.startswith('Subject:'):
                        subject_found = True
                        text += line.replace('Subject:', '')
            if spam:
                df.loc[len(df.index)] = [text, 1]
            else:
                df.loc[len(df.index)] = [text, 0]
    return df


# Reads and processes the SpamAssassin data
# Input: paths to the ham, hard ham, and spam files, Output: pandas dataframe
def process_spam_assassin(ham_path, spam_path):
    # Create dataframes using the read_emails() method for the (easy) ham emails,
    # the hard ham emails, and the spam emails
    ham_df = read_emails(ham_path, False)
    # hard_ham_df = read_emails(hard_ham_path, False)
    spam_df = read_emails(spam_path, True)

    # Concatenate the previous dataframes:
    sa_df = pd.concat([ham_df, spam_df], axis=0, ignore_index=True)

    return sa_df


# Reads and processes the Enron data
# Input: paths to the enron CSV file, Output: pandas dataframe
def process_enron(enron_path):
    # load enron csv
    enron_df = pd.read_csv(enron_path)

    # standardize the Enron data to match the SpamAssassin data
    enron_df.rename(columns={'Spam/Ham': 'Label'}, inplace=True)
    enron_df['Label'] = enron_df['Label'].map({'spam': 1.0, 'ham': 0.0})
    enron_df['Text'] = enron_df['Subject'].map(str) + '. ' + enron_df['Message'].map(str)
    enron_df = enron_df.drop(columns=['Message ID', 'Date', 'Subject', 'Message'])

    return enron_df


def main():
    logging.info('Started data collection')

    # Step 1: Process SpamAssassin data
    ham_path = r'/Users/anastasiaarsky/PycharmProjects/production_repo/data/data_files/spam_assassin/ham.zip'
    spam_path = r'/Users/anastasiaarsky/PycharmProjects/production_repo/data/data_files/spam_assassin/spam.zip'
    sa_df = process_spam_assassin(ham_path, spam_path)

    # Step 2: Process Enron Spam data
    enron_path = r'/Users/anastasiaarsky/PycharmProjects/production_repo/data/data_files/enron_spam.zip'
    enron_df = process_enron(enron_path)

    # Step 3: Merge the SpamAssassin df with the Enron df
    data = pd.concat([sa_df, enron_df], axis=0, ignore_index=True)

    # Step 4: Save the data to a zipped CSV file
    compression_opts = dict(method='zip',
                            archive_name='unprocessed_data.csv')
    data.to_csv(r'/Users/anastasiaarsky/PycharmProjects/production_repo/data/data_files/unprocessed_data.zip',
                index=False,
                compression=compression_opts)

    logging.info('Finished data collection, unprocessed data can be found at data/data_files/unprocessed_data.csv.zip')

    # Step 5: Return the data dataframe
    return data


if __name__ == "__main__":
    main()
