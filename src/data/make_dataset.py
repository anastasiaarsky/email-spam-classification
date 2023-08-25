import data_processing


def main():
    # call data_processing.py, which first calls data_collection.py to collect the data, then processes
    # and outputs the dataset as a pandas dataframe (located at data/processed_data.zip)
    dataset = data_processing.main()

    return dataset


if __name__ == "__main__":
    main()
