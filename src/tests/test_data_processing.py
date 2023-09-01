import pandas as pd
import unittest
from unittest.mock import patch

from src.data import data_processing


class TestDataProcessing(unittest.TestCase):
    def test_replace_url_email_number(self):
        # Test the function that replaces URLs, emails, and numbers
        text = 'go to https://lists.sourceforge.net/lists/listinfo/ on Jul 12, 2009 and log in with the email ' \
               'anigl@yahoo.com to http://github.com'
        result_text = data_processing.replace_url_email_number(text)
        expected_text = 'go to  url  on Jul  number ,  number  and log in with the email  email to  url '
        self.assertEqual(result_text, expected_text)

    def test_exclude_non_ascii(self):
        # Test the function that excludes non-ASCII characters
        text = 'café भारत ∼300'
        result_text = data_processing.exclude_non_ascii(text)
        expected_text = 'cafe  300'
        self.assertEqual(result_text, expected_text)

    def test_exclude_unwanted_punctuation(self):
        # Test the function that excludes unwanted punctuation
        text = '{help} | number% +more = infinite!'
        result_text = data_processing.exclude_unwanted_punctuation(text)
        expected_text = 'help  number more  infinite!'
        self.assertEqual(result_text, expected_text)

    def test_add_punctuation_token(self):
        # Test the function that adds a space before and after certain punctuation (. ! ? $)
        text = 'end. soon! why? $number'
        result_text = data_processing.add_punctuation_token(text)
        expected_text = 'end .  soon !  why ?   $ number'
        self.assertEqual(result_text, expected_text)

    def test_exclude_stopwords(self):
        # Test the function that excludes stopwords
        text = 'i went to get her the candy so that she would be happy'
        result_text = data_processing.exclude_stopwords(text)
        expected_text = 'went get candy would happy'
        self.assertEqual(result_text, expected_text)

    def test_remove_extra_space(self):
        # Test the function that removes extra whitespaces and newlines
        text = '\ni am     very\r  upset\r\n'
        result_text = data_processing.remove_extra_space(text)
        expected_text = 'i am very upset'
        self.assertEqual(result_text, expected_text)


if __name__ == "__main__":
    unittest.main()