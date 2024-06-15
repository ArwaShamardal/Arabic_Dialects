import string
import re
import pandas as pd


class Preprocess():
    def __init__(self, df):
        # is data is of type string make it a dataframe
        if isinstance(df, str):
            df = pd.DataFrame([df], columns=['text'])
        self.df = df

    def __remove_newline(self, text):
        return text.replace('\n', ' ')

    def __remove_punctuation(self, text):
        arabic_punctuations = '؛،؟”“'
        all_punctuations = string.punctuation + arabic_punctuations
        return text.translate(str.maketrans('', '', all_punctuations))

    def __remove_digits(self, text):
        arabic_digits = '٠١٢٣٤٥٦٧٨٩'
        all_digits = string.digits + arabic_digits
        return text.translate(str.maketrans('', '', all_digits))

    def __remove_mentions(self, text):
        return ' '.join([word for word in text.split() if not word.startswith('@')])

    def __remove_links(self, text):
        return ' '.join([word for word in text.split() if not word.startswith('http')])

    def __remove_emojis(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   u"\U00002500-\U00002BEF"
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def __remove_english(self, text):
        english_alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return ' '.join([word for word in text.split() if not any(char in english_alphabets for char in word)])

    def preprocess(self):
        self.df['text'] = self.df['text'].apply(self.__remove_newline)
        self.df['text'] = self.df['text'].apply(self.__remove_mentions)
        self.df['text'] = self.df['text'].apply(self.__remove_links)
        self.df['text'] = self.df['text'].apply(self.__remove_emojis)
        self.df['text'] = self.df['text'].apply(self.__remove_punctuation)
        self.df['text'] = self.df['text'].apply(self.__remove_digits)
        self.df['text'] = self.df['text'].apply(self.__remove_english)
        return self.df


# NOTE:: use case
# it changes the original dataframe
# df_preprocessed = Preprocess(df_copy)
# df_preprocessed = df_preprocessed.preprocess()
# df_preprocessed.head()
