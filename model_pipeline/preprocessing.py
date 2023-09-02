import re
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from model_pipeline.utils import get_average_area

# nltk.download('punkt')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

"""
    This file will store all the preprocessing related functions
"""
# Setting up stopwords for Text Processing
stopwords_list = set(stopwords.words('english'))

# Custom Stopwords list
custom_stopwords = ["i", "project", "living", "home", 'apartment', "pune", "me", "my", "myself", "we", "our",
                    "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd", "your", "yours",
                    "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
                    "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves",
                    "what", "which", "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are",
                    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
                    "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
                    "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
                    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
                    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
                    "don", "don't", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
                    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
                    "needn", "shan", "shan't", "shouldn", "wasn", "weren", "won", "rt", "rt", "qt", "for", "the",
                    "with", "in", "of", "and", "its", "it", "this", "i", "have", "has", "would", "could", "you",
                    "a", "an", "be", "am", "can", "edushopper", "will", "to", "on", "is", "by", "ive", "im",
                    "your", "we", "are", "at", "as", "any", "ebay", "thank", "hello", "know", "need", "want",
                    "look", "hi", "sorry", "http", "https", "body", "dear", "hello", "hi", "thanks", "sir",
                    "tomorrow", "sent", "send", "see", "there", "welcome", "what", "well", "us"]

stopwords_list.update(custom_stopwords)


def get_outlier_range(df, cname):
    try:
        print("Started Executing function utils.get_outlier_range")
        sorted(cname)
        low_limit = df[cname].quantile(0.25)
        high_limit = df[cname].quantile(0.95)
        print("Completed Executing function preprocessing.get_outlier_range")
    except Exception as e:
        print('Error in preprocessing.get_outlier_range function', e)
    else:
        return low_limit, high_limit


def text_preprocess(text):
    """
        text: a string
        return: modified initial string
    """
    try:
        print("Started Executing function utils.text_preprocess")

        text = text.replace("\d+", " ")  # removing digits
        text = re.sub(r"(?:\@|https?\://)\S+", '', text)  # removing mentions and urls
        text = text.lower()
        text = re.sub('[0-9]+', '', text)  # removing numeric characters
        text = re.sub("[/(){}\[\]\|@,;!]", ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub('[^0-9a-z #+_]', ' ', text)  # replace symbols which are in BAD_SYMBOLS_RE from text

        text = ' '.join([word for word in text.split() if word not in stopwords_list])
        text = text.strip()
        print("Completed Executing function preprocessing.text_preprocess")
    except Exception as e:
        print('Error in preprocessing.text_preprocess function', e)
    else:
        return text


def get_pos_counter(text, part_of_speech):
    """
    Returns the count for the given parts of speech tag

    NN - Noun
    VB - Verb
    JJ - Adjective
    RB - Adverb
    """
    try:
        print("Started Executing function utils.get_pos_counter")
        word_list = nltk.word_tokenize(text.lower())
        clean_word_list = [word for word in word_list if word not in stopwords_list]
        text = nltk.Text(clean_word_list)
        tag_pos = nltk.pos_tag(text)
        counts = Counter(tag for word, tag in tag_pos)
        print("Completed Executing function preprocessing.text_preprocess")
    except Exception as e:
        print('Error in preprocessing.get_pos_counter function', e)
    else:
        return counts[part_of_speech]


def preprocess_data(df):
    try:
        print("Started Executing function preprocessing.preprocess_data")
        df_final = pd.DataFrame()
        df_final['City'] = df['Location'].apply(lambda x: x.split(',')[0].strip())
        df_final['State'] = df['Location'].apply(lambda x: x.split(',')[1].strip())
        df_final['Country'] = df['Location'].apply(lambda x: x.split(',')[2].strip())

        regx_numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)")
        df_final['PropertyType'] = df['Propert Type'].apply(
            lambda x: regx_numbers.findall(x)[0] if len(regx_numbers.findall(x)) > 0 else 0)

        df_final['SubArea'] = df['Sub-Area'].apply(lambda x: x.capitalize().strip())
        df_final['CompanyName'] = df['Company Name'].apply(lambda x: x.capitalize())
        df_final['TownshipSocietyName'] = df['TownShip Name/ Society Name'].apply(lambda x: x.capitalize())
        df_final['Description'] = df['Description'].apply(lambda x: x.capitalize())

        regx_numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)")
        df_final['PropertyAreainSqFt'] = df['Property Area in Sq. Ft.'].apply(lambda x: get_average_area(str(x)))

        regx_numbers = re.compile(r"[-+]?(\d*\.\d+|\d+)")
        df_final['PriceInLakhs'] = df['Price in lakhs'].apply(lambda x: np.float(regx_numbers.findall(str(x))[0])
        if len(regx_numbers.findall(str(x))) > 0 else np.NaN)

        df_final['ClubHouse'] = df['ClubHouse'].apply(lambda x: x.lower().strip()).map({'yes': 1, 'no': 0})
        df_final['School/UniversityInTownship'] = df['School / University in Township '].apply(
            lambda x: x.lower().strip()).map({'yes': 1, 'no': 0})
        df_final['HospitalInTownShip'] = df['Hospital in TownShip'].apply(lambda x: x.lower().strip()).map(
            {'yes': 1, 'no': 0})
        df_final['MallInTownShip'] = df['Mall in TownShip'].apply(lambda x: x.lower().strip()).map({'yes': 1, 'no': 0})
        df_final['ParkJoggingTrack'] = df['Park / Jogging track'].apply(lambda x: x.lower().strip()).map(
            {'yes': 1, 'no': 0})
        df_final['SwimmingPool'] = df['Swimming Pool'].apply(lambda x: x.lower().strip()).map({'yes': 1, 'no': 0})
        df_final['Gym'] = df['Gym'].apply(lambda x: x.lower().strip()).map({'yes': 1, 'no': 0})
        df_final = df_final.dropna()
        print("Completed Executing function preprocessing.preprocess_data")
    except Exception as e:
        print('Error in preprocessing.preprocess_data function', e)
    else:
        return df_final


def create_features(df):
    try:
        # Treating outliers in the numeric columns
        clist = ['PropertyAreainSqFt']

        for col in clist:
            low_val, high_val = get_outlier_range(df, col)
            df[col] = np.where(df[col] > high_val, high_val, df[col])
            df[col] = np.where(df[col] < low_val, low_val, df[col])

        # Calculating Average Price based on SubArea
        df['PriceBySubArea'] = df.groupby('SubArea')['PriceInLakhs'].transform('mean')

        # Saving the mapping dict for inference use
        Price_by_Sub_Area = df.groupby('SubArea')['PriceInLakhs'].mean().to_dict()
        filename = 'output/price_by_sub_area.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(Price_by_Sub_Area, f)

        amenities_columns = ['ClubHouse',
                             'School/UniversityInTownship',
                             'HospitalInTownShip',
                             'MallInTownShip',
                             'ParkJoggingTrack',
                             'SwimmingPool',
                             'Gym']

        temp_df = df[amenities_columns]
        temp_df['AmenitiesScore'] = temp_df.sum(axis=1)
        df['AmenitiesScore'] = temp_df['AmenitiesScore']

        # Calculating Price By Amenities Score
        df['PriceByAmenitiesScore'] = df.groupby('AmenitiesScore')['PriceInLakhs'].transform('mean')

        # Saving the mapping dict for inference use
        price_by_amenities_score = df.groupby('AmenitiesScore')['PriceInLakhs'].mean().to_dict()
        filename = 'output/price_by_amenities_score.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(price_by_amenities_score, f)

        # NLP Text Processing to extract new features
        df['Description'] = df["Description"].astype(str).apply(text_preprocess)
        df['Noun_Counts'] = df['Description'].apply(lambda x: get_pos_counter(x, 'NN'))
        df['Verb_Counts'] = df['Description'].apply(lambda x: (get_pos_counter(x, 'VB') + get_pos_counter(x, 'RB')))
        df['Adjective_Counts'] = df['Description'].apply(lambda x: get_pos_counter(x, 'JJ'))

        # creating count vectors
        count_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10).fit(df['Description'])
        X = count_vectorizer.transform(df['Description'])
        ngram_df = pd.DataFrame(X.toarray(), columns=count_vectorizer.get_feature_names_out())
        df = pd.concat([df.reset_index(drop=True), ngram_df.reset_index(drop=True)], axis=1)

        # dump count vectorized object for inference purposes
        fileName = 'output/count_vectorizer.pkl'
        with open(fileName, 'wb') as f:
            pickle.dump(count_vectorizer, f)

        # selecting the final features ready for ML Models
        remove_columns_list = ['City', 'State', 'Country', 'SubArea', 'CompanyName', 'TownshipSocietyName',
                               'Description', 'every day']
        df_final = df.drop(remove_columns_list, axis=1)

        features_list = df_final.columns.tolist()
        final_feature_list = ['Property_Type', 'Property_Area_in_SqFt', 'Price_In_Lakhs', 'Club_House',
                              'School_University_In_Township',
                              'Hospital_In_Township', 'Mall_In_Township', 'Park_Jogging_Track', 'Swimming_Pool', 'Gym',
                              'Price_By_SubArea',
                              'Amenities_Score', 'Price_By_Amenities_Score', 'Noun_Counts', 'Verb_Counts',
                              'Adjective_Counts', 'boasts_elegant',
                              'elegant_towers', 'great_community', 'mantra_gold', 'offering_bedroom',
                              'quality_specification', 'stories_offering',
                              'towers_stories', 'world_class']

        raw_features_mapping = dict(zip(features_list, final_feature_list))
        fileName = 'output/raw_features_mapping.pkl'
        with open(fileName, 'wb') as f:
            pickle.dump(raw_features_mapping, f)

        fileName = 'output/feature_mapping.pkl'
        with open(fileName, 'wb') as f:
            pickle.dump(final_feature_list, f)

        df_final.columns = final_feature_list

    except Exception as e:
        print('Error in preprocessing.create_features function', e)
    else:
        return df_final
