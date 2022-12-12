"""
    AUTHORS:
        Adriel Isaiah V. Amoguis
        Gian Joseph B. Madrid

    Personalied Content-Based Filtering (CBF) on Microsoft MIND Dataset
    TF-IDF Vectorization and Categorical Approach
"""

# Library Imports
import argparse

import os
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download as nltk_dl

nltk_dl('punkt')
nltk_dl('wordnet')
nltk_dl('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error as mse
from scipy import sparse as sps

import spacy
# from spacy.cli import download
# download("en_core_web_sm")
spacy.load("en_core_web_sm")

import string

rmse = lambda y_true, y_pred: mse(y_true, y_pred, squared=False)

# Define Recommender Class
class PersonalizedCBF:
    def __init__(self, mind_news_df, mind_behaviors_df, stopwords=set(spacy.lang.en.stop_words.STOP_WORDS), verbose=False):
        self.mind_news_df = mind_news_df
        self.mind_behaviors_df = mind_behaviors_df
        self.stopwords = stopwords
        self.verbose = verbose

        self._init_dataframes()

        self.user_ids = self.mind_behaviors_df['user_id'].unique().tolist()
        self.article_ids = self.mind_news_df['news_id'].unique().tolist()

        self.news_feature_vectors = None

    def _init_dataframes(self):
        if self.verbose:
            print("News DataFrame:", self.mind_news_df.shape)
            print(self.mind_news_df.head())
            print("Behaviors DataFrame:", self.mind_behaviors_df.shape)
            print(self.mind_behaviors_df.head())

        self.mind_behaviors_df.drop_duplicates(subset=['impression_id'], inplace=True)
        self.mind_news_df.drop_duplicates(subset=["title"], inplace=True)

        self.mind_news_df.dropna(inplace=True)
        self.mind_behaviors_df.dropna(inplace=True)

        if self.verbose:
            print("News DataFrame after init:", self.mind_news_df.shape)
            print(self.mind_news_df.head())
            print("Behaviors DataFrame after init:", self.mind_behaviors_df.shape)
            print(self.mind_behaviors_df.head())

    def _preprocess(self, news_df):
        punctuation_translate_table = dict((ord(char), None) for char in string.punctuation)
        if self.verbose: print("Punctuation Translate Table:", punctuation_translate_table)

        def text_preproc(df, col):
            lemmatizer = WordNetLemmatizer()
            processed = []
            for sentence in df[col]:
                sentence = sentence.translate(punctuation_translate_table)
                sentence = sentence.lower()
                sentence = word_tokenize(sentence)
                sentence = [lemmatizer.lemmatize(word) for word in sentence if word not in self.stopwords]
                sentence = ' '.join(sentence)
                processed.append(sentence)
            df["proc_{}".format(col)] = processed

        text_preproc(news_df, 'title')
        text_preproc(news_df, 'abstract')

        if self.verbose:
            print("Preprocessed News Titles:", news_df.shape)
            print(news_df.head()["title"])

    def _tfidf_vectorize(self, news_df, min_df=0, max_df=0.7):
        tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)

        title_abstract = []
        for _, row in news_df.iterrows():
            title_abstract.append(" ".join([row["proc_title"], row["proc_abstract"]]))
            
        news_df["title_abstract_feature_string"] = title_abstract

        tfidf_vectors = tfidf_vectorizer.fit_transform(news_df['title_abstract_feature_string'])

        if self.verbose: print("TF-IDF Vectors:", tfidf_vectors.shape)

        return tfidf_vectors

    def _one_hot_encode(self, news_df, colname):
        one_hot_encoder = OneHotEncoder()
        one_hot_vectors = one_hot_encoder.fit_transform(news_df[colname].values.reshape(-1, 1))

        if self.verbose: print("{} One-Hot Vectors:".format(colname), one_hot_vectors.shape)

        return one_hot_vectors

    def _build_item_profiles(self, news_df):
        tfidf_vectors = self._tfidf_vectorize(news_df)
        category_vectors = self._one_hot_encode(news_df, "category")
        subcategory_vectors = self._one_hot_encode(news_df, "subcategory")

        cat_subcat_vectors = sps.hstack([category_vectors, subcategory_vectors])
        news_feature_vectors = sps.hstack([cat_subcat_vectors, tfidf_vectors])

        return news_feature_vectors

    def _build_user_profiles(self, news_feature_vectors):
        user_profiles = {}
        for user_id in self.user_ids:
            user_profiles[user_id] = sps.lil_matrix((1, news_feature_vectors.shape[1]), dtype=np.uint8)

        for i, row in self.mind_behaviors_df.iterrows():
            if self.verbose: print("\rBuilding user profiles: {} / {} | {:.2f}%".format(i, self.mind_behaviors_df.shape[0], i / self.mind_behaviors_df.shape[0] * 100), end="")
            user_id = row['user_id']
            impression_news_ids = row['impressions'].split()
            history = row['history'].split()

            for impression in impression_news_ids:
                news_id, imp = impression.split('-')
                imp = int(imp)
                try:
                    news_idx = self.article_ids.index(news_id)
                    user_profiles[user_id] += news_feature_vectors[news_idx, :] * imp
                except ValueError:
                    pass

            for news_id in history:
                try:
                    news_idx = self.article_ids.index(news_id)
                    user_profiles[user_id] += news_feature_vectors[news_idx, :]
                except ValueError:
                    pass

        return user_profiles

    def build_user_profile(self, user_id):
        user_profile = sps.csr_matrix((1, self.news_feature_vectors.shape[1]), dtype=np.uint8)

        user_behaviors = self.mind_behaviors_df[self.mind_behaviors_df['user_id'] == user_id]
        for _, row in user_behaviors.iterrows():
            impression_news_ids = row['impressions'].split()
            history = row['history'].split()

            for impression in impression_news_ids:
                news_id, imp = impression.split('-')
                imp = int(imp)
                try:
                    news_idx = self.article_ids.index(news_id)
                    user_profile += self.news_feature_vectors[news_idx, :] * imp
                except ValueError:
                    pass

            for news_id in history:
                try:
                    news_idx = self.article_ids.index(news_id)
                    user_profile += self.news_feature_vectors[news_idx, :] * 1
                except ValueError:
                    pass

        return user_profile

    def fit(self):
        print("Preprocessing...")
        self._preprocess(self.mind_news_df)
        print("Building item profiles...")
        self.news_feature_vectors = self._build_item_profiles(self.mind_news_df)

        if self.verbose:
            print("Feature Vectors Shape:")
            print(self.news_feature_vectors.shape)

    def _get_article_entry(self, article_id):
        return self.mind_news_df[self.mind_news_df['news_id'] == article_id]

    def recommend(self, user_id, top_n=10):
        user_profile = self.build_user_profile(user_id)
        scores = user_profile.dot(self.news_feature_vectors.T).toarray().ravel()
        top_news = np.array(self.article_ids)[np.argsort(-scores)]

        articles = [self._get_article_entry(a_id) for a_id in top_news[:top_n]]

        return top_news, articles

    def evaluate(self, n, percentage=0.2):
        print("Evaluating...")
        hits = 0
        n_eval = int(self.mind_behaviors_df.shape[0] * percentage)
        for i, row in self.mind_behaviors_df[:n_eval].iterrows():
            print("\rEvaluating: {} / {} | {:.2f}%".format(i, self.mind_behaviors_df[:n_eval].shape[0], i / self.mind_behaviors_df[:n_eval].shape[0] * 100), end="")
            user_id = row['user_id']
            clicked_news_ids = row['history'].split()

            # Ensure that the clicked_news_ids are in the article_ids
            clicked_news_ids = [news_id for news_id in clicked_news_ids if news_id in self.article_ids]

            top_news, _ = self.recommend(user_id, top_n=n)
            for news_id in top_news:
                if news_id in clicked_news_ids:
                    hits += 1
                    break

        print("\nRecall@{}: {:.4f}".format(percentage, hits / self.mind_behaviors_df[:n_eval].shape[0]))

    def k_fold_cross_validation(self, k):
        print("Performing k-fold cross validation...")
        n = int(self.mind_behaviors_df.shape[0] / k)
        for i in range(k):
            print("Fold {} / {}".format(i + 1, k))
            
            eval_news_train_df = self.mind_news_df[i*n:(i+1)*n]
            eval_news_test_df = self.mind_news_df[~self.mind_news_df.index.isin(self.eval_news_train_df.index)]

            print("Preprocessing...")
            self._preprocess(eval_news_train_df)
            self._preprocess(eval_news_test_df)
            print("Building item profiles...")
            train_features = self._build_item_profiles(eval_news_train_df)
            test_features = self._build_item_profiles(eval_news_test_df)
            print("Building user profiles on train set...")
            train_user_profiles = self._build_user_profiles(train_features)
            print("Evaluating on test set...")
            

def main(args):
    DATASET_ROOT = args.data_dir
    VERBOSE = args.verbose
    SAVE_FEATURES = args.save_features
    LOAD_FEATURES = args.load_features
    OUTPUT_FILE = args.output
    EVAL = args.evaluate

    news_df = pd.read_csv(os.path.join(DATASET_ROOT, "news.tsv"), delimiter="\t", names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])
    behaviors_df = pd.read_csv(os.path.join(DATASET_ROOT, "behaviors.tsv"), delimiter="\t", names=["impression_id", "user_id", "time", "history", "impressions"])

    if LOAD_FEATURES != '':
        news_feature_vectors = sps.load_npz(LOAD_FEATURES)
        pCBF = PersonalizedCBF(news_df, behaviors_df, verbose=VERBOSE)
        pCBF.news_feature_vectors = news_feature_vectors
    else:

        # Initialize Personalized CBF
        pCBF = PersonalizedCBF(news_df, behaviors_df, verbose=VERBOSE)
        pCBF.fit()

        # Save article feature vectors to file
        if SAVE_FEATURES:
            sps.save_npz(OUTPUT_FILE, pCBF.news_feature_vectors)

    if EVAL:
        pCBF.evaluate(10)
        return

    while True:
        print("Run predictions for a user ID, type 'r' for random user, or enter 'exit' to quit.")
        user_id = input("Enter user id: ")
        if user_id == "exit":
            break
        elif user_id == "r":
            user_id = np.random.choice(pCBF.user_ids)
            print("Random user ID:", user_id)

        if user_id not in pCBF.user_ids:
            print("User ID not found in dataset.")
            continue

        _, articles = pCBF.recommend(user_id)
        print("Recommended articles:\n")

        for article in articles:
            print("Title:", article['title'].values[0])
            print("Abstract:", article['abstract'].values[0])
            print()

    # # Predict
    # rand_user_id = np.random.choice(pCBF.user_ids)
    # _, articles = pCBF.recommend(rand_user_id)
    # print("Recommended articles:\n")

    # for article in articles:
    #     print("Title:", article['title'].values[0])
    #     print("Abstract:", article['abstract'].values[0])
    #     print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir', type=str, help='Root directory of the MS MIND Dataset', default="MINDsmall_train")
    parser.add_argument('-l','--load-features', type=str, help='Load article feature vectors from file', default='')
    parser.add_argument('-s','--save-features', action="store_true", help='Save article feature vectors to file', default=False)
    parser.add_argument('-o','--output', type=str, help='Output filename of article feature vectors (.npz)', default="article_features.npz")
    parser.add_argument('-v','--verbose', action="store_true", help='Verbose mode', default=False)
    parser.add_argument('-e','--evaluate', action="store_true", help='Evaluate model', default=False)
    args = parser.parse_args()
    main(args)