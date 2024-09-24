"""Create vectorizers and train models, and perform 5-Fold cross validation with an additional feature of senti wordnet lexicon. Combine the best word and character n-gram TF-IDF vectors."""
from argparse import ArgumentParser
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from scipy.sparse import coo_matrix
from scipy.sparse import hstack


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def find_positive_negative_counts_in_Sentences(sentences, pos, neg):
    pos_words_in_sentences = [
        len([word for word in sentence.split() if word in pos]) for sentence in sentences]
    neg_words_in_sentences = [
        len([word for word in sentence.split() if word in neg]) for sentence in sentences]
    return pos_words_in_sentences, neg_words_in_sentences


def get_tf_idf_vectors(data, analyzer, ngram_range, token_pattern=None):
    """Get TF IDF vectors for data, an additional flag is set to differentiate between train and test data."""
    if analyzer == 'word':
        token_pattern = '\S+'
    if token_pattern:
        tf_idf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, token_pattern=token_pattern)
    else:
        tf_idf_vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    data_tf_idf = tf_idf_vectorizer.fit_transform(data)
    return data_tf_idf


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser(description='This program trains a model and find the cross validation score as well')
    parser.add_argument('--input', dest='inp', help='Enter the input file path')
    parser.add_argument('--classifier', dest='cls', help='Enter the classification algorithm', choices=['svm', 'rf', 'logit', 'bnb'])
    args = parser.parse_args()
    input_data_frame = pd.read_csv(args.inp, sep='\t')
    pos_words = read_lines_from_file('POS_Words.txt')
    neg_words = read_lines_from_file('NEG_Words.txt')
    texts = input_data_frame["text"]
    sentiments = input_data_frame["label"]
    assert texts.shape[0] == sentiments.shape[0]
    if args.cls == 'svm':
        classifier = LinearSVC(random_state=1)
    elif args.cls == 'rf':
        classifier = RandomForestClassifier(random_state=1)
    elif args.cls == 'logit':
        classifier = LogisticRegression(random_state=1)
    elif args.cls == 'bnb':
        classifier = BernoulliNB(alpha=0.001)
    # combine the best word and char ngram ranges
    word_tf_idf_vectors = get_tf_idf_vectors(texts, 'word', (1, 2))
    char_tf_idf_vectors = get_tf_idf_vectors(texts, 'char', (3, 6))
    texts_as_list = texts.values.tolist()
    # the below code find the counts of positive words and negative words using the senti wordnet
    pos_words_in_sentences, neg_words_in_sentences = find_positive_negative_counts_in_Sentences(texts_as_list, pos_words, neg_words)
    pos_and_neg_sparse_matrix = coo_matrix(list(zip(pos_words_in_sentences, neg_words_in_sentences)))
    # the below code combines TF IDF with positive and negative word counts
    all_tf_idf_vectors_with_senti_wordnet = hstack([word_tf_idf_vectors, char_tf_idf_vectors, pos_and_neg_sparse_matrix])
    # With this implementation you will get better scores than the paper
    cv_scores_precision = cross_val_score(classifier, all_tf_idf_vectors_with_senti_wordnet, sentiments, cv=5, scoring='precision_micro')
    print('Mean Micro Precision-Score=', cv_scores_precision.mean())
    cv_scores_recall = cross_val_score(classifier, all_tf_idf_vectors_with_senti_wordnet, sentiments, cv=5, scoring='recall_micro')
    print('Mean Micro Recall-Score=', cv_scores_recall.mean())
    cv_scores_f1 = cross_val_score(classifier, all_tf_idf_vectors_with_senti_wordnet, sentiments, cv=5, scoring='f1_micro')
    print('Mean Micro F1-Score=', cv_scores_f1.mean())
    cv_scores_accuracy = cross_val_score(classifier, all_tf_idf_vectors_with_senti_wordnet, sentiments, cv=5, scoring='accuracy')
    print('Mean Micro Accuracy-Score=', cv_scores_accuracy.mean())


if __name__ == '__main__':
    main()
    