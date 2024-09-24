"""Create vectorizers and train models, and perform 5-Fold cross validation"""
from argparse import ArgumentParser
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score


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
    parser.add_argument('--analyzer', dest='ana', help='Enter the analyzer', choices=['word', 'char'])
    parser.add_argument('--ngram', dest='ngr', help='Enter the ngram range', nargs='+', type=int)
    parser.add_argument('--classifier', dest='cls', help='Enter the classification algorithm', choices=['svm', 'rf', 'logit', 'bnb'])
    args = parser.parse_args()
    ngram_range = tuple(args.ngr)
    input_data_frame = pd.read_csv(args.inp, sep='\t')
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
    all_tf_idf_vectors = get_tf_idf_vectors(texts, args.ana, ngram_range)
    # With this implementation you will get better scores than the paper
    cv_scores_precision = cross_val_score(classifier, all_tf_idf_vectors, sentiments, cv=5, scoring='precision_micro')
    print('Mean Micro Precision-Score=', cv_scores_precision.mean())
    cv_scores_recall = cross_val_score(classifier, all_tf_idf_vectors, sentiments, cv=5, scoring='recall_micro')
    print('Mean Micro Recall-Score=', cv_scores_recall.mean())
    cv_scores_f1 = cross_val_score(classifier, all_tf_idf_vectors, sentiments, cv=5, scoring='f1_micro')
    print('Mean Micro F1-Score=', cv_scores_f1.mean())
    cv_scores_accuracy = cross_val_score(classifier, all_tf_idf_vectors, sentiments, cv=5, scoring='accuracy')
    print('Mean Micro Accuracy-Score=', cv_scores_accuracy.mean())


if __name__ == '__main__':
    main()
    