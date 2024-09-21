# -*- coding: utf-8 -*-
# @Author: Prathyusha Danda, Pruthwik Mishra
# @Date:   2017-08-07 22:33:33
# @Last Modified by:   Prathyusha Danda
# @Last Modified time: 2017-08-16 20:50:04

from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
from sys import argv
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics.scorer import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# import sklearn.metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from pickle import dump
from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
# from random import sample


def readLinesFromFile(inputFile, classFile=0):
    with open(inputFile, 'r') as inputFileDesc:
        linesRead = [line.strip()
                     for line in inputFileDesc.readlines() if line.strip()]
        if not classFile:
            return linesRead
        else:
            return list(map(int, linesRead))


def findNgrams(text, n=1):
    tokens = [token for token in text.split() if token.strip()]
    nGramSet = set([' '.join(tokens[i: i + n])
                    for i in range(0, len(tokens) - n + 1)])
    return list(nGramSet)


def getTfidfForAllData(xData, analyzer='word', ngram_range=(1, 1)):
    '''
    did experiments with ngram with words and ngrams with characters, because the spelling variations in social media text is more
    char n-grams better than word n-grams
    '''
    tfIDFVectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)
#    tfIDFVectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
    # tfIDFVectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
    # tfIDFVectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
#    tfIDFVectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 6))
    # print(tfIDFVectorizer.ngram_range, tfIDFVectorizer.analyzer)
    xDataTfidf = tfIDFVectorizer.fit_transform(xData)
    return xDataTfidf, tfIDFVectorizer


def logisticRegression(xTrain, yTrain, xTest):
    clf = LogisticRegression().fit(xTrain, yTrain)
    predicted = clf.predict(xTest)
    return predicted.tolist(), clf


def KNearestNeighbour(xTrain, yTrain, xTest):
    clf = KNeighborsClassifier(n_neighbors=3).fit(xTrain, yTrain)
    predicted = clf.predict(xTest)
    return predicted.tolist(), clf


def gaussianNB(xTrainTfidf, yTrain, xTestTfidf):
    clf = GaussianNB()
    clf.fit(xTrainTfidf.toarray(), yTrain)
    predicted = clf.predict(xTestTfidf.toarray())
    return predicted.tolist(), clf


def bernoulliNB(xTrainTfidf, yTrain, xTestTfidf):
    clf = BernoulliNB()
    clf.fit(xTrainTfidf.toarray(), yTrain)
    predicted = clf.predict(xTestTfidf.toarray())
    return predicted.tolist(), clf


def randomForestClassifier(xTrainTfidf, yTrain, xTestTfidf):
    clf = RandomForestClassifier()
    clf.fit(xTrainTfidf.toarray(), yTrain)
    predicted = clf.predict(xTestTfidf.toarray())
    return predicted.tolist(), clf


def SVCWithLinearKernel(xTrainTfidf, yTrain, xTestTfidf):
    clf = SVC(kernel='linear', random_state=0).fit(xTrainTfidf, yTrain)
    predicted = clf.predict(xTestTfidf)
    return predicted.tolist(), clf


def gradientBoosting(xTrainTfidf, yTrain, xTestTfidf):
    clf = GradientBoostingClassifier()
    clf.fit(xTrainTfidf.toarray(), yTrain)
    predicted = clf.predict(xTestTfidf.toarray())
    return predicted.tolist(), clf


# def XGBoosting(xTrainTfidf, yTrain, xTestTfidf):
#     clf = XGBClassifier()
#     clf.fit(xTrainTfidf.toarray(), yTrain)
#     predicted = clf.predict(xTestTfidf.toarray())
#     return predicted.tolist(), clf


def createReverseDictionary(dataDictionary):
    return {value: key for key, value in dataDictionary.items()}


def writeListToFile(dataList, filePath):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList) + '\n')


def writeDataToFile(data, filePath):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write(data + '\n')


def findPrecisionRecallF1score(goldLabels, predictedLabels, trueLabels=None):
    return classification_report(goldLabels, predictedLabels, target_names=trueLabels)


def dumpObjectToFile(dataObject, pickleFilePath):
    with open(pickleFilePath, 'wb') as fileDumped:
        dump(dataObject, fileDumped)


def findAccuracyOfClassifier(gold, predicted):
    findMatches = [
        1 if gold[index] == item else 0 for index, item in enumerate(predicted)]
    return sum(findMatches) / len(findMatches)


def createKFoldsAndSplit(kFold, data):
    return kFold.split(data)


def runClassifierOnKFoldsAndCaptureMetrics(kFoldSplit, classifier, data, labels, classes=None, k=3):
    foldIndex = 0
    classes = 2 if not classes else classes
    precisionRecallF1ScoreSupportList = np.zeros((k, 4, len(classes)))
    for trainIndex, testIndex in kFoldSplit:
        trainData, testData = data[trainIndex], data[testIndex]
        trainLabels, testLabels = labels[trainIndex], labels[testIndex]
        classifier.fit(trainData, trainLabels)
        predictedLabels = classifier.predict(testData)
        precisionRecallF1ScoreSupportList[foldIndex] = precision_recall_fscore_support(testLabels, predictedLabels, labels=classes)
        foldIndex += 1
    return precisionRecallF1ScoreSupportList


if __name__ == '__main__':
    allData = readLinesFromFile(argv[1])
    allLabels = np.array(readLinesFromFile(argv[2], 1))
    k = int(argv[3])
    # analyzer = 'word'
    analyzer = 'char'
    # ngram_range = (1, 3)
    ngram_range = (2, 6)
    tfidfTransformedData, tfIDFVectorizer = getTfidfForAllData(
        allData, analyzer, ngram_range)
    # svmClassifier = SVC(kernel='linear', random_state=0)
    # bernoulliNB = BernoulliNB()
    logit = LogisticRegression()
    kFoldSplit = createKFoldsAndSplit(KFold(k), tfidfTransformedData)
    # precisionRecallF1ScoreSupportList = runClassifierOnKFoldsAndCaptureMetrics(kFoldSplit, svmClassifier, tfidfTransformedData, allLabels, [0, 1], k)
    # precisionRecallF1ScoreSupportList = runClassifierOnKFoldsAndCaptureMetrics(kFoldSplit, bernoulliNB, tfidfTransformedData, allLabels, [0, 1], k)
    precisionRecallF1ScoreSupportList = runClassifierOnKFoldsAndCaptureMetrics(kFoldSplit, logit, tfidfTransformedData, allLabels, [0, 1], k)
    # print(svmClassifier.__class__)
    print(logit.__class__)
    print('Precision, Recall, F1-Score--[0 1] then overall')
    print(tfIDFVectorizer.analyzer, tfIDFVectorizer.ngram_range)
    prf = np.mean(precisionRecallF1ScoreSupportList, axis=0)[:-1]
    print(prf)
    print(np.mean(prf, axis=1))
