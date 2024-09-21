from sys import argv
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier
from pickle import dump
from random import sample


def readLinesFromFile(inputFile, classFile=0):
    with open(inputFile, 'r') as inputFileDesc:
        linesRead = [line.strip() for line in inputFileDesc.readlines() if line.strip()]
        if not classFile:
            return linesRead
        else:
            return list(map(int, linesRead))


def getTfidf(train, xData, countVect=None, tfidfTransformer=None):
    '''
    did experiments with ngram with words and ngrams with characters, because the spelling variations in social media text is more
    for bn-en char ngrams perform better than word ngrams
    for hi-en word ngrams perform better but they are more or less equal
    '''
    if train:
        countVect = CountVectorizer(analyzer='char', ngram_range=(2, 6))
#        countVect = CountVectorizer(analyzer='word', ngram_range=(1, 3))
#        countVect = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        tfidfTransformer = TfidfTransformer()
        xDataCounts = countVect.fit_transform(xData)
        xDataTfidf = tfidfTransformer.fit_transform(xDataCounts)
    else:
        xDataCounts = countVect.transform(xData)
        xDataTfidf = tfidfTransformer.transform(xDataCounts)
    if train:
        return xDataTfidf, countVect, tfidfTransformer
    else:
        return xDataTfidf, countVect, tfidfTransformer


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
    findMatches = [1 if gold[index] == item else 0 for index, item in enumerate(predicted)]
    return sum(findMatches) / len(findMatches)


if __name__ == '__main__':
    allData = readLinesFromFile(argv[1])
    allLabels = readLinesFromFile(argv[2], 1)
    sampleSize = 150
    allIndices = range(len(allData))
    testIndices = sample(range(len(allData)), sampleSize)
    trainIndices = set(allIndices) - set(testIndices)
    trainData = [allData[index] for index in trainIndices]
    trainLabels = [allLabels[index] for index in trainIndices]
    xTrainTfidf, countVectTrain, tfidfTransformerTrain = getTfidf(1, trainData)
    dumpObjectToFile(countVectTrain, argv[3])
    dumpObjectToFile(tfidfTransformerTrain, argv[4])
    testData = [allData[index] for index in testIndices]
    testLabels = [allLabels[index] for index in testIndices]
    xTestTfidf, countVectTest, tfidfTransformerTest = getTfidf(0, testData, countVectTrain, tfidfTransformerTrain)
    logitClassifier = LogisticRegression(random_state=1)
    svmClassifier = SVC(kernel='linear', probability=True, random_state=1)
    bernoulliNB = BernoulliNB()
    rfClassifier = RandomForestClassifier(random_state=1)
#    ensembleClassifier = VotingClassifier(estimators=[('lr', logitClassifier), ('svm', svmClassifier), ('rf', rfClassifier)], weights=[0.2, 0.5, 0.3])
    ensembleClassifier = VotingClassifier(estimators=[('rf', rfClassifier), ('svm', svmClassifier), ('bnb', bernoulliNB)])
    ensembleClassifier = ensembleClassifier.fit(xTrainTfidf, trainLabels)
    dumpObjectToFile(ensembleClassifier, argv[5])
    predicted = ensembleClassifier.predict(xTestTfidf)
    convertedSentimentsIntoStrings = [str(item) for item in predicted]
    predictedDataWithSentiments = [testData[index] + '\t' + convertedSentimentsIntoStrings[index] for index, item in enumerate(testData)]
    writeListToFile(predictedDataWithSentiments, argv[6])
    classificationReport = findPrecisionRecallF1score(testLabels, predicted) + '\nMicro F1-Score='
    if len(set(predicted)) == 2:
        classificationReport += str(f1_score(testLabels, predicted, average='binary'))
    else:
        classificationReport += str(f1_score(testLabels, predicted, average='micro'))
    print(classificationReport)
    writeDataToFile(classificationReport, argv[7])
    print('Accuracy of the system', findAccuracyOfClassifier(testLabels, predicted))
