from sys import argv
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from pickle import dump
from pickle import load


def readLinesFromFile(inputFile, classFile=0):
    with open(inputFile, 'r') as inputFileDesc:
        linesRead = [line.strip().lower() for line in inputFileDesc.readlines() if line.strip()]
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


def SVCWithLinearKernel(xTrainTfidf, yTrain):
    clf = SVC(kernel='linear', random_state=0).fit(xTrainTfidf, yTrain)
    return clf


def loadObjectFromFile(pickleFilePath):
    with open(pickleFilePath, 'rb') as fileLoaded:
        return load(fileLoaded)


if __name__ == '__main__':
    trainData = readLinesFromFile(argv[1])
    trainLabels = readLinesFromFile(argv[2], 1)
    trainIndices = range(len(trainData))
    trainData = list(map(lambda x: ' '.join([token.strip() for token in x.split() if token.strip()]), [trainData[index] for index in trainIndices]))
    xTrainTfidf, countVectTrain, tfidfTransformerTrain = getTfidf(1, trainData)
    dumpObjectToFile(countVectTrain, argv[3])
    dumpObjectToFile(tfidfTransformerTrain, argv[4])
    svmClassifier = SVCWithLinearKernel(xTrainTfidf, trainLabels)
    dumpObjectToFile(svmClassifier, argv[5])
