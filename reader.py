
from os import listdir
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

porter_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
bad_words = {'aed','oed','eed'} # these words fail in nltk stemmer algorithm
def loadDirTQDM(name,stemming,lower_case):
    # Loads the files in the folder and returns a list of lists of words from
    # the text in each file
    X0 = []
    count = 0
    for f in tqdm(sorted(listdir(name))):
        fullname = name+f
        text = []
        with open(fullname, 'rb') as f:
            for line in f:
                if lower_case:
                    line = line.decode(errors='ignore').lower()
                    text += tokenizer.tokenize(line)
                else:
                    text += tokenizer.tokenize(line.decode(errors='ignore'))
        if stemming:
            for i in range(len(text)):
                if text[i] in bad_words:
                    continue
                text[i] = porter_stemmer.stem(text[i])
        X0.append(text)
        count = count + 1
    return X0

def loadDirNoTQDM(name,stemming,lower_case):
    # Loads the files in the folder and returns a list of lists of words from
    # the text in each file
    X0 = []
    count = 0
    for f in sorted(listdir(name)):
        fullname = name+f
        text = []
        with open(fullname, 'rb') as f:
            for line in f:
                if lower_case:
                    line = line.decode(errors='ignore').lower()
                    text += tokenizer.tokenize(line)
                else:
                    text += tokenizer.tokenize(line.decode(errors='ignore'))
        if stemming:
            for i in range(len(text)):
                if text[i] in bad_words:
                    continue
                text[i] = porter_stemmer.stem(text[i])
        X0.append(text)
        count = count + 1
    return X0


def loadDir(name,stemming,lower_case, use_tqdm=True):
    if use_tqdm:
        return loadDirTQDM(name,stemming,lower_case)
    else:
        return loadDirNoTQDM(name,stemming,lower_case)

def load_dataset(train_dir, dev_dir, stemming, lower_case, use_tqdm=True):
    X0 = loadDir(train_dir + '/ham/',stemming, lower_case, use_tqdm=use_tqdm)
    X1 = loadDir(train_dir + '/spam/',stemming, lower_case, use_tqdm=use_tqdm)
    X = X0 + X1
    Y = len(X0) * [1] + len(X1) * [0]
    Y = np.array(Y)
    X_test0 = loadDir(dev_dir + '/ham/',stemming, lower_case, use_tqdm=use_tqdm)
    X_test1 = loadDir(dev_dir + '/spam/',stemming, lower_case, use_tqdm=use_tqdm)

    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [1] + len(X_test1) * [0]
    Y_test = np.array(Y_test)

    return X,Y,X_test,Y_test

def load_dataset_main(train_dir, dev_dir, stemming, lower_case,use_tqdm=True):
    # X0 = loadDir(train_dir + '/pos/',stemming, lower_case, use_tqdm=use_tqdm)
    # X1 = loadDir(train_dir + '/neg/',stemming, lower_case, use_tqdm=use_tqdm)
    # X = X0 + X1
    # Y = len(X0) * [1] + len(X1) * [0]
    # Y = np.array(Y)
    # X_test0 = loadDir(dev_dir + '/pos/',stemming, lower_case, use_tqdm=use_tqdm)
    # X_test1 = loadDir(dev_dir + '/neg/',stemming, lower_case, use_tqdm=use_tqdm)

    # X_test = X_test0 + X_test1
    # Y_test = len(X_test0) * [1] + len(X_test1) * [0]
    # Y_test = np.array(Y_test)

    # return X,Y,X_test,Y_test
    X0 = loadDir(train_dir + '/ham/',stemming, lower_case, use_tqdm=use_tqdm)
    X1 = loadDir(train_dir + '/spam/',stemming, lower_case, use_tqdm=use_tqdm)
    X = X0 + X1
    Y = len(X0) * [1] + len(X1) * [0]
    Y = np.array(Y)
    X_test0 = loadDir(dev_dir + '/ham/',stemming, lower_case, use_tqdm=use_tqdm)
    X_test1 = loadDir(dev_dir + '/spam/',stemming, lower_case, use_tqdm=use_tqdm)

    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [1] + len(X_test1) * [0]
    Y_test = np.array(Y_test)

    return X,Y,X_test,Y_test
