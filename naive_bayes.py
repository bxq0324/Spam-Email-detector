
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    """
    #print(len(X),'X')
    n=len(X)
    pos_vocab = Counter()
    neg_vocab = Counter()
    ##TODO:
    for i in range(n):
        counter=Counter(X[i])
        if y[i]==1:
            for j in counter:
                pos_vocab[j]+=counter[j]
        elif y[i]==0:
            for j in counter:
                neg_vocab[j]+=counter[j]
    #print(pos_vocab)
    #print(len(pos_vocab))
    #raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    """
    #print(len(X),'X')
    n=len(X)
    pos_vocab = Counter()
    neg_vocab = Counter()
    ##TODO:
    for i in range (n):
        X_bi=[]
        # create a new bigram list of X
        for j in range(len(X[i])-1):   
            X_bi.append(X[i][j]+" "+X[i][j+1])

        counter_bi=Counter(X_bi)
        counter=Counter(X[i])
        if y[i]==1:
            for k in counter_bi:
                pos_vocab[k]=pos_vocab[k]+counter_bi[k]
            for s in counter:
                pos_vocab[s]=pos_vocab[s]+counter[s]
        elif y[i]==0:
            for k in counter_bi:
                neg_vocab[k]=neg_vocab[k]+counter_bi[k]
            for s in counter:
                neg_vocab[s]=neg_vocab[s]+counter[s]
    #raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    #1.training phase
    prob_pos={}
    prob_neg={}
 
    pos_vocab,neg_vocab=create_word_maps_uni(train_set, train_labels, max_size=None)

    distinct_pos=len(pos_vocab)
    distinct_neg=len(neg_vocab)
    N_pos=0
    N_neg=0

    for i in range(len(train_set)):
        if train_labels[i]==1:
            N_pos+=len(train_set[i])
        elif train_labels[i]==0:
            N_neg+=len(train_set[i])
    
    for i in pos_vocab:
        prob_pos[i]=(pos_vocab[i]+laplace)/(N_pos+laplace*(1+distinct_pos))
    for i in neg_vocab:
        prob_neg[i]=(neg_vocab[i]+laplace)/(N_neg+laplace*(1+distinct_neg))
      
    
    #2.development phase
    L=[]
    for i in range(len(dev_set)):
        p_pos=np.log(pos_prior)
        p_neg=np.log(1-pos_prior)
        for j in dev_set[i]:
            if j in pos_vocab:
                p_pos+=np.log(prob_pos[j])
            else:
                p_pos+=np.log(laplace/(N_pos+laplace*(1+distinct_pos)))
            if j in neg_vocab:
                p_neg+=np.log(prob_neg[j])
            else:
                p_neg+=np.log(laplace/(N_neg+laplace*(1+distinct_neg)))
        if p_pos>=p_neg:
            L.append(1)
        else:
            L.append(0)
    
       
    #raise RuntimeError("Replace this line with your code!")

    return L


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.
    
    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    #1.training phase
    #traing unigram model
    prob_pos={}
    prob_neg={}

    pos_vocab,neg_vocab=create_word_maps_uni(train_set, train_labels, max_size=None)

    distinct_pos=len(pos_vocab)
    distinct_neg=len(neg_vocab)

    N_pos=0
    N_neg=0
    for i in range(len(train_set)):
        if train_labels[i]==1:
            N_pos+=len(train_set[i])
        elif train_labels[i]==0:
            N_neg+=len(train_set[i])
    
    for i in pos_vocab:
        prob_pos[i]=(pos_vocab[i]+unigram_laplace)/(N_pos+unigram_laplace*(1+distinct_pos))
       
    for i in neg_vocab:
            prob_neg[i]=(neg_vocab[i]+unigram_laplace)/(N_neg+unigram_laplace*(1+distinct_neg))
        
    
    #training bigram model
    prob_pos_bi={}
    prob_neg_bi={}

    pos_vocab_bi,neg_vocab_bi=create_word_maps_bi(train_set, train_labels, max_size=None)
    
    distinct_pos_bi=len(pos_vocab_bi)
    distinct_neg_bi=len(neg_vocab_bi)

    N_pos_bi=0
    N_neg_bi=0
    for i in range(len(train_set)):
        if train_labels[i]==1:
            N_pos_bi+=2*len(train_set[i])-1
        elif train_labels[i]==0:
            N_neg_bi+=2*len(train_set[i])-1

    for i in pos_vocab_bi:
        prob_pos_bi[i]=(pos_vocab_bi[i]+bigram_laplace)/(N_pos_bi+bigram_laplace*(1+distinct_pos_bi))
      
    for i in neg_vocab_bi:
            prob_neg_bi[i]=(neg_vocab_bi[i]+bigram_laplace)/(N_neg_bi+bigram_laplace*(1+distinct_neg_bi))
 
    
    #2.development phase
    L_ub=[]

    pos_ub=[]   #a dictionary holding the probability of spam given an email using unigram and bigram model
    neg_ub=[]    #a dictionary holding the probability of ham given an email using unigram and bigram model

    #development for unigram model
    for i in range(len(dev_set)):
        p_pos=np.log(pos_prior)
        p_neg=np.log(1-pos_prior)
        for j in dev_set[i]:
            if j in pos_vocab:
                p_pos+=np.log(prob_pos[j])
            else:
                p_pos+=np.log(unigram_laplace/(N_pos+unigram_laplace*(1+distinct_pos)))
            if j in neg_vocab:
                p_neg+=np.log(prob_neg[j])
            else:
                p_neg+=np.log(unigram_laplace/(N_neg+unigram_laplace*(1+distinct_neg)))
        pos_ub.append((1-bigram_lambda)*p_pos)
        neg_ub.append((1-bigram_lambda)*p_neg)
   
    #development for bigram model
    for i in range(len(dev_set)):
        p_pos_bi=np.log(pos_prior)
        p_neg_bi=np.log(1-pos_prior)
        dev_bi=[]
        for k in range(len(dev_set[i])-1):
            dev_bi.append(dev_set[i][k]+" "+dev_set[i][k+1])
        for j in dev_bi:
            if j in pos_vocab_bi:
                p_pos_bi+=np.log(prob_pos_bi[j])
            else:
                p_pos_bi+=np.log(bigram_laplace/(N_pos_bi+bigram_laplace*(1+distinct_pos_bi)))
            if j in neg_vocab_bi:
                p_neg_bi+=np.log(prob_neg_bi[j])
            else:
                p_neg_bi+=np.log(bigram_laplace/(N_neg_bi+bigram_laplace*(1+distinct_neg_bi)))
        pos_ub[i]+=(bigram_lambda*p_pos_bi)
        neg_ub[i]+=(bigram_lambda*p_neg_bi)

    
    for i in range(len(dev_set)):
        if pos_ub[i]>=neg_ub[i]:
            L_ub.append(1)
        else:
            L_ub.append(0)
 

    return L_ub
