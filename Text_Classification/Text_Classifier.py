import os
import time
import sys
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import string
import collections

import nltk
import re
import string, nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from decimal import Decimal, getcontext

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.base import (
    TransformerMixin,
    BaseEstimator
)
from sklearn.metrics import confusion_matrix


# nlp related packages
import nltk
import re

from nltk.stem import WordNetLemmatizer


#############################################################################################
#    Unshuffled Train Test Split
#############################################################################################

def Unshuffle_train_test_split(df:pd.DataFrame,
                        train_size: float,
                        target : List[str]) -> pd.DataFrame:
    
    train_size = train_size/100 
    test_size= round((1 - train_size),2)

    X = df.drop(target, axis = 1)
    y = df[target]

    X_train, X_test, y_train, y_test  = train_test_split(X, y,
                                                         train_size= train_size, 
                                                         test_size= test_size,
                                                         shuffle=False)


    return X_train, y_train, X_test, y_test


#############################################################################################
#    Review column Text preprocessing
#############################################################################################

punc = list(string.punctuation)
stop_word = stopwords.words("english")
lemma = WordNetLemmatizer()

def TextPreProcessing(data: str):   
    data = data.lower()  # lower casing
    data = re.sub('[^a-zA-Z]', ' ', data) # taking only alphabhets
    data = [word for word in word_tokenize(data) if (word not in punc) and (word not in stop_word)] #removing stop words 
    text = [lemma.lemmatize(str(word)) for word in data] # lemmatizing to reduce word-forms
    return text  


#############################################################################################
#    Encoding Sentiments as O (negative) and 1 (positive)
#############################################################################################

def labelEncoder(target):
    target_numpy = np.where(np.array(target) == "positive", 1, 0)    
    return target_numpy


#############################################################################################
#    Getting Vocabulary from Training Dataset
#############################################################################################
def getVocab(df: pd.DataFrame, 
                   column: str) -> list:

    vocab = []
    for i in df[column]:
        vocab.extend(i)

    vocab = tuple(
                        sorted(
                                list(
                                    set(vocab)
                                    ) 
                                ) + ["UNK"]
                    )

    return vocab


#############################################################################################
#    Document Word Frequencies
############################################################################################# 
def docWordFreq(df: pd.DataFrame, 
                       column: str) -> Dict:
    wordFreq = {}
    for i, docCounter in zip(df[column], range(0, len(df))):
        wordFreq[str(docCounter)] = collections.Counter(i)

    return wordFreq



#############################################################################################
#    Document Vectorization
############################################################################################# 

def docVector(df, column, vocab: tuple) -> np.array:

    tempd_tokens = df[column].tolist()
    
    # Creating mapping from terms to their indices
    vocab_index = {term: idx for idx, term in enumerate(vocab)}
    
    # Optionally, add an "<UNK>" token to handle unseen words
    if "UNK" not in vocab_index:
        unk_idx = len(vocab_index)  # Assign the next index to "<UNK>"
        vocab_index["UNK"] = unk_idx
    else:
        unk_idx = vocab_index["UNK"]

    # Initialize an empty document-term matrix
    doc_vector = np.zeros((len(tempd_tokens), len(vocab_index)), dtype=int)

    # Populate the document-term matrix with term frequencies
    for doc_idx, doc in enumerate(tempd_tokens):
        for term in doc:
            # Use the index of the term if it exists in the vocab; otherwise, use "<UNK>"
            term_idx = vocab_index.get(term, unk_idx)
            doc_vector[doc_idx, term_idx] += 1

    return doc_vector




#############################################################################################
#    Word - Class calculation
############################################################################################# 

def class_stats(y_train_n, doc_vector, classLabel, train_vocab):
    
    # Getting indices of the specified class label
    class_idx = np.where(y_train_n == classLabel)[0]
    
    # Count the total no.of words in the specified class
    class_wordCount = np.sum(np.where(doc_vector[class_idx, :] > 0, 1, 0))
    
    #  normalized sum of words for each feature based on the count
    # print("Length of vocab:", len(train_vocab))
    word_probs = (np.sum(doc_vector[class_idx, :], axis=0) + 1 )/ (class_wordCount + len(train_vocab))

    return word_probs


#############################################################################################
#    Word - Class probabilities calculation
############################################################################################# 

def feature_attribute(train_vocab, 
                       classLabels,
                       y_train_n,
                       doc_vector):

    term_probs = pd.DataFrame({})
    # Creating an empty arr to save the values
    word_prob_ar = np.empty(shape = (len(train_vocab), len(classLabels)))

    for label in classLabels:
        freq= np.sum(np.where(y_train_n == label, 1, 0))/len(y_train_n)
        word_probs = class_stats(y_train_n=y_train_n,
                                doc_vector= doc_vector,
                                classLabel=label,
                                train_vocab=train_vocab)
        word_prob_ar[:,label] = word_probs
        
        # Creating word probability dataframe
        word_prob_df = pd.DataFrame({"word_probs": word_probs, 
                                    "vocab":train_vocab,
                                    "classes": label,
                                    "freq": freq})
        
        # Append to the master DataFrame
        term_probs = pd.concat([term_probs, word_prob_df], ignore_index=True)

    return term_probs, word_prob_ar



#############################################################################################
#    Log probabilities calculation
############################################################################################# 

def calculate_probs(test_doc_vec, 
                        learned_params, 
                        prior_class_probs,
                        classLabels):
    
    " calculating probability values "
    log_probs_values = np.empty(shape = (len(test_doc_vec), len(classLabels)))

    for i in range(0, len(classLabels)):

        # Multiplying conditional probabilities of class with document vector
        product = test_doc_vec * np.expand_dims(learned_params[:, i], axis=0)

        # calculating logarithm, replacingreplace log(0) or invalid values with 0
        log_values = np.log10(product)
        log_values[np.isneginf(log_values)] = 0  
        
        #print(log_values)
        log_sum_values = np.sum(log_values, axis = 1)
        log_probs_values[:, i] = log_sum_values
    
    log_probs_values = log_probs_values + np.log10(prior_class_probs)
    return log_probs_values



#############################################################################################
#    Document Term Matrix 
############################################################################################# 

class DTM(BaseEstimator, TransformerMixin): 
    def __init__(self, preprocessed_review: str):
        self.name = "Document Term Matrix"
        self.preprocessed_review = preprocessed_review

    def fit(self, X_train, y_train,X_test, y_test = None):
        self.train_vocab = getVocab(X_train,
                                              column = self.preprocessed_review)
        self.doc_wordFrequency = docWordFreq(X_train, 
                    column= self.preprocessed_review)
        self.doc_vector = docVector(X_train,self.preprocessed_review,vocab= self.train_vocab)
        self.y_train_n = y_train
        self.classLabels = np.unique(self.y_train_n)  
 
        # Get Word probabilities
        self.term_probs, self.word_prob_ar = feature_attribute(self.train_vocab, 
                   self.classLabels,
                   self.y_train_n,
                   self.doc_vector)
 
        # learned_parameters
        self.learned_params = self.word_prob_ar
 
        # prior class probabilities
        train_unique_counts = np.unique(self.y_train_n,return_counts=True)[1]
        self.prior_class_probs = train_unique_counts /np.sum(train_unique_counts, axis = 0)
        self.prior_class_probs = self.prior_class_probs[np.newaxis, :]

        return self
 
#############################################################################################
#    Naive Bayes Class
############################################################################################# 
 
class NaiveBayes():

    def __init__(self, dtm_model):

        # Accessing the attributes from the DTM model instance

        self.name = "Naive Bayes Classifier"
        self.dtm_model = dtm_model
        self.preprocessed_review = dtm_model.preprocessed_review
        self.train_vocab=dtm_model.train_vocab
        self.learned_params = dtm_model.learned_params
        self.prior_class_probs = dtm_model.prior_class_probs
        self.classLabels = dtm_model.classLabels
 
    def predict(self, X_test):
        self.test_doc_wordFrequency = docWordFreq(X_test, column= self.preprocessed_review)
        #print(X_test, self.test_doc_wordFrequency)

        # Non-Binary Document Vectors
        self.test_doc_vec = docVector(X_test,self.preprocessed_review,vocab= self.train_vocab)
 
        # calculating test doc vector's probaility scores
        test_probabilities = calculate_probs(test_doc_vec=self.test_doc_vec, 
                                                    learned_params=self.learned_params,
                                                    prior_class_probs=self.prior_class_probs,
                                                    classLabels=self.classLabels)

        return test_probabilities
 
#############################################################################################
#    Model evaluation metrics
#############################################################################################

def print_metrics(y_test, y_test_pred_argmax):
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred_argmax)
    tn, fp, fn, tp = cm.ravel()  # Unpack confusion matrix values

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    # Output metrics
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Negative Predictive Value (NPV): {npv:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-score: {f1_score:.2f}")

    return None

#############################################################################################
#    Main Function starts from here
#############################################################################################

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




# Command-line Argument Handling
if __name__ == "__main__":
    # Default values
    if len(sys.argv) != 3:
        print("\n Given invalid parameters , so defaulting train size to 80 percentage and using default naive bayes classifier")
        algo = 0  # 0: Naive Bayes, 1: Logistic Regression
        train_size = 80  # Default training size (80%)

    # Parse command-line arguments
    if len(sys.argv) == 3:
        try:
            algo = int(sys.argv[1])
            train_size = int(sys.argv[2])
            if not (50 <= train_size <= 90):
                print("\nInvalid train size, it should be between 50-90%, so defaulting train size to 80%")
                train_size = 80
        except ValueError:
            train_size = 80
            
 
    # Load IMDB dataset
    df = pd.read_csv("IMDB Dataset.csv")  
    df=df[:1000]

    # Unshuffled Train-test split
    X_train, y_train, X_test, y_test = Unshuffle_train_test_split(df = df,train_size= train_size,target= ["sentiment"])
    X_train["processed_review"] = X_train["review"].apply(lambda x : TextPreProcessing(str(x)))
    X_test["processed_review"] = X_test["review"].apply(lambda x : TextPreProcessing(str(x)))

    y_train = labelEncoder(target=y_train)
    y_test = labelEncoder(target=y_test)

    
    dtm_model = DTM(preprocessed_review = "processed_review")
    
    # algo == 0 is Naive Bayes Model
    if algo == 0:
        print("\n\nResults of Naive Bayes classifier are :")
        print(f"\nTraining set size: {train_size}%")
        print("\nClassifier type: Naive Bayes")
 
        print("\nTraining classifier…")
        dtm_model.fit(X_train, y_train,X_test, y_test)

        model_path = r'DTM_Finalv1.pkl'
        joblib.dump(dtm_model, model_path)

        print("\nTesting classifier…")
        dtm_model = joblib.load('DTM_Finalv1.pkl')
        naive_bayes_pipeline = NaiveBayes(dtm_model)
        NB_y_test_pred = naive_bayes_pipeline.predict(X_test)
        NB_y_test_pred_argmax = np.argmax(NB_y_test_pred, axis = 1)
        
        
        print("\nTest results / metrics:")
        print_metrics(y_test,NB_y_test_pred_argmax)


    # algo == 1 is Logistic Regression Model
    if algo == 1:
        print("Results of Logistic Regression are \n:")
        print(f"Training set size: {train_size}%")
        print("Classifier type: Logistic Regression")
 
        print("\nTraining classifier…")
        dtm_model.fit(X_train, y_train,X_test, y_test)
        # # Save DTM model
        model_path = r'DTM_Finalv2.pkl'
        joblib.dump(dtm_model, model_path)
        log_reg = LogisticRegression()  # Increase max_iter if convergence issues occur
        log_reg.fit(dtm_model.doc_vector, y_train)

        print("\nTesting classifier…")
        dtm_model = joblib.load('DTM_Finalv2.pkl')
        naive_bayes_pipeline = NaiveBayes(dtm_model)
        NB_y_test_pred = naive_bayes_pipeline.predict(X_test)
        log_y_pred = log_reg.predict(naive_bayes_pipeline.test_doc_vec)
        
        
        # Evaluate Metrics
        print("\nTest results / metrics:")
        print_metrics(y_test,log_y_pred)


    
    X= True
    while X:
        user_input = input("\nEnter your sentence/document : ")
        print(f"\nSentence/document S: {user_input}\n")
        
        user_text = pd.DataFrame({"Index":[0], "Review": [user_input]})
        user_text["processed_review"] = user_text["Review"].apply(lambda x : TextPreProcessing(str(x)))
        # user_pred_argmax = np.argmax(naive_bayes_pipeline.predict(user_text), axis = 1)
        user_pred = naive_bayes_pipeline.predict(user_text)
        #print(user_pred)
        user_test_pred_argmax = np.argmax(user_pred, axis = 1)
        getcontext().prec = 5
        flat_user_pred = user_pred.flatten()
        x_values_very_small = [10 ** Decimal(value) for value in flat_user_pred.tolist()]
        neg = x_values_very_small[0]
        neg = str(neg)

        pos = x_values_very_small[1]
        pos = str(pos)


        
        print(f"\n\nwas classified as {'Positive' if user_test_pred_argmax == 1 else 'Negative'} .")
        print(f"P(Positive | S) = {pos}")
        print(f"P(Negative | S) = {neg}")

        u1 = input("Do you want to enter another sentence [Y/N]?")
        
        if u1.lower()=='y':
            X= True
        else:
            X= False
 

