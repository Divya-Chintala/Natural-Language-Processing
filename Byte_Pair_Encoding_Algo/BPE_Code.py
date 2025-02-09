import pandas as pd
import numpy as np
from collections import Counter
import re
import string
import sys
import time
import os
import codecs


def clean_sentence(sentence): # new
    sentence = codecs.decode(sentence,"unicode-escape")
    sentence = re.sub(r'[^A-Za-z0-9\s]', '', sentence)
    #sentence = re.sub(r'\\x[0-9A-Fa-f]{2}', '', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\d', '', sentence)
    
    #remove all punctuation if any
    sentence = ''.join([ch for ch in sentence if ch not in string.punctuation])

    #remove all non-printable
    sentence = ''.join([ch for ch in sentence if ch in string.printable or ' '])

    #other characters that are NOT in the INITIAL vocabulary V
    sentence = ''.join([ch for ch in sentence if ch in string.ascii_letters or ' '])
    # print(sentence)
    return sentence


# def clean_sentence(sentence): # old
#     sentence = re.sub(r'\\x[0-9A-Fa-f]{2}', '', sentence)
#     sentence = re.sub(r'[^\w\s]', '', sentence)
#     sentence = ''.join([ch for ch in sentence if ch in string.printable])    
#     return sentence

def read_files(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    sentences = [clean_sentence(sentence.strip()) for sentence in sentences]
    return sentences

def read_data(sentences,stop_token=None):
    words = []
    corpus = []
    for each_sent in sentences:
        words.append(each_sent.split())
    if stop_token:    
        words = [j + stop_token for i in words for j in i]
    # words = [j for i in words for j in i]
    for i in words:
        string1 = []
        for j in i:
            string1.append(j)
        corpus.append(string1)

    #unq_v = [j.lower() for i in sentences for j in i if j not in [' ']]
    #unq_v = list(set(unq_v))
    #unq_v.insert(0,'_')
    # print('--------------------- \n')
    # print(unq_v)

    v = list(string.ascii_uppercase)  # Uppercase Alphabets
    lowercase_alphabet = list(string.ascii_lowercase)  # Lowercase Alphabets
    v.extend(lowercase_alphabet)
    v.insert(52,'_')
    return corpus, v#, unq_v

def replace(data, join_str): #'er'
    new_corpus = []
    for lst in data: #'l o w e r _'
        # print(lst)
        new_lst = []
        i = 0
        while i < len(lst):
            if ''.join(lst[i:i + 2]) == join_str:
                # print(join_str,"-------------")
                # print(lst,"#############")
                new_lst.append(join_str)
                i += 2  # Skip the next characters since they are merged
            else:
                new_lst.append(lst[i])
                i += 1
        new_corpus.append(new_lst)
    return new_corpus

def BPE(c, k):
    modified_corpus, v = read_data(c, stop_token='_')
    
    
    for _ in range(k):
        bigrams = []
        all_bigrams = []
        for i in modified_corpus:
            for j in range(len(i)):
                k = j + 1
                if i[j] == '_':
                    break
                try:
                    all_bigrams.append(i[j] + i[k])
                    if i[j] + i[k] not in bigrams:
                        bigrams.append(i[j] + i[k])
                except:
                    continue
        
        try:
            new_voc = most_freq_pairs(bigrams, all_bigrams, _)
        except:
            print(f"K merges not possible. Stopped after {_} merges")
            break     
        v.append(new_voc)
        modified_corpus = replace(modified_corpus, new_voc)
        new_v = v+[char for char in v if len(char) > 1]



    return v, new_v

# def most_freq_pairs(bigrams, all_bigrams):
#     freq = [] 
#     for i in bigrams:
#         c = 0
#         for j in all_bigrams:
#             if i == j:
#                 c += 1
#         freq.append(c)
#     final_bigrams = list(zip(bigrams, freq))
#     max_freq = max(freq for bigram, freq in final_bigrams)
#     high_freq_bigrams = [bigram for bigram, f in final_bigrams if f == max_freq]
#     return high_freq_bigrams[0]

def most_freq_pairs(bigrams, all_bigrams, counter):
    bigram_counts = Counter(all_bigrams)
    most_common_bigram = bigram_counts.most_common(1)[0][0] # Get the most frequent bigram
    return most_common_bigram

if __name__ == '__main__':
    numberOfArgumentsPassedFromCommandLine = len(sys.argv) - 1
    
    # Check if exactly 3 arguments (K, Train_file, Test_file) are passed
    if numberOfArgumentsPassedFromCommandLine != 3:
        print("Error: Incorrect number of arguments provided. Expected 3 arguments (K, Train_file, Test_file).")
        sys.exit(1)
    
    try:
        # Arguments passed
        k = int(sys.argv[1])
    except ValueError:
        print("Error: K should be positive integer. Assuming K=5.")
        k = 5  # Default to 5 if an invalid value is provided
    
    try:
        if k < 1:
            print("Error: K should be positive integer. Assuming K=5.")
            raise ValueError
    except ValueError:
        k = 5
    finally:    
        Train_file = sys.argv[2]
        Test_file = sys.argv[3]
        
        # Check if files exist
        if not os.path.exists(Train_file):
            print(f"Error: Training file '{Train_file}' does not exist in this folder.")
            sys.exit(1)
        
        if not os.path.exists(Test_file):
            print(f"Error: Test file '{Test_file}' does not exist in this folder.")
            sys.exit(1)
        
        print("\n BPE Tokenizer":)
        print("Number of merges:", k)
        print("Training file name:", Train_file)
        print("Test file name:", Test_file)

        Train_sentences = read_files(Train_file)
        Test_sentences = read_files(Test_file)
        
        start_time = time.time()
        new_vocab = BPE(Train_sentences, k)[0]
        end_time = time.time()
        print("\nTraining time:", end_time - start_time, "seconds")
        
        # Save the new vocab to a file
        vocab_file = "VOCAB.txt"
        with open(vocab_file, 'w') as f:
            for vocab in new_vocab:
                f.write(f"{vocab}\n")
        # print(f"New vocabulary saved to {vocab_file}")
        
        start_time = time.time()
        test_corpus = read_data(Test_sentences,'_')[0]
        # new_vocab = [word.replace('_', '') for word in new_vocab if word != '_']
        for i in new_vocab:
            test_corpus = replace(test_corpus, i)
        print("Tokenization time:", time.time() - start_time, "seconds")
        print("\nTokenization result:\n")
        ############ New tokens separated ####################
        cleaned_text_corpus=[','.join(sentence) for sentence in test_corpus]
        cleaned_text_corpus = [word.rstrip('_,') for word in cleaned_text_corpus if word != '_']
        # print(cleaned_text_corpus)
        print(','.join(''.join(i) for i in cleaned_text_corpus[:20]))
        # print(','.join(''.join(i) for i in cleaned_text_corpus[:20]))
        if len(cleaned_text_corpus) > 20:
            print('\nTokenized text is longer than 20 tokens....')
        result_file = "RESULT.txt"
        with open(result_file, 'w') as f:
            f.write(','.join(''.join(i) for i in cleaned_text_corpus))
        

        # for i,sentence in enumerate(cleaned_text_corpus):
        #     if i<20:
        #         print(''.join(i) for i in cleaned_text_corpus)
        #     else:
        #         # print(','.join(''.join(i) for i in cleaned_text_corpus))
        #         print('Tokenized text is longer than 20 tokens....')
        #         break
        # vocab_file = "result.txt"
        # with open(vocab_file, 'w') as f:
        #     for tokens in test_corpus:
        #         # f.write(','.join(''.join(i) for i in tokens))
        #         f.write(f"{' '.join(tokens)}\t")
        # print(f"Test tokens saved to {vocab_file}")