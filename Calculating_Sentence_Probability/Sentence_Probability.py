import nltk
from nltk.corpus import brown
from nltk import bigrams
from collections import Counter

def clean_sentences(brown):
    
    sentences = brown.sents()    
    cleaned_corpus = []
    
    for sentence in sentences:
        lower_case_sentence = [word.lower() for word in sentence]
        modified_sentence = ['<s>'] + lower_case_sentence + ['</s>']
        #print(modified_sentence)
        cleaned_corpus.append(modified_sentence)
    return cleaned_corpus



cleaned_corpus=clean_sentences(brown)
words = [word for sentence in cleaned_corpus for word in sentence]

word_freq = Counter(words)


all_bigrams = [bg for sentence in cleaned_corpus for bg in bigrams(sentence)]
bigram_freq = Counter(all_bigrams)

 
   
        
bigram_conditional_probs = {}

for bigram, bigram_count in bigram_freq.items():
    if (bigram[0]=='<s>'):
        conditional_prob = 0.25
    elif(bigram[1]=='</s>'):
        conditional_prob = 0.25
    else:
        conditional_prob = bigram_count / word_freq[bigram[0]]  
        
    bigram_conditional_probs[bigram] = conditional_prob
    
    
                                                                                                                       

                                                                                                                           

S = input("\nEnter Sentence: ")
S = S.lower()
sentence = '<s> '+S+' </s>'

print(f"\n\nEntered Sentence is : \n\n '{S}'\n")
words = sentence.split()
sentence_bigrams = list(bigrams(words))
print("\n Possible bigrams for above entered sentence are : \n",sentence_bigrams,"\n")


sentence_probability = 1.0

missing_bigrams=False

for bigram in sentence_bigrams:
    if bigram in bigram_conditional_probs or bigram[0]=='<s>' or bigram[1]=='</s>':
        if (bigram[0]=='<s>' or bigram[1]=='</s>'):
            sentence_probability *= 0.25
            print(f"probability of {bigram}:",0.25)
            
        else:
            sentence_probability *= bigram_conditional_probs[bigram]
            print(f"probability of {bigram}:",bigram_conditional_probs[bigram])
            
    else:
        print(f"probability of {bigram}:",0.0)
        sentence_probability *= 0
        missing_bigrams=True

if missing_bigrams:
    print(f"\nThe final probability of the sentence '{sentence}' is {sentence_probability}, as it contains bigram with zero probability.")
else:
    print(f"\nThe final probability of the sentence '{sentence}' is approximately {sentence_probability}")
