import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
import warnings
warnings.filterwarnings("ignore")



nltk.download('brown', quiet=True)


nltk.download('stopwords', quiet=True)


def clean_sentences():
    
    sentences = brown.sents()    
    cleaned_sentences = []
    
    for sentence in sentences:
        lower_case_sentence = [word.lower() for word in sentence]         
        cleaned_sentences.append(lower_case_sentence)
    return cleaned_sentences


stop_words = set(stopwords.words('english'))
cleaned_sentences=clean_sentences()
brown_words = [word for sentence in cleaned_sentences for word in sentence if word not in stop_words]

stop_words = set(stopwords.words('english'))
s1 = [sents for sents in brown.sents()]
brown_words1 = []
for sentence in s1:
    #print(sentence)
    words = [word.lower() for word in sentence if word.lower() not in stop_words ]
    brown_words1.append(words)
   
    

#brown_words = [word.lower() for word in brown.words()] #if word.isalpha()]




def check_word_in_corpus():
  
    x = True
    w2=None
    while x:
        
        w1 = input("\nPlease enter a word : ").lower()
        
        if w1 in brown_words:
            print(f"\nThe word '{w1}' is in the corpus and so proceeding further...")
            x = False 
            
        
        else:
            print(f"\nThe word '{w1}' is not in the corpus.")
            
           
            w2 = input("\nChoose an option ( 'a' or 'b' ):\n a. Again - To choose a word\n b. Quit - To Quit\n ").lower()
            if w2 =="again" or w2=="a":
                x=True
                
            elif w2=="quit" or w2=="b":
                print("\nQuitting the program.")
                x=False
            else:
                print("\n Invalid option choosed, Defaulting to a. Again option")
                x=True
                
    
    return w1,w2
                

    

bigrams = []
for words in brown_words1:
    bigram = list(nltk.bigrams(words))
    bigrams.extend(bigram)


    



fdist = FreqDist(brown_words)  
cfd = ConditionalFreqDist(bigrams)  

def predict_next_words(w1):
    sentence = [w1]
    while w1:
        
        if w1 not in cfd:
            print(f"\nNo more likely words to follow '{w1}' , Do you want to enter another word and continue... or Quit.")
            w3=input("\n Choose an option ( 'a' or 'b' ):\n a. Again - To choose a word\n b. Quit - To Quit\n ").lower()
            if (w3=="a" or w3=="again"):
                w1,w2=check_word_in_corpus()
            elif (w3=="b" or w3=="quit"):
                print("\nQuitting....")
                break
            else:
                print("\nInvalid choice, defaulting to the first option")
                w1,w2=check_word_in_corpus()
                
            
        
        top_three = cfd[w1].most_common(3)
        print(f"\n{w1} ...")
        print(f"\nWhich word should follow :'{w1}'")
        
        for i, (word, count) in enumerate(top_three, start=1):
            probability = count / fdist[w1]
            print(f"{i}) {word} P({w1} {word}) = {probability:.5f}")
        print("4) QUIT")

        choice = input("\nSelect an option (1-4): ")
        z=True
       
        if choice == '4' or choice.lower() == 'quit':
            print("Quitting...")
            break
        elif choice in ['1', '2', '3'] and int(choice) <= len(top_three):
            w1 = top_three[int(choice) - 1][0]
            sentence.append(w1)
        else:
            print("\nInvalid choice, defaulting to the first option.")
            w1 = top_three[0][0]
            sentence.append(w1)
                
    
    print("\nConstructed sentence:",' '.join(sentence))

w1,w2 = check_word_in_corpus()

if w2 == "quit" or w2=="b":
    exit(1)
else:
    predict_next_words(w1)
