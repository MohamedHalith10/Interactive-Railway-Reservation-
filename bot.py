import random
import string 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 
nltk.download('punkt') 
nltk.download('wordnet')
with open('halith.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
lemmer = WordNetLemmatizer()
def Tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def Normalize(text):
    return Tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
gre_inp = ["hello", "hi","hii", "greets", "sup", "what's up","hey"]
gre_out = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def greet(sentence):
    for word in sentence.split():
        if word.lower() in gre_inp:
            return random.choice(gre_out)
def response(user_response):
    chatty_res=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatty_res=chatty_res+"I don't understand you"
        return chatty_res
    else:
        chatty_res = chatty_res+sent_tokens[idx]
        return chatty_res
check=1
print("Chatty: Hii I am Chatty. I will answer your queries about Railway reservation process. If you want to exit, type Bye!")
while(check==1):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            check=0
            print("chatty: You are welcome..")
        else:
            if(greet(user_response)!=None):
                print("chatty: "+greet(user_response))
            else:
                print("chatty: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        check=0
        print("chatty: Bye!")    
        
        

