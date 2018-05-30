
# coding: utf-8

# In[ ]:

# modify_accordingly

#should contain "train_questions.txt" and "test_questions.txt and "train_labels.csv"
data_dir  = "./data/" 

#produces test_labels.csv and best_model.pkl
save_dir = "./output/"

#dir containg files of synpnyms
synonym_dir = "./resources/synonyms/"


# In[ ]:

print "loading necessary resources and packages. Make sure you have them"
import os,sys
import cPickle as pickle

import numpy as np
import re

import spacy
nlp = spacy.load('en')

from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


# In[ ]:

class_labesl = {0 : "Abbreviation",
                1 : "Human",
                2 : "Location",
                3 : "Description",
                4 : "Entity",
                5 : "Numeric"
               }


# In[ ]:

#make sure all these files are present!

syn_list =  ['money', 'place', 'comp', 'word', 'letter', 'city', 'an', 'code', 'unit', 'def', 'title',
 'symbol', 'singleBe', 'popu', 'food', 'country', 'InOn', 'How', 'tech', 'speak', 'At',
 'last', 'state', 'presentBe', 'On', 'lang', 'pastBe', 'sport', 'quot', 'ord', 'desc',
 'speed', 'cause', 'eff', 'body', 'be', 'num', 'event', 'peop', 'plant', 'weight',
 'In', 'animal', 'art', 'accompany', 'mount', 'culture', 'substance', 'abbreviation',
 'color', 'prof', 'currency', 'date', 'other', 'fast', 'instrument', 'prod', 'temp',
 'religion', 'loca', 'dist', 'dimen', 'perc', 'Why', 'big', 'job', 'name', 'group', 'dise',
 'univ', 'vessel', 'do', 'Who', 'What', 'time', 'Where', 'stand', 'term']


# In[ ]:

synonyms = {}
for l in syn_list:
    with open(synonym_dir+l) as f:
        for word in f.read().split():            
            synonyms[stemmer.stem(word)] = l.upper()

syn_dict = {"PROF" : wn.synsets("person",["n"])[0],
            "LOCA" : wn.synsets("place",["n"])[0],
            "SUBSTANCE":wn.synsets("substance",["n"])[0],
            "NUMBER":wn.synsets("quantity",["n"])[0]
           }


# In[ ]:

#diffeerent feature dictionaries

#train_data
data_dict = {} #text
data_tok_dict = {} #tokenised_text
data_ne_dict = {} #named_entity_tagged
data_pos_dict = {} #pos
data_tag_dict = {} #pos_rag
data_nv_dict = {} #noun_vector

#test_data
test_dict = {"data":[],"tok":[],"ne":[],"pos":[],"tag":[],"nv":[], }

#test_id for output
test_id = []


# In[ ]:

#load_train_data

data_key ={}

with open("./data/train_questions.txt") as f:
    f.readline();
    print "Loading... train_data"
    for l in f.readlines():
        key, ques = l.split(",")
        data_key[int(key.strip())] = ques.strip()
        
with open(data_dir+"train_labels.csv") as f:
    f.readline();
    print "Loading... train_data"
    for l in f.readlines():
        key, label = l.split(",")
        data_dict.setdefault(int(label.strip()),[]).append(data_key[int(key.strip())])
        
del data_key       

#load_test_data
with open(data_dir+"test_questions.txt") as f:
    f.readline();
    print "Loading... test_data"
    for l in f.readlines():
        key,ques = l.split(",")
        test_id.append(int(key.strip()))
        test_dict["data"].append(ques.strip())
print "LOADED"


# In[ ]:

def parse_data(sentence):
    tok = []
    ne = []
    pos = []
    tag = []
    nv = []
    
    ne_list = {}
    
    sentence = unicode(sentence,"utf-8")
    doc = nlp(sentence)
    
    for n in doc.ents:
         ne_list[n.start] =  (n.end,n.label_)
            
    for n in doc:
        tok.append(n.text)
        pos.append(n.pos_)
        tag.append(n.tag_)
    
    cur_stop = 0
    for i,t in enumerate(tok):
        if i < cur_stop:
            continue
        if i in ne_list:
            cur_stop, label = ne_list[i]
            ne.append(label)
            
            if label in ["LOC","GPE","FACILITY"]:
                nv.append("loca")
            elif label == "NPOR":
                nv.append("lan")
            elif label == "PERSON":
                nv.append("prof")
            else:
                if stemmer.stem(label) in synonyms:
                    nv.append(synonyms[stemmer.stem(label)])
                else:
                    nv.append(stemmer.stem(label))
            
        else:
            ne.append(t)
            
            if i==0:
                nv.append(t)
                continue
            if stemmer.stem(t) in synonyms:
                nv.append(synonyms[stemmer.stem(t)])
                continue
            if pos[i] =="NOUN":
                try:
                    syn = wn.synsets(n.lemma_,["n"])[0]
                    min_dist = float("inf")
                    label = None
                    for key,val in syn_dict.items():
                        sim = syn.shortest_path_distance(val)
                        if  sim and sim < min_dist:
                            min_dist = sim
                            label = key
                    if min_dist<=6:
                        nv.append(stemmer.stem(label))
                        continue
                except IndexError:
                    pass
                if stemmer.stem(t) in synonyms:
                    nv.append(synonyms[stemmer.stem(t)])
                else:
                    nv.append(stemmer.stem(t))
    
    assert len(tok) >= len(ne)
    return tok , ne , pos ,tag ,nv


# In[ ]:

print "Extracting Features..."
for l,ques in data_dict.items():
    for q in ques:
        
        tok,ne,pos,tag,nv = parse_data(q)
        
        data_tok_dict.setdefault(l,[]).append(tok)
        data_ne_dict.setdefault(l,[]).append(ne)
        data_pos_dict.setdefault(l,[]).append(pos)
        data_tag_dict.setdefault(l,[]).append(tag)
        data_nv_dict.setdefault(l,[]).append(nv)
        
for q in test_dict["data"]:
    
    tok,ne,pos,tag,nv = parse_data(q)

    test_dict["tok"].append(tok)
    test_dict["ne"].append(ne)
    test_dict["pos"].append(pos)
    test_dict["tag"].append(tag)
    test_dict["nv"].append(nv)


# In[ ]:

test_size = len(test_dict["data"])
test_y = dict(zip(range(test_size), [None]*test_size))
cnt = 0
print "Running Rule Based Classifier..."


# In[ ]:

#abbrev rules
for idx,label in test_y.items():
    if label is not None:
        continue
    d = test_dict["data"][idx]
    regex1 = r"\b[a-zA-Z&\.]{1,}[A-Z]\b\.?"
    regex2 = r"\b[A-Z\.]{2,}[a-z]\b\.?"
    regex = r"What is [a-zA-Z&\.]*[A-Z]\b\.?"
    r = re.findall(regex1,d) +re.findall(regex2,d)
    if (len(r) and ("mean" in d.lower() or re.match(regex,d))) or "stand for" in d.lower() or "abbreviation" in d.lower() or "acronym" in d.lower():
        cnt+=1
        test_y[idx] = 0
        continue

### who whom whose rules
    ques = test_dict["tok"][idx]
    if ques[0] in ["Whom","Whose","Who"]:
        cnt+=1
        test_y[idx] = 1
        continue
    if ques[0] in ["Why","If"]:
        cnt+=1
        test_y[idx] = 3
        continue
        


### location rules

    ques = test_dict["tok"][idx]
    if ques[0]=="Where":
        cnt+=1
        test_y[idx] = 2
        continue
    if ques[0]=="Is" and "capital" in ques and "GPE" in test_dict["ne"][idx]:
        cnt+=1
        test_y[idx] = 2
        continue
    if ques[0] == "Which" and "place" in ques:
        cnt+=1
        test_y[idx] = 2
        continue


### how define describe explain rules

    ques = test_dict["tok"][idx]
    if ques[0] in ["Define","Describe","Explain","Suggest"] or (ques[0]=="Give" and ques[1]=="reason"):
        cnt+=1
        test_y[idx] = 3
        continue
    if ques[0]=="How":
        if test_dict["tag"][idx][1] in ["JJ","RB"]:
            cnt+=1
            test_y[idx] = 5
            continue
        cnt+=1
        test_y[idx] = 3
        continue
already = [a for a,b in test_y.items() if b is not None]
assert cnt == len(already)
print cnt,"questions classified using Rule Based Classifier"


# In[ ]:

def get_svm_features():
    train_x = []
    train_y = []
    
    for i in range(1,6):
        train_x+=[" ".join(a+b) for a,b in zip(data_nv_dict[i], data_tok_dict[i])]
        train_y+=[i]*len(data_dict[i])
        
    assert len(train_x)==len(train_y)

    test_x = [" ".join(a+b) for a,b in zip(test_dict["nv"], test_dict["tok"])]    
    
    return train_x, train_y, test_x


# In[ ]:

print "preparing data for SVM"
train_x, train_y, test_x = get_svm_features()


# In[ ]:

vectorizer = CountVectorizer()
vectorizer.fit(np.hstack((train_x,test_x)));

train_matrix = np.array(vectorizer.transform(train_x).todense())
test_matrix =  np.array(vectorizer.transform(test_x).todense())


# In[ ]:

print "Running SVM"
lin_clf = LinearSVC(C=1.0)
y_pred_linsvm = lin_clf.fit(train_matrix, train_y).predict(train_matrix)


# In[ ]:

train_acc = np.mean(y_pred_linsvm==train_y)


# In[ ]:

print "SVM finihsed with", train_acc,"accuracy on train_set"


# In[ ]:

print "Predicting on test set..."
test_preds = lin_clf.predict(test_matrix)

for idx,label in test_y.items():
    if  idx in already:
        test_preds[idx] = test_y[idx]
        continue
    test_preds[idx] = test_preds[idx]


# In[ ]:

print "writing predictions and model to the file"
with open(save_dir+"test_labels.csv","w") as f:
    f.write("Id,Prediction\n")
    for i,l in zip(test_id,test_preds):
        f.write(str(i)+","+str(l)+"\n")
        
with open(save_dir+"model.pkl","w") as f:
    pickle.dump(lin_clf,f)

print "FINISHED"


# In[ ]:

#code to estimate c
# def estimate_c():
#     cval = []
#     for exp in range(0,5,1):
#         cval.append(2**exp)
        
#     best_score = 0
#     best_c = None
    
#     for c in cval:
#         lin_clf = LinearSVC(C=c)
#         scores = cross_val_score(lin_clf, train_matrix, train_y, cv=10)
#         score = np.mean(scores)
#         if score > best_score:
#             best_score = score
#             best_c = c
#         print score, c
#     return best_c

