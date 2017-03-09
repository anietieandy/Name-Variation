import os, sys
import nltk
import pickle
import itertools as IT
from collections import defaultdict
path = 'C:/Python34/name_variation_research/bins/'
named_entities=[]
with open('only_named_entities.txt') as n:
    for x in n:
        named_entities.append(x.lower().rstrip())

name_list = list(set(named_entities))
dirs = os.listdir(path)

for file in dirs:
    tweets=[]
    wordcount=defaultdict(int)
    #count= IT.count()
    with open(file) as f:
        for line in f:
            tweets.append(line.lower().rstrip())
        for entity in name_list:
            wordcount[entity]=0
            for x in tweets:
                if entity in x:
                    wordcount[entity] +=1
        f = open('dict_'+file+'_'+'.pkl', 'wb')
                #print(entity)
        pickle.dump(wordcount, f)
                #count = next(count)



    
    
