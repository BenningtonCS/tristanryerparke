import pandas as pd 
import numpy as np
import torch
import pickle

def preprocess(filename,outfilename):
    csv = pd.read_csv(filename, sep='\t', header=None, usecols=[1,2])
    wordcount = {}
    wordcount["<pad>"] = 0
    wordcount["<oov>"] = 2
    data = []
    #split sentances and count uses of words
    for index, row in csv.iterrows():
        titleWords = str(row[1]).split()
        for word in titleWords:
            if word in wordcount:
                wordcount[word] += 1
            else:
                wordcount[word] = 1
        data.append((titleWords,str(row[2])))
    
    #remove little used words
    for key in wordcount.copy():
        if wordcount[key] == 1:
            wordcount.pop(key)
    id = 0
    for key in wordcount:
        wordcount[key] = id
        id += 1
    #create vectors for each sentance
    sentVects = []
    labels = []
    vocab = wordcount
    for item in data:
        wordVec = []
        sentance = item[0]
        for i in range(0,8):
        
            if i < len(sentance):
                word = sentance[i]
                #print(word)
                if word in vocab: 
                    wordVec.append(vocab[word])
                else:
                    wordVec.append(vocab['<oov>'])
            else:
                wordVec.append(vocab['<pad>'])
        sentVects.append(wordVec)
        
        if item[1] == 'b':
            labels.append(0)
        elif item[1] == 't':
            labels.append(1)
        elif item[1] == 'e':
            labels.append(2)
        else:
            labels.append(3)
    features = torch.as_tensor(sentVects) 
    labels = torch.as_tensor(labels)

    data = (features, labels, len(vocab))
    torch.save(data,open(outfilename,'wb'))
        
preprocess('data/train.txt','data/trainData.tch')
preprocess('data/valid.txt','data/validData.tch')
preprocess('data/test.txt','data/testData.tch')