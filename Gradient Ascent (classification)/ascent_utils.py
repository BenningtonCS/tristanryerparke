import numpy as np

def import_data(path_pos,path_neg):
    textfile_pos = open(path_pos, "r+")
    textfile_neg = open(path_neg, "r+")
    pos_sentances = [sentance.split() for sentance in textfile_pos.readlines()]
    neg_sentances = [sentance.split() for sentance in textfile_neg.readlines()]
    all_data = [(i,1) for i in pos_sentances] + [(i,0) for i in neg_sentances]
    return all_data

def create_vocab(data,min_uses=3):
    word_counter = {}
    for sentance in data:
        for word in sentance[0]:
            if word not in word_counter.keys():
                word_counter[word] = 1
            else:
                word_counter[word] += 1
    vocab = {"<oov>":0}
    counter = 1
    for word in word_counter.keys():
        if word_counter[word] > min_uses:
            vocab[word] = counter
            counter += 1
    return vocab

def to_intWords(data,vocab):
    intWords = []
    for sentance in data:
        intWord = []
        for word in sentance[0]:
            if word in vocab.keys():
                intWord.append(vocab[word])
            else:
                intWord.append(0)
        intWords.append(intWord)
    return intWords

def create_arrays(vocab,intWords,data):
    vectors = []
    for intWord in intWords:
        vector = [0] * (len(vocab))
        for int_id in intWord:
            vector[int_id] += 1
        vectors.append(vector)
    y = np.array([sentance[1] for sentance in data])
    x = np.asarray(vectors)
    return x, y

def sigmoid(u):
        return 1 / (1 + np.exp(u))


            
            




    


    