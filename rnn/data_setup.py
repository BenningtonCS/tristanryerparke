import pandas as pd

raw_data = pd.read_csv('data/newsCorpora.csv', sep='\t', usecols=[1,3,4], header=None)

data = raw_data[
    ( raw_data[3] == "Reuters" ) | 
    ( raw_data[3] == "Daily Mail" ) |
    ( raw_data[3] == "Huffington Post" ) |
    ( raw_data[3] == "Businessweek") ]

train_val = int(len(data) * 0.8)
valid_val = int(len(data) * 0.1) + train_val
test_val = int(len(data) * 0.1 - 1) + valid_val

data[1] = data[1].str.lower()
print(data.head(5))

train_data = data.iloc[0:train_val-1].sample(frac=1).reset_index(drop=True)
valid_data = data.iloc[train_val:valid_val-1].sample(frac=1).reset_index(drop=True)
test_data = data.iloc[valid_val:len(data)].sample(frac=1).reset_index(drop=True)

print(data.shape)
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

train_data.to_csv('data/train.txt', sep='\t', columns=[1,4])
valid_data.to_csv('data/valid.txt', sep='\t', columns=[1,4])
test_data.to_csv('data/test.txt', sep='\t', columns=[1,4])

