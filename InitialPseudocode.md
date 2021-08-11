### Initial Pseudocode
Austin Larson
larsoaj1

## General Overview


## Exploration

# Look at number of samples
To decide what kind of models might be appropriate for this dataset, we should look and see how many samples we have. If there are relatively few, a deep network is likely not a good choice, and an SVM might work much better. However, if there are more, a PyTorch network with a few layers may perform very well.
print(len(df['SentenceId'].value_counts()))

# Look at length of average sample
If the average length of each sentence is relatively small, then including information about every word tokenized is feasible. If it is longer, like 20+ words, this might take too much computational effort, and using mean word vectors would simplify the machine learning task.
lengths = length(df['SentenceId'])
print(lengths.min(), lengths.mean(), lengths.max())

# Look at size of vocabulary
This will help us determine which vocabulary we should load in from spaCy or nltk. If it's relatively small, we should load the small vocabulary to save processing time
df.get_single_words().value_counts()
print(len(df))

# Graphs
These are just visual things so that I can see what the distrbution looks like
displot(sentence_length)
displot(sentiment)

## Preprocessing
I need to get it into two representations: word sentiment statistics and average word vectors.
This will also be done fro test.csv

import spacy
nlp = spacy.load('english')

data = train.csv
for sentence in data
    with nlp.disable_pipes():
        vector_phrase = [token.vector for token in processed sentence]
    vector_data = vector_data.append(average of vector_sentence)
vector_data = vector.merge(data, keep='Sentiment')
vector_data.length = length of phrase in words
save(vector_data as vector_train.csv)

Repeat for test

## Model
train_data, test_data = load_data()

# Linear SVM
import SVM from sklearn
linear_svm = LinearSVM()
linear_svc.fit(train_data)

# Radial SVM
rad_svm = RadialSVM()
rad_svm.fit(train_data)

# PyTorch Neural Network
import torch
train_dataloader = DataLoader(tensor(train_data), batch_size)
torch_model = Sequential(Linear(201, 500), ReLU(), Linear(500,100), ReLU(), Linear(100,5), Softmax())
optimizer = RAdam()
loss_fn = CrossEntropyLoss()

for i in range(epochs):
    for X, y in train_dataloader:
        predict = torch_model(X)
        loss = loss_fn(y, predict)
        zero_grad()
        backpropagate()
        step_optimizer()
    printf("Epoch %d is complete", i+1)

## Evaluation
import metrics from sklearn

def predict_labels(test_dataloader, model):
    predict = []
    for X, y in test_dataloader:
        predict.append(model(X))
    return predict
    
test_dataloader = DataLoader(tensor(test_data))
linear_predict = linear_svm(test_data)
rad_predict = rad_svm(test_data)
net_predict = predict_labels(test_dataloader, torch_model)

metrics = [Fscore(), Accuracy(), Precision()]
for metric in metrics:
    scores.append({'Type': metric, 'Linear': metric(test_data, linear_predict), 'Radial': metric(test_data, rad_predict),\
            'Net': metric(test_data, net_predict)})

print(scores.min(axis=1))