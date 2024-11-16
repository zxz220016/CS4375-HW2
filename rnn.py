import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embed_matrix, dtype = torch.float32), freeze = True)
        self.rnn = nn.LSTM(input_dim, h, num_layers=self.numOfLayer, batch_first=True)
        self.W1 = nn.Linear(h, 32)
        self.activation = nn.ReLU()
        self.W2 = nn.Linear(32, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, (hidden,_) = self.rnn(embedded)

        # [to fill] obtain output layer representations

        output_layer = self.activation(self.W1(hidden[-1]))
        output_layer = self.W2(output_layer)
        
        # [to fill] sum over output 
        sum_output = torch.sum(output_layer, dim = 0)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(sum_output.reshape(1, -1))
        return predicted_vector


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []

    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    print("the size of training data is {}.".format(len(tra)))
    print("the size of validation data is {}.".format(len(val)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    word_embedding = pickle.load(open('Data_Embedding/word_embedding.pkl', 'rb'))
    ############################## ADD
    embedd_index = {word: index for index, word in enumerate(word_embedding.keys())}
    embed_dimension = len(next(iter(word_embedding.values())))
    embed_matrix = np.zeros((len(embedd_index), embed_dimension))

    for word, index in embedd_index.items():
        embed_matrix[index] = word_embedding[word]
    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(embed_dimension, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    stopping_condition = False
    epoch = 0

    patient = 3
    epoch_compare = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while epoch < args.epochs and not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                input_indices = [embedd_index.get(word.lower(), embedd_index['unk']) for word in input_words]

                # Transform the input into required shape
                
                input_tensor = torch.tensor(input_indices, dtype = torch.long).view(-1, 1)
                output = model(input_tensor)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print("The avger loss is {}".format(loss_total/loss_count))
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Output Log-P: {}".format(output.tolist()))
        print("predicated label for this epoch: {}".format(predicted_label + 1))
        trainning_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("----------Validation started------------")
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        with torch.no_grad():
            for input_words, gold_label in valid_data:
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                input_indices = [embedd_index[i.lower()] if i.lower() in embedd_index.keys() else embedd_index['unk'] for i in input_words]

                input_tensor = torch.tensor(input_indices, dtype = torch.long).view(-1, 1)
                output = model(input_tensor)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Output Log-P: {}".format(output.tolist()))
            print("predicated label for this epoch: {}".format(predicted_label + 1))
            validation_accuracy = correct/total
        
        
        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1
