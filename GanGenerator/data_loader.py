import numpy as np 
import random


class Generator_Data_Loader():
    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        self.length = length
        self.sentences = []
    
    def create_batches(self, data_file):
        self.sentences = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parsed_line = [int(x) for x in line]
                if len(parsed_line == self.length):
                    self.sentences.append(parsed_line)
        
        self.num_batches = int(len(self.sentences) / self.batch_size)
        self.sentences = self.sentences[:self.num_batches * self.batch_size]
        self.sequence_batch = np.split(np.array(self.sentences), self.num_batches, 0)
        self.pointer = 0
    
    def next_batch(self):
        batch = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return batch
    
    def reset_pointer(self):
        self.pointer = 0


class Discriminator_Data_Loader():
    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        self.length = length
        self.sentences = np.array([])
        self.labels = np.array([])
        
    def load_train_data(self, positive_file, negative_file):
        positive_examples = []
        negative_examples = []
        
        with open(positive_file) as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parsed_line = [int(x) for x in line]
                positive_examples.append(parsed_line)
        with open(negative_file) as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parsed_line = [int(x) for x in line]
                if len(parsed_line) == self.length:
                    negative_examples.append(parsed_line)
        positive_file = random.sample(positive_examples, 10000)
        self.sentences = np.array(positive_examples + negative_examples)
        
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)
        
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        
        self.num_batches = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.batch_size * self.num_batches]
        self.labels = self.labels[:self.batch_size * self.num_batches]
        self.sentences_batches = np.split(self.sentences, self.num_batches, 0)
        self.labels_batches = np.split(self.labels, self.num_batches, 0)
        
        self.pointer = 0
        
    def next_batch(self):
        batch = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return batch

    def reset_pointer(self):
        self.pointer = 0
        