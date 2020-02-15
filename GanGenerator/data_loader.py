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