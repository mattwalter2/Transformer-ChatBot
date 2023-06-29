import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, word_embedding_size, number_heads):
        super().__init__()
        # This linear transformation produces the query vectors from the input embeddings. 
        # The query vectors are used to determine how much attention each word should pay to all other words in the sequence.
        self.query = nn.Linear(word_embedding_size, word_embedding_size)

        # This linear transformation produces the key vectors from the input embeddings. 
        # The key vectors are used together with the query vectors to compute attention scores.
        self.key = nn.Linear(word_embedding_size, word_embedding_size)

        # This linear transformation produces the value vectors from the input embeddings.
        # The value vectors represent the actual content of the words and are used to compute the output of the attention mechanism. 
        # Each value vector is weighted by the attention scores to form the final output.
        self.value = nn.Linear(word_embedding_size, word_embedding_size)
    
    def forward(self, input):
        Q = self.query(input)
        K = self.key(input)
        V = self.value(input)

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self):
        super.__init__()
    
    def forward(self,input):
        pass

class AddAndNormalization(nn.Module):
    def __init__(self):
        super.__init__()


dimensionality_of_words = 512
number_heads = 8
dropout_percentage = .1
batch_size = 30
max_sequence_length = 50
feed_forward_neural_network_hidden_layer_amount = 2048
number_of_encoder_layers = 5

class EncoderLayer(nn.Module):
    def __init__(self, dimensionality_of_words, number_heads, dropout_percentage, feed_forward_neural_network_hidden_layer_amount):
        super.__init__()
        self.multihead_attention = SelfAttention(number_heads = number_heads)
        self.add_norm_one = AddAndNormalization()
        self.feed_forward_neural_network = FeedForwardNeuralNetwork()
        self.add_norm_two = AddAndNormalization()

class Encoder(nn.Module):
    def __init__(self, dimensionality_of_words, number_heads, dropout_percentage, batch_size, max_sequence_length, feed_forward_neural_network_hidden_layer_amount, number_of_encoder_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(dimensionality_of_words, number_heads, dropout_percentage, batch_size, max_sequence_length, feed_forward_neural_network_hidden_layer_amount, number_of_encoder_layers) for _ in range(number_of_encoder_layers)])

    def forward(self, input):
        encoder_output = self.layers(input)
        return encoder_output

encoder = Encoder()