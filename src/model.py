"""
Module: model.py
Models:
    - EncoderLSTM
    - DecoderLSTM
    - TrajectoryModel (encoder + decoder)
"""
import numpy as np
from torch import nn

# model config
INPUT_SIZE = 5
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
OUTPUT_SIZE = 2

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.linear = nn.Linear(input_size, embedding_size)
        self.lstm = nn.LSTMCell(embedding_size, hidden_size)

    def init_hidden(self, batch_size, hidden_size):
        # Initialize encoder hidden state
        h_0 = torch.zeros(batch_size, hidden_size).to(device)
        c_0 = torch.zeros(batch_size, hidden_size).to(device)
        nn.init.xavier_normal_(h_0)
        nn.init.xavier_normal_(c_0)
        return (h_0, c_0)
    
    def forward(self, X):
        init_hidden = self.init_hidden(X.shape[0], self.hidden_size)
        
        embedded = F.relu(self.linear(X))
        hidden_state = self.lstm(embedded, init_hidden)
        return hidden_state
    
    
class DecoderLSTM(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, X, encoder_hidden):        
        embedded = F.relu(self.linear1(X))
        hidden = self.lstm1(embedded, encoder_hidden)
        output = self.linear2(hidden[0])
        return output, hidden


class TrajectoryModel(nn.Module):
    def __init__(self, input_size: int, pred_len: int=30, use_teacher_forcing: bool=False):
        """
        Args:
            input_size (int): input size
            pred_len (int): prediction length
            use_teacher_forcing (bool): whether to use teacher forcing technique
        """
        super(TrajectoryModel, self).__init__()
        self.pred_len = pred_len
        self.use_teacher_forcing = use_teacher_forcing
        
        self.encoder = EncoderLSTM(input_size)
        self.decoder = DecoderLSTM()
    
    def forward(self, inp: np.ndarray, out: np.ndarray=None):
        """
        Args:
            inp (np.ndarray): (num_tracks x obs_len x input_size) input trajectory
            out (np.ndarray): (optional) (num_tracks x pred_len x input_size) output trajectory
        Returns:
            decoder_outputs (np.ndarray): (num_tracks x obs_len x input_size) decoder outputs
        """
        for i in range(input_length):
            encoder_input = inp[:, i, :]
            encoder_hidden = self.encoder(encoder_input)
    
        # Initialize decoder input with last coordinate in encoder
        decoder_hidden = encoder_hidden
        decoder_input = encoder_input[:, :2]
        decoder_outputs = torch.zeros((len(inp), self.pred_len, 2))

        # Decode hidden state in future trajectory
        for i in range(self.pred_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # record decoder_output
            decoder_outputs[:, i, :] = decoder_output

            # Use own predictions as inputs at next step
            if self.use_teacher_forcing and out is not None:
                decoder_input = out[:, i, :2]
            else:
                decoder_input = decoder_output
                
        return decoder_outputs