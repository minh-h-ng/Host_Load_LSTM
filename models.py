import torch.nn as nn
import logging
import sys
import torch

from LSTM import LSTMHardSigmoid
from ConvLSTM import ConvLSTM

# Initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class VanillaLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        #self.input_size = config['input_length']
        self.rnn = nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True)
        if self.config['model'] == 'EC_LSTM':
            self.decoder = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        #self.rnn = LSTMHardSigmoid(1, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, config['output_length'])

    def forward(self, x):
        # forward propagate LSTM
        if self.config['lstm_state'] == 'stateless':
            out, self.hidden = self.rnn(x)
        elif self.config['lstm_state'] == 'stateful':
            out, self.hidden = self.rnn(x, self.hidden)
            # detach the hidden state from its history so we don't backpropagate through the entire history
            self.hidden = self.hidden.detach()
        else:
            logger.info('Program exiting: lstm_state in config not recognizable')
            sys.exit(0)

        if self.config['model'] == 'EC_LSTM':
            out, self.hidden = self.decoder(out)

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

    def init_hidden(self, config):
        return (torch.zeros(self.num_layers, config['truncated_length'], self.hidden_size),
                torch.zeros(self.num_layers, config['truncated_length'], self.hidden_size))