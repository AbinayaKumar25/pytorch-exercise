import torch
import torch.nn as nn
import torch.autograd as autograd
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()
#use_cuda = False

class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=False):

        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.n_hidden = n_hidden

        QRNN = False
        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        elif self.cell_type == "QRNN":
            QRNN = True
            from torchqrnn import QRNN as QRNNcell
            cell = QRNNcell
        else:
            raise NotImplementedError
        """
        for i in range(n_layers):
            if i == 0:
                if QRNN:
                    c = QRNNcell(n_input, n_hidden, dropout=dropout, use_cuda=use_cuda)
                else:
                    c = cell(n_input, n_hidden, dropout=dropout)
            else:
                if QRNN:
                    c = QRNNcell(n_hidden, n_hidden, dropout=dropout, use_cuda=use_cuda)
                else:
                    c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        """
        for i in range(n_layers):
            if QRNN:
                c = QRNNcell(n_input, n_hidden, dropout=dropout, use_cuda=use_cuda)
            else:
                c = cell(n_input, n_hidden, dropout=dropout)
            layers.append(c)
        
        self.cells = nn.Sequential(*layers)
    
    def forward(self, inputs, lengths=None, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        outputs = []
        hidden_states = []
        if lengths is not None:
            result = torch.zeros(inputs.shape[0]-1, inputs.shape[1],self.n_hidden).cuda()
        else:
            result = torch.zeros(inputs.shape[0], inputs.shape[1],self.n_hidden).cuda()

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                #inputs, _ = self.drnn_layer(cell, inputs, dilation, lengths)
                output, hidden_state = self.drnn_layer(cell, inputs, dilation, lengths)
                hidden_states.append(hidden_state)
            else:

                #inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, lengths, hidden[i])
                output, hidden_state = self.drnn_layer(cell, inputs, dilation, lengths, hidden[i])
                hidden_states.append(hidden_state)
            result = torch.add(result, output[:result.shape[0], :, :])
            #outputs.append(inputs[-dilation:])
            outputs.append(output[-dilation:])

        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        if self.batch_first:
            result = result.transpose(0, 1)
        return result, hidden_states

    def drnn_layer(self, cell, inputs, rate, lengths, hidden=None):

        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, dilated_steps, expanded_lengths = self._pad_inputs(inputs, n_steps, rate, lengths)
        dilated_inputs, new_lengths = self._prepare_inputs(inputs, rate, expanded_lengths)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, new_lengths)
        else:
            #hidden, new_lengths = self._prepare_inputs(hidden, rate, expanded_lengths)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, new_lengths, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, lengths, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        #pack sequences of varying length
        if lengths is not None:
            pads = torch.sum(lengths, 1)
            dilated_inputs = pack_padded_sequence(dilated_inputs, pads, enforce_sorted=False)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)
        if lengths is not None:
            dilated_outputs, hidden = pad_packed_sequence(dilated_outputs)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate, lengths):
        iseven = (n_steps % rate) == 0

        expanded_lengths = []
        seq = inputs.shape[0]

        if lengths is not None:
            for length in lengths:
                tmp = [1] * length.item() + [0] * (seq-length.item())
                expanded_lengths.append(torch.Tensor(tmp).unsqueeze(0))
            expanded_lengths = torch.cat(expanded_lengths)

        if not iseven:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2))
            
            
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs, autograd.Variable(zeros_)))
            if lengths is not None:
                zeros_pad_len = torch.zeros(expanded_lengths.shape[0], dilated_steps * rate - inputs.size(0))
                expanded_lengths = torch.cat((expanded_lengths, zeros_pad_len), 1)
            
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps, expanded_lengths

    def _prepare_inputs(self, inputs, rate, expanded_lengths):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        new_lengths = None
        if len(expanded_lengths) > 0:
            new_lengths = torch.cat([expanded_lengths[:, j::rate] for j in range(rate)])

        return dilated_inputs, new_lengths

    def init_hidden(self, batch_size, hidden_dim):
        hidden = autograd.Variable(torch.zeros(batch_size, hidden_dim))
        #pdb.set_trace()
        if use_cuda:
            hidden = hidden.cuda()
        if self.cell_type == "LSTM":
            memory = autograd.Variable(torch.zeros(batch_size, hidden_dim))
            if use_cuda:
                memory = memory.cuda()
            return (hidden, memory)
        else:
            return hidden
