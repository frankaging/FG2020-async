from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

def pad_shift(x, shift, padv=0.0):
    """Shift 3D tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x

def convolve(x, attn):
    """Convolve 3D tensor (x) with local attention weights (attn)."""
    stacked = torch.stack([pad_shift(x, i) for
                           i in range(attn.shape[2])], dim=-1)
    return torch.sum(attn.unsqueeze(2) * stacked, dim=-1)

class AsyncNaiveLinguistic(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(AsyncNaiveLinguistic, self).__init__()

        self.hidden_dim = {'linguistic' : 128, 'emotient' : 20, 'acoustic' : 64, 'image' : 128}
        self.lstm = dict()
        for mod in mods:
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        self.fuseOut = nn.Sequential(nn.Linear(128, 64),
                                  nn.Dropout(0.3),
                                  nn.Linear(64, 1))

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, seq_length, lstm_masks):
        # each input is 
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        n = len(inputs)
        t = max([len(seq) for seq in inputs])
        _inputs_hs = []
        for seq in inputs:
            _seq_hs = []
            _prev_h = torch.zeros(1, 128).to(self.device)
            _prev_c = torch.zeros(1, 128).to(self.device)
            for token in seq:
                _input_token = torch.unsqueeze(token['linguistic'][0], 0)
                _prev_h, _prev_c = self.lstm['linguistic'](_input_token, (_prev_h, _prev_c))
                _seq_hs.append(_prev_h)
            for i in range(t-len(seq)):
                _seq_hs.append(torch.zeros(1, 128).to(self.device))
            _seq_hs = torch.stack(_seq_hs).reshape(t, 128)
            _inputs_hs.append(_seq_hs)
        _inputs_hs = torch.stack(_inputs_hs)
        _outputs = self.fuseOut(_inputs_hs)
        _outputs = _outputs * lstm_masks.float()
        return _outputs

class AsyncNaiveMultimodal(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(AsyncNaiveMultimodal, self).__init__()
        self.mods = mods
        self.hidden_dim = {'linguistic' : 128, 'emotient' : 20, 'acoustic' : 64, 'image' : 128}
        self.lstm = dict()
        total_hidden = 0
        for mod in mods:
            total_hidden += self.hidden_dim[mod]
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        self.fuseOut = nn.Sequential(nn.Linear(total_hidden, 64),
                                  nn.Dropout(0.3),
                                  nn.Linear(64, 1))

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, seq_length, lstm_masks):
        # each input is 
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        n = len(inputs)
        t = max([len(seq) for seq in inputs])

        _inputs_hs = []

        for seq in inputs:

            _seq_hs = {}
            for mod in self.mods:
                _seq_hs[mod] = []

            _prev_h = {}
            _prev_c = {}
            for mod in self.mods:
                _prev_h[mod] = torch.zeros(1, self.hidden_dim[mod]).to(self.device)
                _prev_c[mod] = torch.zeros(1, self.hidden_dim[mod]).to(self.device)

            for token in seq:
                for mod in token.keys():
                    # assume it is 1 by 1
                    _input_token = torch.unsqueeze(token[mod][0], 0)
                    # print(_input_token.shape)
                    _prev_h[mod], _prev_c[mod] = self.lstm[mod](_input_token, (_prev_h[mod], _prev_c[mod]))
                for mod in token.keys():
                    _seq_hs[mod].append(_prev_h[mod].clone())
                    # we also want to append the other missing mods prev states
                for mod_in in self.mods:
                    if mod_in not in token.keys():
                        _seq_hs[mod_in].append(_prev_h[mod_in].clone())

            # padding and concat
            for i in range(t-len(seq)):
                for mod_in in self.mods:
                    _seq_hs[mod_in].append(torch.zeros(1, self.hidden_dim[mod_in]).to(self.device))
            _seq_all_hs = []
            for mod_in in self.mods:
                _seq_hs[mod_in] = torch.stack(_seq_hs[mod_in]).reshape(t, self.hidden_dim[mod_in])
                _seq_all_hs.append(_seq_hs[mod_in])

            _seq_all_hs = torch.cat(_seq_all_hs, 1)
            _inputs_hs.append(_seq_all_hs)

        _inputs_hs = torch.stack(_inputs_hs)
        _outputs = self.fuseOut(_inputs_hs)
        _outputs = _outputs * lstm_masks.float()
        return _outputs

class AsyncAttentionMultimodal(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(AsyncAttentionMultimodal, self).__init__()
        self.mods = mods
        self.hidden_dim = {'linguistic' : 128, 'emotient' : 20, 'acoustic' : 64, 'image' : 128}
        self.lstm = dict()
        total_hidden = 0
        for mod in mods:
            total_hidden += self.hidden_dim[mod]
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        self.attention = nn.Sequential(nn.Linear(total_hidden, 64),
                                  nn.Dropout(0.3),
                                  nn.Linear(64, len(mods)))

        self.out = dict()
        for mod in mods:
            self.out[mod] = nn.Sequential(nn.Linear(self.hidden_dim[mod], 16),
                                  nn.Dropout(0.3),
                                  nn.Linear(16, 1))
            self.add_module('out_{}'.format(mod), self.out[mod])

        self.final_out = nn.Linear(len(mods), 1)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, seq_length, lstm_masks):
        # each input is 
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        n = len(inputs)
        t = max([len(seq) for seq in inputs])

        _inputs_attended_hs = []

        for seq in inputs:

            _seq_hs = {}
            for mod in self.mods:
                _seq_hs[mod] = []

            _seq_hidden_ratings = {}
            for mod in self.mods:
                _seq_hidden_ratings[mod] = []

            _prev_h = {}
            _prev_c = {}
            for mod in self.mods:
                _prev_h[mod] = torch.zeros(1, self.hidden_dim[mod]).to(self.device)
                _prev_c[mod] = torch.zeros(1, self.hidden_dim[mod]).to(self.device)

            for token in seq:
                for mod in token.keys():
                    # assume it is 1 by 1
                    _input_token = torch.unsqueeze(token[mod][0], 0)
                    # print(_input_token.shape)
                    _prev_h[mod], _prev_c[mod] = self.lstm[mod](_input_token, (_prev_h[mod], _prev_c[mod]))
                for mod in token.keys():
                    _seq_hs[mod].append(_prev_h[mod].clone())
                    # we also want to append the other missing mods prev states
                for mod_in in self.mods:
                    if mod_in not in token.keys():
                        _seq_hs[mod_in].append(_prev_h[mod_in].clone())

            # padding and concat
            for i in range(t-len(seq)):
                for mod_in in self.mods:
                    _seq_hs[mod_in].append(torch.zeros(1, self.hidden_dim[mod_in]).to(self.device))
            _seq_all_hs = []
            _seq_all_hidden_ratings = []
            for mod_in in self.mods:
                _seq_hs[mod_in] = torch.stack(_seq_hs[mod_in]).reshape(t, self.hidden_dim[mod_in])
                _seq_hidden_ratings[mod_in] = self.out[mod_in](_seq_hs[mod_in])
                _seq_all_hs.append(_seq_hs[mod_in])
                _seq_all_hidden_ratings.append(_seq_hidden_ratings[mod_in])

            _seq_all_hs = torch.cat(_seq_all_hs, 1)
            _seq_all_hidden_ratings = torch.cat(_seq_all_hidden_ratings, 1)
            _attention_mod = self.attention(_seq_all_hs)
            _attended_rating = _seq_all_hidden_ratings*_attention_mod
            _inputs_attended_hs.append(_attended_rating)

        _inputs_attended_hs = torch.stack(_inputs_attended_hs)
        _outputs = self.final_out(_inputs_attended_hs)
        _outputs = _outputs * lstm_masks.float()
        return _outputs

class AsyncLSTMAttentionMultimodal(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(AsyncLSTMAttentionMultimodal, self).__init__()
        self.mods = mods
        self.hidden_dim = {'linguistic' : 512, 'emotient' : 20, 'acoustic' : 64, 'image' : 1024}
        self.lstm = dict()
        total_hidden = 0
        for mod in mods:
            total_hidden += self.hidden_dim[mod]
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        # attention layers 1
        self.attending_window = 2
        attInShape = total_hidden*self.attending_window
        h_att1 = 128
        att1_dropout = 0.3

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        # attention layers 2
        h_att2 = 256
        self.mem_dim = total_hidden
        att2_dropout = 0.3
        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        # lstm to remember fusion history
        self.attention_memory_dim = 256
        self.attention_lstm = nn.LSTMCell(self.mem_dim, self.attention_memory_dim)

        # out layers
        final_out = self.attention_memory_dim + total_hidden
        h_out = 256
        output_dim = 1
        out_dropout = 0.3
        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, seq_length, lstm_masks):
        # each input is 
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        n = len(inputs)
        t = max([len(seq) for seq in inputs])

        _inputs_hs = []
        _input_attention_cs = []

        for seq in inputs:

            _seq_hs = {}
            for mod in self.mods:
                _seq_hs[mod] = []

            _seq_cs = {}
            for mod in self.mods:
                _seq_cs[mod] = []

            _seq_attention_cs = []

            _prev_h = {}
            _prev_c = {}
            for mod in self.mods:
                _prev_h[mod] = torch.zeros(1, self.hidden_dim[mod]).to(self.device)
                _prev_c[mod] = torch.zeros(1, self.hidden_dim[mod]).to(self.device)

            _attention_prev_h = torch.zeros(1, self.attention_memory_dim).to(self.device)
            _attention_prev_c = torch.zeros(1, self.attention_memory_dim).to(self.device)

            for token in seq:
                for mod in token.keys():
                    # assume it is 1 by 1
                    _input_token = torch.unsqueeze(token[mod][0], 0)
                    # print(_input_token.shape)
                    _prev_h[mod], _prev_c[mod] = self.lstm[mod](_input_token, (_prev_h[mod], _prev_c[mod]))
                for mod in token.keys():
                    _seq_hs[mod].append(_prev_h[mod].clone())
                    _seq_cs[mod].append(_prev_c[mod].clone())
                    # we also want to append the other missing mods prev states
                for mod_in in self.mods:
                    if mod_in not in token.keys():
                        _seq_hs[mod_in].append(_prev_h[mod_in].clone())
                        _seq_cs[mod_in].append(_prev_c[mod_in].clone())

                # attending to each channel
                _new_cs = []
                _prev_cs = []
                for mod in self.mods:
                    _new_cs.append(_seq_cs[mod][-1])
                    if len(_seq_cs[mod]) > 1:
                        _prev_cs.append(_seq_cs[mod][-2])
                    else:
                        _prev_cs.append(torch.zeros(1, self.hidden_dim[mod]).to(self.device))
                _prev_cs = torch.cat(_prev_cs, dim=1)
                _new_cs = torch.cat(_new_cs, dim=1)
                cStar = torch.cat([_prev_cs,_new_cs], dim=1)
                attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
                attended = attention*cStar
                cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
                # lstm to remember the fusion history
                _attention_prev_h, _attention_prev_c = self.attention_lstm(cHat, (_attention_prev_h, _attention_prev_c))
                _seq_attention_cs.append(_attention_prev_c)

            # padding and concat
            for i in range(t-len(seq)):
                for mod_in in self.mods:
                    _seq_hs[mod_in].append(torch.zeros(1, self.hidden_dim[mod_in]).to(self.device))
                _seq_attention_cs.append(torch.zeros(1, self.attention_memory_dim).to(self.device))
                

            _seq_all_hs = []
            for mod_in in self.mods:
                _seq_hs[mod_in] = torch.stack(_seq_hs[mod_in]).reshape(t, self.hidden_dim[mod_in])
                _seq_all_hs.append(_seq_hs[mod_in])

            _seq_all_hs = torch.cat(_seq_all_hs, 1)
            _seq_all_attention_cs = torch.cat(_seq_attention_cs, 0)
            _inputs_hs.append(_seq_all_hs)
            _input_attention_cs.append(_seq_all_attention_cs)

        _inputs_hs = torch.stack(_inputs_hs)
        _input_attention_cs = torch.stack(_input_attention_cs)
        _final_out = torch.cat([_inputs_hs, _input_attention_cs], dim=2)
        _outputs = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(_final_out))))

        _outputs = _outputs * lstm_masks.float()
        return _outputs

##################################################################
#
# Backlog Models
#
##################################################################

class AsyncNaive(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(AsyncNaive, self).__init__()

        # input dims
        self.mods = mods
        self.dims = dims
        total_embed_size = 0
        total_hidden_size = 0
        self.hidden_dim = {'linguistic' : 300, 'emotient' : 20, 'acoustic' : 64, 'image' : 128}
        for mod in mods:
            total_embed_size += dims[mod]
            total_hidden_size += self.hidden_dim[mod]
        # config params TODO: from orginal paper https://github.com/pliang279/MFN/blob/master/test_mosi.py
        self.mem_dim = 128
        window_dim = 2

        # lstm layers
        self.lstm = dict()
        for mod in mods:
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        # sub out layer
        self.featureOut = dict()
        for mod in mods:
            self.featureOut[mod] = nn.Sequential(nn.Linear(self.hidden_dim[mod], 32),
                                  nn.Dropout(0.3),
                                  nn.Linear(32, 1))
            self.add_module('featureOut_{}'.format(mod), self.featureOut[mod])

        # Layer that computes attention from embeddings
        self.attn = nn.Sequential(nn.Linear(total_hidden_size, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, len(mods)),
                                  nn.Softmax(dim=2))

        self.simpleOut = nn.Sequential(nn.Linear(len(mods), 1))

        self.fuseOut = nn.Sequential(nn.Linear(total_hidden_size, 256),
                                  nn.Dropout(0.3),
                                  nn.Linear(256, 1))

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, seq_length, lstm_masks):
        # each input is 
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        self.h = dict()
        self.c = dict()
        all_hs = dict()
        for mod in self.mods:
            n = len(inputs)
            self.h[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
            self.c[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
        # get the max time point count
        t = max([len(seq) for seq in inputs])
        # print([len(seq) for seq in inputs])
        for i in range(t):
            # some seq will not have t, we will not propagate
            # print(i)
            # print(self.h['linguistic'][0])
            # prev time step
            # prev_h = dict()
            # for mod in self.mods:
            #     prev_h[mod] = self.h[mod]
            # prev_c = dict()
            # for mod in self.mods:
            #     prev_c[mod] = self.c[mod]

            new_h = dict()
            new_c = dict()
            # initialize these two differently
            for mod in self.mods:
                new_h[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
                new_c[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
            # propagate cells
            seq_id = 0
            for seq in inputs:
                if len(seq) <= i:
                    # TODO: we need to skip, because we dont have any inputs for this sequence
                    pass
                else:
                    # current sequence, lookup this time step, how many mods
                    for mod in self.mods:
                        if mod in seq[i].keys():
                            # update a couple of times if more than one feature vector existing in this time frame
                            _prev_h = torch.unsqueeze(self.h[mod][seq_id], 0)
                            _prev_c = torch.unsqueeze(self.c[mod][seq_id], 0)
                            for feature_in in seq[i][mod]:
                                # print(feature_in)
                                _h, _c = self.lstm[mod](torch.unsqueeze(feature_in, 0), (_prev_h, _prev_c))
                                _prev_h = _h
                                _prev_c = _c
                            new_h[mod][seq_id] = torch.squeeze(_prev_h, 0)
                            new_c[mod][seq_id] = torch.squeeze(_prev_c, 0)
                        else:
                            # basically keep the same as last time point
                            new_h[mod][seq_id] = self.h[mod][seq_id]
                            new_c[mod][seq_id] = self.c[mod][seq_id]
                seq_id = seq_id + 1

            # update
            for mod in self.mods:
                self.h[mod] = new_h[mod]
                self.c[mod] = new_c[mod]
                if mod not in all_hs.keys():
                    all_hs[mod] = []
                all_hs[mod].append(self.h[mod])
        # for vvv in all_hs['emotient']:
        #     print(vvv)
        #     print(vvv.shape)
        for mod in self.mods:
            all_hs[mod] = torch.stack(all_hs[mod]).reshape(n, t, self.hidden_dim[mod])
        # print(all_hs['emotient'].shape)
        
            # print(all_hs[mod].shape)
        # combining to get the output at each time step
        outputs = []
        _fusion_outputs = torch.cat([all_hs[mod] for mod in self.mods], 2)
        # print(_fusion_outputs.shape)
        # return
        _attended_outputs = self.fuseOut(_fusion_outputs)

        # _attention_score = self.attn(torch.cat([all_hs[mod] for mod in self.mods], 2))
        # # _attended_outputs = convolve(_fusion_outputs, _attention_score)
        # _attended_outputs =  torch.unsqueeze(torch.sum(_fusion_outputs.mul(_attention_score), dim=2), 2)

        # _attended_outputs = self.simpleOut(_fusion_outputs)
        # print(_attended_outputs.shape)

        # print(_attended_outputs)
        # print(_attended_outputs.shape)
        #         last_hs.append(all_hs[mod][i])
        #     last_hs.append(all_mems[i])
        #     last_hs = torch.cat(last_hs, dim=1)
        #     output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        #     outputs.append(output.unsqueeze(1))
        # outputs = torch.cat(outputs, dim=1)
        # print(outputs.shape)
        # print(lstm_masks.shape)
        # print(_attended_outputs.shape)
        # print(lstm_masks.shape)
        outputs = _attended_outputs * lstm_masks.float()
        # print(_attended_outputs)
        # print(outputs)
        return outputs

class AsyncLSTM(nn.Module):
    def __init__(self, mods, dims, output_dim,
                 device=torch.device('cuda:0')):
        super(AsyncLSTM, self).__init__()

        # input dims
        self.mods = mods
        self.dims = dims
        total_embed_size = 0
        total_hidden_size = 0
        self.hidden_dim = {'linguistic' : 512, 'emotient' : 20, 'acoustic' : 128, 'image' : 128}
        for mod in mods:
            total_embed_size += dims[mod]
            total_hidden_size += self.hidden_dim[mod]
        # config params TODO: from orginal paper https://github.com/pliang279/MFN/blob/master/test_mosi.py
        self.mem_dim = 128
        window_dim = 2
        attInShape = total_hidden_size*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_hidden_size+self.mem_dim
        h_att1 = 128
        h_att2 = 256
        h_gamma1 = 64
        h_gamma2 = 64
        h_out = 64
        att1_dropout = 0.0
        att2_dropout = 0.0
        gamma1_dropout = 0.2
        gamma2_dropout = 0.2
        out_dropout = 0.5

        # lstm layers
        self.lstm = dict()
        for mod in mods:
            self.lstm[mod] = nn.LSTMCell(dims[mod], self.hidden_dim[mod])
            self.add_module('lstm_{}'.format(mod), self.lstm[mod])

        # layers
        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, seq_length, lstm_masks):
        # each input is 
        n = -1
        t = -1
        # construct needs for each mods
        # TODO: this assume cuda is avaliable?
        self.h = dict()
        self.c = dict()
        all_hs = dict()
        all_cs = dict()
        all_mems = []
        for mod in self.mods:
            n = len(inputs)
            self.h[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
            self.c[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
        self.mem = torch.zeros(n, self.mem_dim).to(self.device)
        # get the max time point count
        t = max([len(seq) for seq in inputs])
        for i in range(t):
            # some seq will not have t, we will not propagate

            # prev time step
            prev_c = dict()
            for mod in self.mods:
                prev_c[mod] = self.c[mod]
            new_h = dict()
            new_c = dict()
            # initialize these two differently
            for mod in self.mods:
                new_h[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
                new_c[mod] = torch.zeros(n, self.hidden_dim[mod]).to(self.device)
            # propagate cells
            seq_id = 0
            for seq in inputs:
                if len(seq) <= i:
                    # TODO: we need to skip, because we dont have any inputs for this sequence
                    continue
                else:
                    # current sequence, lookup this time step, how many mods
                    for mod in self.mods:
                        if mod in seq[i].keys():
                            for feature_in in seq[i][mod]:
                                # print(torch.unsqueeze(self.c[mod][seq_id], 0).shape)
                                # print(self.h[mod][seq_id].is_cuda)
                                # print(self.c[mod][seq_id].is_cuda)
                                # print(feature_in.is_cuda)
                                _h, _c = self.lstm[mod](torch.unsqueeze(feature_in, 0),
                                                            (torch.unsqueeze(self.h[mod][seq_id], 0),
                                                             torch.unsqueeze(self.c[mod][seq_id], 0)))
                                new_h[mod][seq_id] = torch.squeeze(_h)
                                new_c[mod][seq_id] = torch.squeeze(_c)
                        else:
                            # basically keep the same as last time point
                            new_h[mod][seq_id] = self.h[mod][seq_id]
                            new_c[mod][seq_id] = self.c[mod][seq_id]
                        # print(self.c[mod].shape)
                seq_id = seq_id + 1

            # calculate attentions
            prev_cs = []
            new_cs = []
            for mod in self.mods:
                prev_cs.append(prev_c[mod])
                new_cs.append(new_c[mod])
            prev_cs = torch.cat(prev_cs, dim=1)
            new_cs = torch.cat(new_cs, dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            for mod in self.mods:
                self.h[mod] = new_h[mod]
                self.c[mod] = new_c[mod]
                if mod not in all_hs.keys():
                    all_hs[mod] = []
                if mod not in all_cs.keys():
                    all_cs[mod] = []
                all_hs[mod].append(self.h[mod])
                all_cs[mod].append(self.c[mod])

        # combining to get the output at each time step
        outputs = []
        for i in range(t):
            last_hs = []
            for mod in self.mods:
                last_hs.append(all_hs[mod][i])
            last_hs.append(all_mems[i])
            last_hs = torch.cat(last_hs, dim=1)
            output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        # print(outputs.shape)
        # print(lstm_masks.shape)
        outputs = outputs * lstm_masks.float()
        # print(outputs)
        return outputs