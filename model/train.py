"""Training code for synchronous multimodal LSTM model."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse
import copy
import csv

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import seq_collate_dict, load_dataset
from models import AsyncNaiveLinguistic, AsyncNaiveMultimodal, AsyncAttentionMultimodal, AsyncLSTMAttentionMultimodal

from random import shuffle
from operator import itemgetter
import pprint

import logging
logFilename = "./train_cnn.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(logFilename, 'w'),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

'''
helper to chunknize the data for each a modality
'''
def generateInputChunkHelper(data_chunk, length_chunk):
    # sort the data with length from long to short
    combined_data = list(zip(data_chunk, length_chunk))
    combined_data.sort(key=itemgetter(1),reverse=True)
    data_sort = []
    for pair in combined_data:
        data_sort.append(pair[0])
    # produce the operatable tensors
    data_sort_t = torch.tensor(data_sort, dtype=torch.float)
    return data_sort_t

'''
yielding training batch for the training process
'''
def generateTrainBatch(input_data, input_target, seq_length, args, batch_size=25, onEval=False):
    # TODO: support input_data as a dictionary
    # get chunk
    input_size = len(input_data) # all values have same size
    index = [i for i in range(0, input_size)]
    if not onEval:
        shuffle(index)
    shuffle_chunks = [i for i in chunks(index, batch_size)]
    # print(shuffle_chunks)
    for chunk in shuffle_chunks:
        # same across a single chunk
        input_data_chunk = [input_data[index] for index in chunk]
        target_chunk = [input_target[index] for index in chunk]
        length_chunk = [seq_length[index] for index in chunk]
        max_length = max(length_chunk)
        # produce the operatable tensors
        target_chunk_t = torch.tensor(target_chunk, dtype=torch.float)
        target_chunk_t = target_chunk_t[:,:max_length]
        # produce the operatable tensors
        for seq in input_data_chunk:
            for token_dict in seq:
                for mod in token_dict.keys():
                    token_dict[mod] = torch.tensor(token_dict[mod], dtype=torch.float)

        # we did not sort, since we are not doing any batch operations here?
        lstm_masks = torch.zeros(target_chunk_t.size()[0], target_chunk_t.size()[1], 1, dtype=torch.float)
        for i in range(lstm_masks.size()[0]):
            lstm_masks[i,:length_chunk[i]] = 1

        yield input_data_chunk, target_chunk_t, lstm_masks, length_chunk

def train(input_data, input_target, seq_length, model, criterion, optimizer, epoch, modality_comb, args):
    # TODO: support input_data as a dictionary
    # input_data = input_data['linguistic']

    model.train()
    data_num = 0
    loss = 0.0
    batch_num = 0
    # batch our data
    for (data, target, lstm_masks, length_chunk) in generateTrainBatch(input_data, input_target, seq_length, args):
        # we will train the model three times for each batch
        batch_loss_total = 0.0
        target = target.to(args.device)
        lstm_masks = lstm_masks.to(args.device)
        for seq in data:
            for token_dict in seq:
                for mod in token_dict.keys():
                    token_dict[mod] = token_dict[mod].to(args.device)
        for mods in modality_comb:
            # print("Current training batch modality includes: " + str(mods))
            corrupt_data = []
            # send all data to te device
            for seq in data:
                corrupt_token_dict = []
                for token_dict in seq:
                    corrupt_dict = {}
                    for mod in token_dict.keys():
                        if mod in mods:
                            corrupt_dict[mod] = token_dict[mod]
                    corrupt_token_dict.append(corrupt_dict)
                corrupt_data.append(corrupt_token_dict)
            output = model(corrupt_data, length_chunk, lstm_masks)
            batch_loss = criterion(output, torch.unsqueeze(target, 2))
            # Accumulate total loss for epoch
            batch_loss /= sum(length_chunk)
            batch_loss_total += batch_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Average over number of non-padding datapoints before stepping
        # batch_loss /= sum(length_chunk)
        batch_loss_total.backward()
        loss += batch_loss_total
        # Step, then zero gradients
        optimizer.step()
        optimizer.zero_grad()
        # Keep track of total number of time-points
        data_num += sum(length_chunk)
        logger.info('Batch: {:5d}\tJoint Loss: {:2.5f}'.\
              format(batch_num, batch_loss_total))
        batch_num += 1
    # Average losses and print
    loss /= batch_num
    logger.info('---')
    logger.info('Epoch: {}\t Joint Loss: {:2.5f}'.format(epoch, loss))
    return loss

def evaluate(input_data, input_target, seq_length, model, criterion, modality_comb, args, fig_path=None):

    # input_data = input_data['linguistic']

    model.eval()
    predictions = []
    data_num = 0
    loss, corr, ccc = 0.0, [], []
    count = 0

    local_best_output = []
    local_best_target = []
    local_best_index = 0
    index = 0
    local_best_ccc = -1
    mod_ccc = {}
    mod_corr = {}
    for ms in modality_comb:
        mod_ccc[tuple(ms)] = []
        mod_corr[tuple(ms)] = []
    for (data, target, lstm_masks, length_chunk) in generateTrainBatch(input_data,
                                                    input_target,
                                                    seq_length,
                                                    args,
                                                    batch_size=1):

        # we will train the model three times for each batch
        batch_loss_total = 0.0
        target = target.to(args.device)
        lstm_masks = lstm_masks.to(args.device)
        for seq in data:
            for token_dict in seq:
                for mod in token_dict.keys():
                    token_dict[mod] = token_dict[mod].to(args.device)
        for mods in modality_comb:
            # print("Current evaluation batch modality includes: " + str(mods))
            corrupt_data = []
            # send all data to te device
            for seq in data:
                corrupt_token_dict = []
                for token_dict in seq:
                    corrupt_dict = {}
                    for mod in token_dict.keys():
                        if mod in mods:
                            corrupt_dict[mod] = token_dict[mod]
                    corrupt_token_dict.append(corrupt_dict)
                corrupt_data.append(corrupt_token_dict)
            output = model(corrupt_data, length_chunk, lstm_masks)
            # Compute loss
            loss += criterion(output, torch.unsqueeze(target, 2))
            # Keep track of total number of time-points
            data_num += sum(length_chunk)
            # Compute correlation and CCC of predictions against ratings
            output_a = torch.squeeze(torch.squeeze(output, dim=2), dim=0).cpu().numpy()
            target_a = torch.squeeze(target, dim=0).cpu().numpy()
            if count == 0:
                # print(output)
                # print(target)
                count += 1
            curr_ccc = eval_ccc(output_a, target_a)
            mod_ccc[tuple(mods)].append(curr_ccc)
            mod_corr[tuple(mods)].append(pearsonr(output_a, target_a)[0])
            
        index += 1
        # if curr_ccc > local_best_ccc:
        #     local_best_output = output
        #     local_best_target = target
        #     local_best_index = index
        #     local_best_ccc = curr_ccc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # print(data_num)
    # Average losses and print
    loss /= data_num
    # Average statistics and print
    stats_dict = {}
    for ms in modality_comb:
        stats = {'corr': np.mean(mod_corr[tuple(ms)]), 'corr_std': np.std(mod_corr[tuple(ms)]),
             'ccc': np.mean(mod_ccc[tuple(ms)]), 'ccc_std': np.std(mod_ccc[tuple(ms)])}
        logger.info(str(ms)+' Joint Evaluation\tLoss: {:2.5f}\tCorr: {:0.3f}\tCCC: {:0.9f}'.\
            format(loss, stats['corr'], stats['ccc']))
        stats_dict[tuple(ms)] = stats
    return predictions, loss, stats_dict 

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against ratings for representative fits."""
    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[-4:][::-1],
                              np.argsort(metric)[:4]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [dataset.orig['ratings'][i] for i in sel_idx]
    sel_pred = [predictions[i] for i in sel_idx]
    for i, (true, pred, m) in enumerate(zip(sel_true, sel_pred, sel_metric)):
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()
        args.axes[i,j].plot(true, 'b-')
        args.axes[i,j].plot(pred, 'c-')
        args.axes[i,j].set_xlim(0, len(true))
        args.axes[i,j].set_ylim(-1, 1)
        args.axes[i,j].set_title("Fit = {:0.3f}".format(m))
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

def plot_eval(pred_sort, ccc_sort, actual_sort, seq_sort, window_size=5):
    sub_graph_count = len(pred_sort)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, 11):
        ax = fig.add_subplot(2, 5, i)

        ccc = ccc_sort[i-1]
        pred = pred_sort[i-1]
        actual = actual_sort[i-1]
        seq = seq_sort[i-1]
        minL = min(len(pred), len(actual))
        pred = pred[:minL]
        actual = actual[:minL]
        # rescale y
        for i in range(0, len(pred)):
            pred[i] = (pred[i]-0.5)*2.0
        for i in range(0, len(actual)):
            actual[i] = (actual[i]-0.5)*2.0
        t = []
        curr_t = 0.0
        for i in pred:
            t.append(curr_t)
            curr_t += window_size
        pred_line, = ax.plot(t, pred, '-' , color='r', linewidth=2.0, label='Prediction')
        ax.legend()
        actual_line, = ax.plot(t, actual, '-', color='b', linewidth=2.0, label='True')
        ax.legend()
        ax.set_ylabel('valence(0-1)')
        ax.set_xlabel('time(s)')
        ax.set_ylim(-1, 1)
        ax.set_title('ccc='+str(ccc)[:5]+"-vid="+seq)
    plt.show()
    # plt.savefig("./lstm_save/top_ccc.png")

def save_predictions(dataset, predictions, path):
    for p, seq_id in zip(predictions, dataset.seq_ids):
        df = pd.DataFrame(p, columns=['rating'])
        fname = "target_{}_{}_normal.csv".format(*seq_id)
        df.to_csv(os.path.join(path, fname), index=False)

def save_params(args, model, train_stats, test_stats):
    fname = 'param_hist.tsv'
    df = pd.DataFrame([vars(args)], columns=vars(args).keys())
    df = df[['modalities', 'batch_size', 'split', 'epochs', 'lr',
             'sup_ratio', 'base_rate']]
    for k in ['ccc_std', 'ccc']:
        v = train_stats.get(k, float('nan'))
        df.insert(0, 'train_' + k, v)
    for k in ['ccc_std', 'ccc']:
        v = test_stats.get(k, float('nan'))
        df.insert(0, 'test_' + k, v)
    df.insert(0, 'model', [model.__class__.__name__])
    df['embed_dim'] = model.embed_dim
    df['h_dim'] = model.h_dim
    df['attn_len'] = model.attn_len
    if type(model) is MultiARLSTM:
        df['ar_order'] = [model.ar_order]
    else:
        df['ar_order'] = [float('nan')]
    df.set_index('model')
    df.to_csv(fname, mode='a', header=(not os.path.exists(fname)), sep='\t')

def save_checkpoint(modalities, mod_dimension, window_size, model, path):
    checkpoint = {'modalities': modalities, 'mod_dimension' : mod_dimension, 'window_size' : window_size, 'model': model.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def load_data(modalities, data_dir, eval_dir=None):
    print("Loading data...")
    if eval_dir == None:
        train_data = load_dataset(modalities, data_dir, 'Train',
                                base_rate=args.base_rate,
                                truncate=True, item_as_dict=True)
        # train_data = None
        test_data = load_dataset(modalities, data_dir, 'Valid',
                                base_rate=args.base_rate,
                                truncate=True, item_as_dict=True)
        print("Done.")
        return train_data, test_data
    eval_data = load_dataset(modalities, data_dir, eval_dir,
                             base_rate=args.base_rate,
                             truncate=True, item_as_dict=True)
    print("Loading Eval Set Done.")
    return eval_data

def videoInputHelper(input_data, window_size, channel):
    # channel features
    vectors_raw = input_data[channel]
    ts = input_data[channel+"_timer"]
    # remove nan values
    vectors = []
    for vec in vectors_raw:
        inner_vec = []
        for v in vec:
            if np.isnan(v):
                inner_vec.append(0)
            else:
                inner_vec.append(v)
        vectors.append(inner_vec)

    #  get the window size and repeat rate if oversample is needed
    oversample = int(window_size[channel]/window_size['ratings'])
    window_size = window_size[channel]

    video_vs = []
    count_v = 0
    current_time = 0.0
    window_vs = []
    while count_v < len(vectors):
        t = ts[count_v]
        if type(t) == list:
            t = t[0]
        if t <= current_time + window_size:
            window_vs.append(vectors[count_v])
            count_v += 1
        else:
            for i in range(0, oversample):
                video_vs.append(window_vs)
            window_vs = []
            current_time += window_size
    return video_vs

def ratingInputHelper(input_data, window_size):
    ratings = input_data['ratings']
    ts = input_data['ratings_timer']
    window_size = window_size['ratings']

    current_time = 0.0
    count_r = 0
    window_rs = []
    video_rs = []
    while count_r < len(ratings):
        t = ts[count_r]
        if t <= current_time + window_size:
            window_rs.append(ratings[count_r])
            count_r += 1
        else:
            avg_r = sum(window_rs)*1.0/len(window_rs)
            video_rs.append(avg_r)
            window_rs = []
            current_time += window_size
    return video_rs

'''
Construct inputs for different channels: emotient, linguistic, ratings, etc..
'''
def constructInput(input_data, window_size, channels):
    ret_input_features = {}
    ret_ratings = []
    for data in input_data:
        # channel features
        minL = 99999999
        for channel in channels:
            video_vs = videoInputHelper(data, window_size, channel)
            # print("Channel: " + channel + " ; vector size: " + str(len(video_vs)))
            if channel not in ret_input_features.keys():
                ret_input_features[channel] = []
            ret_input_features[channel].append(video_vs)
            if len(video_vs) < minL:
                minL = len(video_vs)
        video_rs = ratingInputHelper(data, window_size)
        # print("video_rs vector size: " + str(len(video_rs)))
        if len(video_rs) < minL:
            minL = len(video_rs)
        # concate
        for channel in channels:
             ret_input_features[channel][-1] = ret_input_features[channel][-1][:minL]
        ret_ratings.append(video_rs[:minL])
    return ret_input_features, ret_ratings

def padInputHelper(input_data, dim, old_version=False):
    output = []
    max_num_vec_in_window = 0
    max_num_windows = 0
    seq_lens = []
    for data in input_data:
        if max_num_windows < len(data):
            max_num_windows = len(data)
        seq_lens.append(len(data))
        if max_num_vec_in_window < max([len(w) for w in data]):
            max_num_vec_in_window = max([len(w) for w in data])

    padVec = [0.0]*dim
    for vid in input_data:
        vidNewTmp = []
        for wind in vid:
            if not old_version:
                # window might not contain any vector due to null during this window
                if len(wind) != 0:
                    windNew = [padVec] * max_num_vec_in_window
                    # pad with last frame features in this window
                    windNew[:len(wind)] = wind
                    vidNewTmp.append(windNew)
                    # update the pad vec to be the last avaliable vector
                else:
                    windNew = [padVec] * max_num_vec_in_window
                    vidNewTmp.append(windNew)
            else:
                windNew = [padVec] * max_num_vec_in_window
                windNew[:len(wind)] = wind
                vidNewTmp.append(windNew)
        vidNew = [[padVec] * max_num_vec_in_window]*max_num_windows
        vidNew[:len(vidNewTmp)] = vidNewTmp
        output.append(vidNew)
    return output, seq_lens

'''
pad every sequence to max length, also we will be padding windows as well
'''
def padInput(input_data, channels, dimensions):
    # input_features <- list of dict: {channel_1: [117*features],...}
    ret = {}
    seq_lens = []
    for channel in channels:
        pad_channel, seq_lens = padInputHelper(input_data[channel], dimensions[channel])
        ret[channel] = pad_channel
    return ret, seq_lens

'''
pad targets
'''
def padRating(input_data, max_len):
    output = []
    # pad ratings
    for rating in input_data:
        ratingNew = [0]*max_len
        ratingNew[:len(rating)] = rating
        output.append(ratingNew)
    return output

def getSeqList(seq_ids):
    ret = []
    for seq_id in seq_ids:
        ret.append(seq_id[0]+"_"+seq_id[1])
    return ret

def extractEmitTimersFeatures(input_data, mods, reduce=False, reduceWindow=1):

    # final output
    final_total_timer = []
    final_total_feature = []

    for seq in input_data:
        timer_seq = []
        mod_seq = []
        feature_seq = []
        counter_map = {}
        timer_map = {}
        feature_map = {}
        total_ppt = 0
        for mod in mods:
            counter_map[mod] = 0
            timer = mod + "_timer"
            timer_map[mod] = seq[timer]
            feature_map[mod] = seq[mod]
            total_ppt += len(timer_map[mod])
        while total_ppt > 0:
            next_mod = ""
            minT = 9999
            for mod in mods:
                if counter_map[mod] < len(timer_map[mod]) and timer_map[mod][counter_map[mod]] < minT:
                    minT = timer_map[mod][counter_map[mod]]
                    next_mod = mod
                    next_feature = feature_map[mod][counter_map[mod]]
            # remove nan values
            add_in = True
            for v in next_feature:
                if np.isnan(v):
                    add_in = False
                    break
            if add_in:
                feature_seq.append(next_feature)
                timer_seq.append(minT)
                mod_seq.append(next_mod)
            counter_map[next_mod] = counter_map[next_mod] + 1
            total_ppt = total_ppt - 1
        current_time = 0.0
        current_timer = []
        current_mod = []
        current_feature = []
        total_timer = []
        total_mod = []
        total_feature = []
        final_timer = []
        final_feature = []
        if not reduce:
            # consolidation same time point if exist
            prev_time = -1
            feature_dict = {}
            for i in range(0, len(timer_seq)):
                if timer_seq[i] == prev_time:
                    # append to the last time point
                    if mod_seq[i] in final_feature[-1].keys():
                        final_feature[-1][mod_seq[i]].append(feature_seq[i])
                    else:
                        final_feature[-1][mod_seq[i]] = [feature_seq[i]]
                else:
                    feature_dict[mod_seq[i]] = [feature_seq[i]]
                    final_feature.append(feature_dict)
                    final_timer.append(timer_seq[i])
                    feature_dict = {}
                prev_time = timer_seq[i]
        # print(timer_seq)
        if reduce:
            for i in range(0, len(timer_seq)):
                if timer_seq[i] > current_time + reduceWindow:
                    total_timer.append(current_timer)
                    total_mod.append(current_mod)
                    total_feature.append(current_feature)
                    # do not miss this current time step
                    current_timer = [timer_seq[i]]
                    current_mod = [mod_seq[i]]
                    current_feature = [feature_seq[i]]
                    current_time += reduceWindow
                    continue
                current_timer.append(timer_seq[i])
                current_mod.append(mod_seq[i])
                current_feature.append(feature_seq[i])
            # add the residual things to total
            total_timer.append(current_timer)
            total_mod.append(current_mod)
            total_feature.append(current_feature)
            # print(total_timer)
            # print(total_mod)
            # print(total_feature)
            index = 0  
            for interval in total_timer:
                final_timer.append(interval[-1])
                feature_dict = {}
                i = 0
                for mod in total_mod[index]:
                    if mod in feature_dict.keys():
                        feature_dict[mod].append(total_feature[index][i])
                    else:
                        feature_dict[mod] = [total_feature[index][i]]
                    i += 1
                final_feature.append(feature_dict)
                index += 1
        # print(final_timer)
        # print(final_feature[0]['emotient'])
        # print(final_feature)
        # print(final_timer)
        final_total_timer.append(final_timer)
        final_total_feature.append(final_feature)
    return final_total_timer, final_total_feature

def main(args):
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Convert device string to torch.device
    args.device = (torch.device(args.device) if torch.cuda.is_available()
                   else torch.device('cpu'))

    # loss function define
    criterion = nn.MSELoss(reduction='sum')

    # TODO: i think for this project, we will assume always we have 3 modalities
    #       it is just we are missing data.
    args.modalities = ['image', 'linguistic', 'acoustic']
    mod_dimension = {'linguistic' : 768, 'acoustic' : 88, 'image' : 4096}
    # TODO: Finish the input encoding
    train_data, test_data = load_data(args.modalities, args.data_dir)
    # train_data = load_data(args.modalities, args.data_dir, 'UnitTest')
    # test_data = train_data
    # train_data = [{'acoustic':[[1],[2],[3]], 'acoustic_timer':[1,2,3],
    #               'emotient':[[11],[22],[33]], 'emotient_timer':[3,4,5],
    #               'linguistic':[[111],[222],[333]], 'linguistic_timer':[4,5,6]}]
    window_size = 1
    windowed = True
    # print(train_data[:1])
    # print(train_data)

    final_timer, final_feature = extractEmitTimersFeatures(train_data, args.modalities, windowed, window_size)
    # print(final_feature)
    seq_length = [len(seq) for seq in final_timer]
    final_rating = []
    for i in range(0, len(final_timer)):
        final_rating.append(np.interp(final_timer[i], train_data[i]['ratings_timer'], train_data[i]['ratings']).tolist())
    max_length = max(seq_length)
    final_rating_padded = padRating(final_rating, max_length)

    final_timer_test, final_feature_test = extractEmitTimersFeatures(test_data, args.modalities, windowed, window_size)
    seq_length_test = [len(seq) for seq in final_timer_test]
    final_rating_test = []
    for i in range(0, len(final_timer_test)):
        final_rating_test.append(np.interp(final_timer_test[i], test_data[i]['ratings_timer'], test_data[i]['ratings']).tolist())
    max_length_test = max(seq_length)
    final_rating_padded_test = padRating(final_rating_test, max_length_test)

    model = AsyncLSTMAttentionMultimodal(mods=args.modalities,  dims=mod_dimension, output_dim=1, device=args.device)
    # Setting the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)
    # Train and save best model
    best_ccc = {}
    joint_ccc = -1.0
    modality_comb = [['image', 'linguistic', 'acoustic']]
    for mods in modality_comb:
        best_ccc[tuple(mods)] = -1.0
    single_best_ccc = -1
    for epoch in range(1, 1001):
        print('---')
        train(final_feature, final_rating_padded, seq_length,
            model, criterion, optimizer, epoch, modality_comb, args)
        if epoch % args.eval_freq == 0:
            with torch.no_grad():
                # final_feature_test, final_rating_padded_test, seq_length_test
                pred, loss, stats  =\
                    evaluate(final_feature_test, final_rating_padded_test, seq_length_test,
                            model, criterion, modality_comb, args)
                # reduce LR if necessary
                scheduler.step(loss)
            joint_ccc_curr = -1.0
            for mods in modality_comb:
                joint_ccc_curr += stats[tuple(mods)]['ccc']
                if stats[tuple(mods)]['ccc'] > best_ccc[tuple(mods)]:
                    best_ccc[tuple(mods)] = stats[tuple(mods)]['ccc']
                    path = os.path.join("./save_model/Async-LSTM-"+ "-".join(mods) +".pth")
                    save_checkpoint(args.modalities, mod_dimension, window_size, model, path)
            if joint_ccc_curr > joint_ccc:
                joint_ccc = joint_ccc_curr
                path = os.path.join("Async-LSTM-"+ "joint" +".pth")
                save_checkpoint(args.modalities, mod_dimension, window_size, model, path)
        print("===current best performance stats==='")
        for mods in modality_comb:
            print("BEST " + str(mods) + " : " + str(best_ccc[tuple(mods)]))
        print("====================================")
            # if stats['max_ccc'] > single_best_ccc:
            #     single_best_ccc = stats['max_ccc']
            #     logger.info('===single_max_predict===')
            #     logger.info(local_best_output)
            #     logger.info(local_best_target)
            #     logger.info(local_best_index)
            #     logger.info('===end single_max_predict===')
            # logger.info('CCC_STATS\tSINGLE_BEST: {:0.9f}\tBEST: {:0.9f}'.\
            # format(single_best_ccc, best_ccc))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modalities', type=str, default=None, nargs='+',
                        help='input modalities (default: all')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--split', type=int, default=1, metavar='N',
                        help='sections to split each video into (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--sup_ratio', type=float, default=0.5, metavar='F',
                        help='teacher-forcing ratio (default: 0.5)')
    parser.add_argument('--base_rate', type=float, default=2.0, metavar='N',
                        help='sampling rate to resample to (default: 2.0)')
    parser.add_argument('--log_freq', type=int, default=5, metavar='N',
                        help='print loss N times every epoch (default: 5)')
    parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                        help='evaluate every N epochs (default: 1)')
    parser.add_argument('--save_freq', type=int, default=10, metavar='N',
                        help='save every N epochs (default: 10)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use (default: cuda:0 if available)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='flag to visualize predictions (default: false)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='whether to normalize inputs (default: false)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='evaluate on test set (default: false)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate on eval set (default: false)')
    parser.add_argument('--load', type=str, default=None,
                        help='path to trained model (either resume or test)')
    parser.add_argument('--data_dir', type=str, default="../../AffWildv1-data",
                        help='path to data base directory')
    parser.add_argument('--save_dir', type=str, default="./lstm_save",
                        help='path to save models and predictions')
    args = parser.parse_args()
    main(args)
