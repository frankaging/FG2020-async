from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, re, copy, itertools
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class MultiseqDataset(Dataset):
    """Multimodal dataset for (synchronous) time series and sequential data."""

    def __init__(self, modalities, dirs, regex, preprocess,
                 base_rate=None, truncate=False, item_as_dict=False):
        """Loads valence ratings and features for each modality.

        modalities -- names of each input modality
        dirs -- list of directories containing input features
        regex -- regex patterns for the filenames of each modality
        preprocess -- data pre-processing functions for pandas dataframes
        rates -- sampling rates of each modality
        base_rate -- base_rate to subsample/ovesample to
        truncate -- if true, truncate to modality with minimum length
        item_as_dict -- whether to return data as dictionary
        """
        # Store arguments
        self.modalities = modalities
        self.item_as_dict = item_as_dict
        self.ratios = {}

        # Convert to modality-indexed dictionaries
        dirs = {m: d for m, d in zip(modalities, dirs)}
        if type(regex) is not list:
            regex = [regex] * len(self.modalities)
        regex = {m: r for m, r in zip(modalities, regex)}
        if preprocess is None:
            preprocess = lambda x : x
        if type(preprocess) is not list:
            preprocess = [preprocess] * len(self.modalities)
        preprocess = {m: p for m, p in zip(modalities, preprocess)}

        # Load filenames into lists and extract regex-captured sequence IDs
        paths = dict()
        seq_ids = dict()
        for m in modalities:
            paths[m] = []
            seq_ids[m] = []
            for fn in os.listdir(dirs[m]):
                match = re.match(regex[m], fn)
                if not match:
                    continue
                paths[m].append(os.path.join(dirs[m], fn))
                # 16-30-1920x1080_linguisticFeatures
                # 16-30-1920x1080_left_linguisticFeatures
                # 16-30-1920x1080_2_linguisticFeatures
                # # 16-30-1920x1080_left
                if m == "ratings" or m == "ratings_timer":
                    seq_ids[m].append(fn.split(".")[0])
                else:
                    if len(fn.split("_")) == 2:
                        seq_ids[m].append(fn.split("_")[0])
                    elif len(fn.split("_")) > 2:
                        seq_ids[m].append("_".join(fn.split("_")[:-1]))                    
            # Sort by values of captured indices
            paths[m] = [p for _, p in sorted(zip(seq_ids[m], paths[m]))]
            seq_ids[m].sort()

        # Check that number and IDs of files/sequences are matched
        self.seq_ids = seq_ids[modalities[0]]
        for m in modalities:
            if len(paths[m]) != len(self.seq_ids):
                raise Exception("Number of files ({}) do not match.".\
                                format(len(paths[m])))
            if seq_ids[m] != self.seq_ids:
                raise Exception("Sequence IDs do not match.")

        # Load data from files
        self.data = {m: [] for m in modalities}
        self.orig = {m: [] for m in modalities}
        self.lengths = []
        for i in range(len(self.seq_ids)):
            seq_len = float('inf')
            # Load each input modality
            for m, data in self.data.items():
                fp = paths[m][i]
                if re.match("^.*\.p", fp):
                    # Load as numpy array
                    with open(fp, 'rb') as handle:
                        vgg_d = pickle.load(handle)
                    # hacky way to preprocess the vgg data
                    output = []
                    if m == "image":
                        for row in vgg_d:
                            output.append(row[1])
                    elif m == "image_timer":
                        for row in vgg_d:
                            output.append(row[0])
                    d = np.array(output)
                elif re.match("^.*\.(csv|txt)", fp):
                    # Use pandas to read and pre-process CSV files
                    d = pd.read_csv(fp)
                    d = np.array(preprocess[m](d))
                # Flatten inputs
                if len(d.shape) > 2:
                    d = d.reshape(d.shape[0], -1)
                # Store original data before resampling
                self.orig[m].append(d)
                data.append(d.tolist())
            self.lengths.append(seq_len)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        if self.item_as_dict:
            d = {m: self.data[m][i] for m in self.modalities}
            d['length'] = self.lengths[i]
            return d
        else:
            return tuple(self.data[m][i] for m in self.modalities)

    def normalize_(self):
        """Rescale all inputs to [-1, 1] range (in-place)."""
        # Find max and min for each dimension of each modality
        m_max = {m: np.stack([a.max(0) for a in self.data[m]]).max(0)
                  for m in self.modalities}
        m_min = {m: np.stack([a.min(0) for a in self.data[m]]).min(0)
                  for m in self.modalities}
        # Compute range per dim and add constant to ensure it is non-zero
        m_rng = {m: (m_max[m]-m_min[m]) for m in self.modalities}
        m_rng = {m: m_rng[m] * (m_rng[m] > 0) + 1e-10 * (m_rng[m] <= 0)
                  for m in self.modalities}
        # Actually rescale the data
        for m in self.modalities:
            self.data[m] = [(a-m_min[m]) / m_rng[m] * 2 - 1 for
                               a in self.data[m]]

    def normalize(self):
        """Rescale all inputs to [-1, 1] range (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.normalize_()
        return dataset

    def split_(self, n):
        """Splits each sequence into n chunks (in place)."""
        for m in self.modalities:
            self.data[m] = list(itertools.chain.from_iterable(
                [np.array_split(a, n, 0) for a in self.data[m]]))
        self.seq_ids = list(itertools.chain.from_iterable(
            [[i] * n for i in self.seq_ids]))
        self.lengths = [len(d) for d in self.data[self.modalities[0]]]

    def split(self, n):
        """Splits each sequence into n chunks (returns new dataset)."""
        dataset = copy.deepcopy(self)
        dataset.split_(n)
        return dataset

    @classmethod
    def merge(cls, set1, set2):
        """Merge two datasets."""
        if (set1.modalities != set2.modalities):
            raise Exception("Modalities need to match.")
        if (set1.base_rate != set2.base_rate):
            raise Exception("Base rates need to match.")
        merged = copy.deepcopy(set1)
        merged.orig.clear()
        merged.seq_ids += set2.seq_ids
        merged.rates = [merged.base_rate] * len(merged.modalities)
        merged.ratios = [1] * len(merged.modalities)
        for m in merged.modalities:
            merged.data[m] += copy.deepcopy(set2.data[m])
        return merged

def len_to_mask(lengths):
    """Converts list of sequence lengths to a mask tensor."""
    mask = torch.arange(max(lengths)).expand(len(lengths), max(lengths))
    mask = mask < torch.tensor(lengths).unsqueeze(1)
    return mask.unsqueeze(-1)

def pad_and_merge(sequences, max_len=None):
    """Pads and merges unequal length sequences into batch tensor."""
    dims = sequences[0].shape[1]
    lengths = [len(seq) for seq in sequences]
    if max_len is None:
        max_len = max(lengths)
    padded_seqs = torch.zeros(len(sequences), max_len, dims)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end, :] = torch.from_numpy(seq[:end,:])
    if len(sequences) == 1:
        padded_seqs = padded_seqs.float()
    return padded_seqs

def seq_collate(data):
    """Collates multimodal variable length sequences into padded batch."""
    padded = []
    n_modalities = len(data) #n_modalities = len(data[0])
    lengths = np.zeros(n_modalities, dtype=int)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data = zip(*data)
    for modality in data:
        m_lengths = [len(seq) for seq in modality]
        lengths = np.maximum(lengths, m_lengths)
    lengths = list(lengths)
    for modality in data:
        padded.append(pad_and_merge(modality, max(lengths)))
    mask = len_to_mask(lengths)
    return tuple(padded + [mask, lengths])

def seq_collate_dict(data):
    """Collate that accepts and returns dictionaries."""
    batch = {}
    modalities = [k for k in data[0].keys() if  k != 'length']
    data.sort(key=lambda d: d['length'], reverse=True)
    lengths = [d['length'] for d in data]
    for m in modalities:
        m_data = [d[m] for d in data]
        batch[m] = pad_and_merge(m_data, max(lengths))
    mask = len_to_mask(lengths)
    return batch, mask, lengths

def load_dataset(modalities, base_dir, subset, va="valence",
                 base_rate=2.0, truncate=False, item_as_dict=False):
    """Helper function specifically for loading TAC-EA datasets."""
    dirs = {
        'linguistic': os.path.join(base_dir, 'features', subset, 'bert'),
        'linguistic_timer': os.path.join(base_dir, 'features', subset, 'bert'),
        'ratings' : os.path.join(base_dir, 'ratings', subset, va),
        'ratings_timer' : os.path.join(base_dir, 'ratings', subset, va),
        'image': os.path.join(base_dir, 'features', subset, 'vgg'),
        'image_timer': os.path.join(base_dir, 'features', subset, 'vgg'),
        'acoustic': os.path.join(base_dir, 'features', subset, 'egemaps'),
        'acoustic_timer': os.path.join(base_dir, 'features', subset, 'egemaps'),
    }
    regex = {
        'linguistic': ".*\.csv",
        'linguistic_timer': ".*\.csv",
        'ratings' : ".*\.csv",
        'ratings_timer' : ".*\.csv",
        'image': ".*\.p",
        'image_timer': ".*\.p",
        'acoustic': ".*\.csv",
        'acoustic_timer': ".*\.csv"
    }
    preprocess = {
        'linguistic_timer': lambda df : df.loc[:,'time'],
        'linguistic': lambda df : df.loc[:,'feature_0':'feature_767'],
        'image' : None,
        'image_timer' : None,
        'ratings' : lambda df : (df.loc[:,va]),
        'ratings_timer' : lambda df : df.loc[:,'time'],
        'acoustic': lambda df : df.loc[:,' F0semitoneFrom27.5Hz_sma3nz_amean':' equivalentSoundLevel_dBp'],
        'acoustic_timer': lambda df : df.loc[:,' frameTime']
    }
    if 'ratings' not in modalities:
        modalities = modalities + ['ratings']
    if 'ratings_timer' not in modalities:
        modalities = modalities + ['ratings_timer']
    if 'emotient' in modalities:
        modalities = modalities + ['emotient_timer']
    if 'linguistic' in modalities:
        modalities = modalities + ['linguistic_timer']
    if 'image' in modalities:
        modalities = modalities + ['image_timer']
    if 'acoustic' in modalities:
        modalities = modalities + ['acoustic_timer']

    return MultiseqDataset(modalities, [dirs[m] for m in modalities],
                           [regex[m] for m in modalities],
                           [preprocess[m] for m in modalities],
                           base_rate, truncate, item_as_dict)

if __name__ == "__main__":
    # Test code by loading dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="../data",
                        help='data directory')
    parser.add_argument('--subset', type=str, default="Train",
                        help='whether to load Train/Valid/Test data')
    args = parser.parse_args()

    print("Loading data...")
    modalities = ['acoustic', 'linguistic', 'emotient', 'ratings']
    dataset = load_dataset(modalities, args.dir, args.subset)
    print("Testing batch collation...")
    data = seq_collate([dataset[i] for i in range(min(10, len(dataset)))])
    print("Batch shapes:")
    for d in data[:-2]:
        print(d.shape)
    print("Sequence lengths: ", data[-1])
    print("Checking through data for mismatched sequence lengths...")
    for i, data in enumerate(dataset):
        print("Subject, Video: ", dataset.seq_ids[i])
        acoustic, linguistic, emotient, ratings = data
        print(acoustic.shape, linguistic.shape, emotient.shape, ratings.shape)
        if not (len(acoustic) == len(ratings) and
                len(linguistic) == len(ratings) and
                len(emotient) == len(ratings)):
            print("WARNING: Mismatched sequence lengths.")
