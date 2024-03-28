"""
A function to prepare the dataloaders
"""
# Import packages
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from llmprop_utils import *
import pymatgen
import pymatgen.core.structure 
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint,AGNIFingerprints

np.random.seed(42)

def tokenize(tokenizer, dataframe, max_length, pooling='cls'):
    """
    1. Takes in the the list of input sequences and return 
    the input_ids and attention masks of the tokenized sequences
    2. max_length = the max length of each input sequence 
    """
    if pooling == 'cls':
        encoded_corpus = tokenizer(text=["[CLS] " + str(descr) for descr in dataframe.description.tolist()],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True,
                    
                                    )
    elif pooling == 'mean':
        encoded_corpus = tokenizer(text=dataframe.description.tolist(),
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True) 
    elif pooling == 'max':
        encoded_corpus = tokenizer(text=dataframe.description.tolist(),
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation='longest_first',
                                    max_length=max_length, # According to ByT5 paper
                                    return_attention_mask=True) 
        
    input_ids = encoded_corpus['input_ids']
    attention_masks = encoded_corpus['attention_mask']
    
    return input_ids, attention_masks


def pymatgen_deserialize_dicts(dicts, to_unit_cell: bool = False):
        s = pymatgen.core.structure.Structure.from_dict(dicts)
        if to_unit_cell:
                for site in s.sites:
                    site.to_unit_cell(in_place=True)
        return s


def featurize_structure(structure):
        CrystalNNFinger = []
        cnnfp = AGNIFingerprints(directions=["x", "y", "z"])
        # cnnfp = CrystalNNFingerprint.from_preset("ops")  # 'cn'
        # convert dict ss to pymatgen.structure
        structure = pymatgen_deserialize_dicts(structure)
        for i in range(len(structure)):
            try:
                sites_feature = np.array(cnnfp.featurize(structure, i))
            except:
                sites_feature = np.zeros((24,))
            CrystalNNFinger.append(sites_feature)
        return CrystalNNFinger


def create_dataloaders(tokenizer, dataframe, max_length, batch_size, property_value="band_gap", pooling='cls', normalize=False,
                       normalizer='z_norm'):
    """
    Dataloader which arrange the input sequences, attention masks, and labels in batches
    and transform the to tensors
    """
    input_ids, attention_masks = tokenize(tokenizer, dataframe, max_length, pooling=pooling)
    labels = dataframe[property_value].to_numpy()

    # structures = dataframe['structure'].tolist()
    # featurized_list = [np.sum(featurize_structure(eval(s)), axis=0) for s in structures]
    # featurized_tensor = torch.tensor(featurized_list).float()
    
    input_tensor = torch.tensor(input_ids)
    mask_tensor = torch.tensor(attention_masks)
    labels_tensor = torch.tensor(labels)

    # 沿着指定的维度（假设是第0维）拼接两个张量
    # concatenated_input_tensor = torch.cat((featurized_tensor, input_tensor), dim=-1)
    
    if normalize:
        if normalizer == 'z_norm':
            normalized_labels = z_normalizer(labels_tensor)
        elif normalizer == 'mm_norm':
           normalized_labels = min_max_scaling(labels_tensor)
        elif normalizer == 'ls_norm':
            normalized_labels = log_scaling(labels_tensor)
        elif normalizer == 'no_norm':
            normalized_labels = labels_tensor

        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor, normalized_labels)
    else:
        dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # Set the shuffle to False for now since the labels are continues values check later if this may affect the result

    return dataloader


