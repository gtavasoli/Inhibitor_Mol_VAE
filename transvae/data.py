import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from transvae.tvae_util import *

def vae_data_gen(mols, props, char_dict):
    """
    Encodes the input SMILES string into a tensor with tag IDs.
    Parameters:
        mols (np.array, required): array containing the molecular structure
        props (np.array, required): array containing scalar chemical property values
        char_dict (dict, required): dictionary mapping tags to integer IDs
    Returns:
        encoded_data (torch.tensor): a tensor containing the encoding of each SMILES string
    """
    # print(mols)
    smiles = mols[:,0]  # Extract SMILES string
    #print(smiles)
    if props is None:  # If no attributes are provided, create an all-zero attribute array
        props = np.zeros(smiles.shape)
    del mols  # Delete the original molecule array to save memory
    smiles = [tokenizer(x) for x in smiles]  # Tokenize a SMILES string
    #print(smiles)
    print(len(smiles))
    encoded_data = torch.empty((len(smiles), 136))  # Create an empty encoded data tensor, where 224 is the longest number of tokens in the SMILES formula.
    #print(encoded_data)
    for j, smi in enumerate(smiles):
        #encoded_smi = encode_smiles(smi, 126, char_dict)  # Encoding SMILES strings
        encoded_smi = encode_smiles(smi, 134, char_dict)  # 编码SMILES字符串
        encoded_smi = [0] + encoded_smi  # Add a start marker before encoding
        #print(len(encoded_smi))
        encoded_data[j,:-1] = torch.tensor(encoded_smi)  # Fill the encoded data tensor
        
        #print(props[j])
       # encoded_data[j,-1] = torch.tensor(props[j])  # Adding Property Values
       # print("encoded_data shape:", encoded_data.shape)
        #print("props[j] shape:", props[j].shape)
        #encoded_data[j,-1] = torch.tensor(props[j])
        
        encoded_data[j,-1] = torch.tensor(props[j][0], dtype=torch.float)  # Convert the first attribute value to a tensor and add it to the last column of the encoded_data tensor
        encoded_data[j,-2] = torch.tensor(props[j][1], dtype=torch.float)  # Convert the second attribute value to a tensor and add it to the second-to-last column of the encoded_data tensor
        encoded_data[j,-3] = torch.tensor(props[j][2], dtype=torch.float)
        encoded_data[j,-4] = torch.tensor(props[j][3], dtype=torch.float)
        encoded_data[j,-5] = torch.tensor(props[j][4], dtype=torch.float)
        encoded_data[j,-6] = torch.tensor(props[j][5], dtype=torch.float)
        encoded_data[j,-7] = torch.tensor(props[j][6], dtype=torch.float)
        
        if props[j][7]=='nan':
            props[j][7]=0
        encoded_data[j,-8] = torch.tensor(props[j][7], dtype=torch.float)
        if props[j][8]=='nan':
            props[j][8]=0
            #encoded_data[j,-8] = torch.tensor(props[j][7], dtype=torch.float)
        encoded_data[j,-9] = torch.tensor(props[j][8], dtype=torch.float)
        #print(encoded_data)
    return encoded_data  # Returns the encoded data

def make_std_mask(tgt, pad):
    """
    Creates sequential mask matrix for target input (adapted from
    http://nlp.seas.harvard.edu/2018/04/03/attention.html)

    Arguments:
        tgt (torch.tensor, req): Target vector of token ids
        pad (int, req): Padding token id
    Returns:
        tgt_mask (torch.tensor): Sequential target mask
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask
