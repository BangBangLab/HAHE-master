"""
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# yapf: disable
from __future__ import print_function
from __future__ import division

import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import json
import numpy as np
import collections
import logging
import time
import scipy.sparse as sp
import copy

from reader.vocab_reader import Vocabulary

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())


class NaryExample(object):
    """
    A single training/test example of n-ary fact.
    """
    def __init__(self,
                 arity,
                 relation,
                 head,
                 tail,
                 auxiliary_info=None):
        """
        Construct NaryExample.

        Args:
            arity (mandatory): arity of a given fact
            relation (mandatory): primary relation
            head (mandatory): primary head entity (subject)
            tail (mandatory): primary tail entity (object)
            auxiliary_info (optional): auxiliary attribute-value pairs,
                with attributes and values sorted in alphabetical order
        """
        self.arity = arity
        self.relation = relation
        self.head = head
        self.tail = tail
        self.auxiliary_info = auxiliary_info


class NaryFeature(object):
    """
    A single set of features used for training/test.
    """
    def __init__(self,
                 feature_id,
                 example_id,
                 input_tokens,
                 input_ids,
                 input_mask,
                 mask_position,
                 mask_label,
                 mask_type,
                 arity):
        """
        Construct NaryFeature.

        Args:
            feature_id: unique feature id
            example_id: corresponding example id
            input_tokens: input sequence of tokens
            input_ids: input sequence of ids
            input_mask: input sequence mask
            mask_position: position of masked token
            mask_label: label of masked token
            mask_type: type of masked token,
                1 for entities (values) and -1 for relations (attributes)
            arity: arity of the corresponding example
        """
        self.feature_id = feature_id
        self.example_id = example_id
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.mask_position = mask_position
        self.mask_label = mask_label
        self.mask_type = mask_type
        self.arity = arity


def read_examples(input_file):
    """
    Read a n-ary json file into a list of NaryExample.
    """
    examples, total_instance = [], 0
    with open(input_file, "r") as fr:
        for line in fr.readlines():
            obj = json.loads(line.strip())
            assert "N" in obj.keys() \
                   and "relation" in obj.keys() \
                   and "subject" in obj.keys() \
                   and "object" in obj.keys(), \
                "There are 4 mandatory fields: N, relation, subject, and object."
            arity = obj["N"]
            relation = obj["relation"]
            head = obj["subject"]
            tail = obj["object"]

            auxiliary_info = None
            if arity > 2:
                auxiliary_info = collections.OrderedDict()
                # store attributes in alphabetical order
                for attribute in sorted(obj.keys()):
                    if attribute == "N" \
                            or attribute == "relation" \
                            or attribute == "subject" \
                            or attribute == "object":
                        continue
                    # store corresponding values in alphabetical order
                    auxiliary_info[attribute] = sorted(obj[attribute])
            """
            if len(examples) % 1000 == 0:
                logger.debug("*** Example ***")
                logger.debug("arity: %s" % str(arity))
                logger.debug("relation: %s" % relation)
                logger.debug("head: %s" % head)
                logger.debug("tail: %s" % tail)
                if auxiliary_info:
                    for attribute in auxiliary_info.keys():
                        logger.debug("attribute: %s" % attribute)
                        logger.debug("value(s): %s" % " ".join(
                            [value for value in auxiliary_info[attribute]]))
            """

            example = NaryExample(
                arity=arity,
                relation=relation,
                head=head,
                tail=tail,
                auxiliary_info=auxiliary_info)
            examples.append(example)
            total_instance += (2 * (arity - 2) + 3)

    return examples, total_instance


def convert_examples_to_features(examples, vocabulary, max_arity, max_seq_length):
    """
    Convert a set of NaryExample into a set of NaryFeature. Each single
    NaryExample is converted into (2*(n-2)+3) NaryFeature, where n is
    the arity of the given example.
    """
    max_aux = max_arity - 2
    assert max_seq_length == 2 * max_aux + 3, \
        "Each input sequence contains relation, head, tail, " \
        "and max_aux attribute-value pairs."

    features = []
    feature_id = 0
    nomask=list()
    for (example_id, example) in enumerate(examples):
        # get original input tokens and input mask
        hrt = [ example.head,example.relation, example.tail]
        hrt_mask = [1, 1, 1]

        aux_av = []
        aux_av_mask = []

        if example.auxiliary_info is not None:
            for attribute in example.auxiliary_info.keys():
                for value in example.auxiliary_info[attribute]:
                    aux_av.append(attribute)
                    aux_av.append(value)
                    aux_av_mask.append(1)
                    aux_av_mask.append(1)

        while len(aux_av) < (max_aux*2):
            aux_av.append("[PAD]")
            aux_av.append("[PAD]")
            aux_av_mask.append(0)
            aux_av_mask.append(0)
        assert len(aux_av) == max_aux*2
        assert len(aux_av_mask) == max_aux*2

        orig_input_tokens = hrt + aux_av
        orig_input_mask = hrt_mask + aux_av_mask
        assert len(orig_input_tokens) == max_seq_length
        assert len(orig_input_mask) == max_seq_length
        nomask.append(vocabulary.convert_tokens_to_ids(orig_input_tokens))

        # generate a feature by masking each of the tokens
        for mask_position in range(max_seq_length):
            if orig_input_tokens[mask_position] == "[PAD]":
                continue
            mask_label = vocabulary.vocab[orig_input_tokens[mask_position]]
            mask_type = 1 if mask_position %2==0  else -1

            input_tokens = orig_input_tokens[:]
            input_tokens[mask_position] = "[MASK]"
            input_ids = vocabulary.convert_tokens_to_ids(input_tokens)
            assert len(input_tokens) == max_seq_length
            assert len(input_ids) == max_seq_length

            feature = NaryFeature(
                feature_id=feature_id,
                example_id=example_id,
                input_tokens=input_tokens,
                input_ids=input_ids,
                input_mask=orig_input_mask,
                mask_position=mask_position,
                mask_label=mask_label,
                mask_type=mask_type,
                arity=example.arity)
            features.append(feature)
            feature_id += 1

    return features,nomask


class MultiDataset(Dataset.Dataset):
    def __init__(self, vocabulary:Vocabulary,examples,max_arity=2,max_seq_length=3,shuffle=True):
        self.examples=examples
        self.vocabulary=vocabulary
        self.max_arity=max_arity
        self.max_seq_length=max_seq_length
        self.shuffle=shuffle
        if self.shuffle is True:
            np.random.shuffle(self.examples)
        
        self.features ,self.nomask= convert_examples_to_features(
            examples=self.examples,
            vocabulary=self.vocabulary,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length)
        self.multidataset=[]
        for (index, feature) in enumerate(self.features):
            input_ids = feature.input_ids
            input_mask = feature.input_mask
            mask_position = feature.mask_position
            mask_label = feature.mask_label
            mask_type = feature.mask_type
            feature_out = [input_ids] + [input_mask] + \
                            [mask_position] + [mask_label] + [mask_type]
            self.multidataset.append(feature_out)
        """       
        self.edge_labels=prepare_edge_labels(            
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length)
        """
   
    def __len__(self):
        return len(self.multidataset)

    def __getitem__(self,index):        
        x=self.multidataset[index]
        x=[x]
        batch_input_ids,\
            batch_input_mask,\
            batch_mask_position,\
            batch_mask_label,\
            batch_mask_type,\
            batch_edge_labels=prepare_batch_data(
            x,
            max_arity=self.max_arity,
            max_seq_length=self.max_seq_length)

        return batch_input_ids,batch_input_mask,batch_mask_position,batch_mask_label,batch_mask_type,batch_edge_labels


def prepare_batch_data(insts, max_arity, max_seq_length):
    """
    Format batch input for training/test. Output a list of six entries:
        return_list[0]: batch_input_ids (batch_size * max_seq_length * 1)
        return_list[1]: batch_input_mask (batch_size * max_seq_length * 1)
        return_list[2]: batch_mask_position (batch_size * 1)
        return_list[3]: batch_mask_label (batch_size * 1)
        return_list[4]: batch_mask_type (batch_size * 1)
        return_list[5]: edge_labels (max_seq_length * max_seq_length * 1)
    Note: mask_position indicates positions in a batch (not individual instances).
    And edge_labels are shared across all instances in the batch.
    """
    batch_input_ids = np.array([
        inst[0] for inst in insts
    ]).astype("int64").reshape([max_seq_length, 1])
    batch_input_mask = np.array([
        inst[1] for inst in insts
    ]).astype("float32").reshape([max_seq_length, 1])

    batch_mask_position = np.array([
        idx * max_seq_length + inst[2] for (idx, inst) in enumerate(insts)
    ]).astype("int64").reshape([1])
    batch_mask_label = np.array(
        [inst[3] for inst in insts]).astype("int64").reshape([1])
    batch_mask_type = np.array(
        [inst[4] for inst in insts]).astype("int64").reshape([1])

    # edge labels between input nodes (used for GRAN-hete)
    #     0: no edge
    #     1: relation-subject
    #     2: relation-object
    #     3: relation-attribute
    #     4: attribute-value
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 2] + [3,4] * max_aux )
    edge_labels.append([1, 0, 5] + [6,7] * max_aux )
    edge_labels.append([2, 5, 0] + [8,9] * max_aux )
    for idx in range(max_aux):
        edge_labels.append([3,6,8] + [11,12] * idx + [0,10] + [11,12] * (max_aux - idx - 1))
        edge_labels.append([4,7,9] + [12,13] * idx + [10,0] + [12,13] * (max_aux - idx - 1))
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])  
    """
    # unlabeled edges between input nodes (used for GRAN-homo)
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 1] + [1] * max_aux + [0] * max_aux)
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [1, 0, 0] + [0] * max_aux + [0] * idx + [1] + [0] * (max_aux - idx - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [0, 0, 0] + [0] * idx + [1] + [0] * (max_aux - idx - 1) + [0] * max_aux)
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    """
    """
    # no edges between input nodes (used for GRAN-complete)
    edge_labels = [[0] * max_seq_length] * max_seq_length
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    """   


    return batch_input_ids,batch_input_mask,batch_mask_position,batch_mask_label,batch_mask_type,edge_labels
"""
def prepare_edge_labels(max_arity, max_seq_length):
        # edge labels between input nodes (used for GRAN-hete)
    #     0: no edge
    #     1: relation-subject
    #     2: relation-object
    #     3: relation-attribute
    #     4: attribute-value
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 2] + [3] * max_aux + [0] * max_aux)
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    edge_labels.append([2] + [0] * (max_seq_length - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [3, 0, 0] + [0] * max_aux + [0] * idx + [4] + [0] * (max_aux - idx - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [0, 0, 0] + [0] * idx + [4] + [0] * (max_aux - idx - 1) + [0] * max_aux)
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    
    # unlabeled edges between input nodes (used for GRAN-homo)
    edge_labels = []
    max_aux = max_arity - 2
    edge_labels.append([0, 1, 1] + [1] * max_aux + [0] * max_aux)
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    edge_labels.append([1] + [0] * (max_seq_length - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [1, 0, 0] + [0] * max_aux + [0] * idx + [1] + [0] * (max_aux - idx - 1))
    for idx in range(max_aux):
        edge_labels.append(
            [0, 0, 0] + [0] * idx + [1] + [0] * (max_aux - idx - 1) + [0] * max_aux)
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    
    
    # no edges between input nodes (used for GRAN-complete)
    edge_labels = [[0] * max_seq_length] * max_seq_length
    edge_labels = np.asarray(edge_labels).astype("int64").reshape(
        [max_seq_length, max_seq_length, 1])
    
    return edge_labels
"""

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
        return sparse_mx

def gen_hadj(total_ent_num, statements):
    start = time.time()
    total_state_num=len(statements)
    row = list()
    col = list()
    e=list()

    for j,item in enumerate(statements):
        for p,i in enumerate(item):
            # if i!=0:
            if i!=0 and p%2==0:
                col.append(i)
                row.append(j)

    data_len = len(row)
    data = np.ones(data_len)
    H = sp.coo_matrix((data, (row, col)), shape=(total_state_num,total_ent_num )) 
    H2 = sp.coo_matrix((data, (col, row)), shape=(total_ent_num ,total_state_num)) 
    '''
    for j,item in enumerate(statements):
        for p,i in enumerate(item):
            if i!=0:
                row.append(i)
                col.append(j)

    data_len = len(row)
    data = np.ones(data_len)
    H = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_state_num))  
    
    n_edge = H.shape[1]# 超边矩阵
    # the weight of the hyperedge
    W = np.ones(n_edge) # 超边权重矩阵
    # the degree of the node
    DV = np.array(H.sum(1))  # 节点度; (12311,)
    # the degree of the hyperedge
    DE = np.array(H.sum(0))  # 超边的度; (24622,)
    
    invDE = sp.diags(np.power(DE, -1).flatten())  # DE^-1; 建立对角阵
    DV2 = sp.diags(np.power(DV, -0.5).flatten())  # DV^-1/2
    W = sp.diags(W)  # 超边权重矩阵
    HT = H.transpose()

    G = DV2 * H * W * invDE * HT * DV2
    '''
    logger.info('generating G costs time: {:.4f}s'.format(time.time() - start))    
    return sparse_to_tuple(H),sparse_to_tuple(H2)

def prepare_adj_info(info,  device):
    adj,adj2= gen_hadj(len(info.vocabulary.vocab), info.nomask)
    adj_info=dict()
    adj_info['indices']=torch.tensor(adj[0]).t().to(device)
    adj_info['values']=torch.tensor(adj[1]).to(device)
    adj_info['size']=torch.tensor(adj[2]).to(device)
    adj_info2=dict()
    adj_info2['indices']=torch.tensor(adj2[0]).t().to(device)
    adj_info2['values']=torch.tensor(adj2[1]).to(device)
    adj_info2['size']=torch.tensor(adj2[2]).to(device)

    return [adj_info,adj_info2]