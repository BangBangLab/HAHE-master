from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sre_parse
from sys import stdin
from sklearn.feature_selection import f_oneway
import torch
import torch.nn
from model.graph_encoder import encoder,truncated_normal
from model.HGNN_encoder import SpGAT
import numpy as np

import scipy.sparse as sp

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class HAHEModel(torch.nn.Module):
    def __init__(self,adj_info,config):
        super(HAHEModel,self).__init__()

        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._emb_size = config['hidden_size']
        self._intermediate_size = config['intermediate_size']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_dropout_prob']

        self._voc_size = config['vocab_size']
        self._hyperedge_num =config['hyperedge']

        self._n_relation = config['num_relations']
        self._n_edge = config['num_edges']
        self._max_seq_len = config['max_seq_len']
        self._max_arity = config['max_arity']
        self._e_soft_label = config['entity_soft_label']
        self._r_soft_label = config['relation_soft_label']


        self._device=config["device"]

        self._encoder_order = config['encoder_order']
        self._L_config = config['L_config']

        #self.node_embedding [V,N*H]
        self.node_embedding=torch.nn.Embedding(self._voc_size, self._emb_size)
        self.node_embedding.weight.data=truncated_normal(self.node_embedding.weight.data,std=0.02)
        self.hyperedge_embedding=torch.nn.Embedding(self._hyperedge_num, self._emb_size)
        self.hyperedge_embedding.weight.data=truncated_normal(self.hyperedge_embedding.weight.data,std=0.02)
        self.layer_norm1=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True) 

        # Global Hypergraph Embedding Learning

        self.global_layer_norm=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True) 
        self.global_hyperedge_layer_norm=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True) 

        self.adj_mat=torch.sparse_coo_tensor(adj_info[0]['indices'],adj_info[0]['values'],torch.Size(adj_info[0]['size']), dtype=torch.float32)
        self.adj_mat2=torch.sparse_coo_tensor(adj_info[1]['indices'],adj_info[1]['values'],torch.Size(adj_info[1]['size']), dtype=torch.float32)
        
        self.activation = torch.nn.Tanh()
        self.output = list()
        self._hgnn_head=4
        '''
        self.layer_num = config["HGNN_layer_num"]
        for i in range(self.layer_num):
            activation = self.activation
            if i == self.layer_num - 1:
                activation = None
            setattr(self,"gcn_layer{}".format(i),HGNNLayer(self.adj_mat, self._emb_size, self._emb_size, act=activation))
            setattr(self,"hgnn_layer_norm{}".format(i),torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True))
        '''
        self.layer_num = config["HGNN_layer_num"]
        for i in range(self.layer_num):
            setattr(self,"HyperGAT_layer{}".format(i),SpGAT(self.adj_mat,self._emb_size,self._emb_size,0.1,0.1,self._hgnn_head))
            setattr(self,"HyperGAT2_layer{}".format(i),SpGAT(self.adj_mat2,self._emb_size,self._emb_size,0.1,0.1,self._hgnn_head))
            setattr(self,"hgnn_layer_norm{}".format(i),torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True) )#orch.nn.BatchNorm1d(self._emb_size)



        self.batch_norm1=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True) #orch.nn.BatchNorm1d(self._emb_size)
        

        self.edge_embedding_q=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_q.weight.data=truncated_normal(self.edge_embedding_q.weight.data,std=0.02)
        self.edge_embedding_k=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_k.weight.data=truncated_normal(self.edge_embedding_k.weight.data,std=0.02)
        self.edge_embedding_v=torch.nn.Embedding(self._n_edge, self._emb_size // self._n_head)
        self.edge_embedding_v.weight.data=truncated_normal(self.edge_embedding_v.weight.data,std=0.02)
        self.encoder_model=encoder( 
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._intermediate_size,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            device=self._device,
            L_config=self._L_config)

        self.fc1=torch.nn.Linear(self._emb_size, self._emb_size)
        self.fc1.weight.data=truncated_normal(self.fc1.weight.data,std=0.02)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        self.layer_norm2=torch.nn.LayerNorm(normalized_shape=self._emb_size,eps=1e-12,elementwise_affine=True)
        self.fc2_bias = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.Tensor(self._voc_size)), 0.0)
        #softmax crossentropyloss with soft labels
        self.myloss = softmax_with_cross_entropy()
        
    def forward(self, input_ids, input_mask, edge_labels, mask_pos,mask_label, mask_type):

        if self._encoder_order == "GL":
            self._hypergraph_attention(input_ids)
            global_seq_embedding = self.node_embedding.weight[input_ids].squeeze()
            global_seq_embedding = torch.add(self.output[-1], global_seq_embedding)
        elif self._encoder_order == "true_GL":
            self._hypergraph_attention(input_ids)
            global_seq_embedding = self.output[-1]        
        elif self._encoder_order == "L":
            global_seq_embedding = self.node_embedding.weight[input_ids].squeeze()
        else:
            return


        emb_out = torch.nn.Dropout(self._prepostprocess_dropout)(self.layer_norm1(global_seq_embedding))

        # get edge embeddings between input tokens
        edges_query = self.edge_embedding_q(torch.squeeze(edge_labels))
        edges_key = self.edge_embedding_k(torch.squeeze(edge_labels))
        edges_value = self.edge_embedding_v(torch.squeeze(edge_labels))
        edge_mask = torch.sign(edge_labels) 
        edges_query = torch.mul(edges_query, edge_mask)
        edges_key = torch.mul(edges_key, edge_mask)
        edges_value = torch.mul(edges_value, edge_mask)
        # get multi-head self-attention mask
        self_attn_mask = torch.matmul(input_mask,input_mask.transpose(1,2))
        self_attn_mask=1000000.0*(self_attn_mask-1.0)
        n_head_self_attn_mask = torch.stack([self_attn_mask] * self._n_head, dim=1)###1024x4个相同的11x64个mask
        # stack of graph transformer encoders       
        _enc_out = self.encoder_model(
            enc_input=emb_out,
            edges_query=edges_query,
            edges_key=edges_key,
            edges_value=edges_key,
            attn_bias=n_head_self_attn_mask)       
        #Get the loss & logits for masked entity/relation prediction.
        mask_pos=mask_pos[:,:,None].expand(-1,-1,self._emb_size)
        h_masked=torch.gather(input=_enc_out, dim=1, index=mask_pos).squeeze()##若没有squeeze可预测多个
        # transform: fc1
        h_masked=torch.nn.GELU()(self.fc1(h_masked))
        # transform: layer norm
        h_masked=self.layer_norm2(h_masked)
        # transform: fc2 weight sharing
        fc_out=torch.nn.functional.linear(h_masked, self.node_embedding.weight, self.fc2_bias)
        #type_indicator [vocab_size,(yes1 or no0)]
        special_indicator = torch.empty(input_ids.size(0),2).to(self._device)
        torch.nn.init.constant_(special_indicator,-1)
        relation_indicator = torch.empty(input_ids.size(0), self._n_relation).to(self._device)
        torch.nn.init.constant_(relation_indicator,-1)
        entity_indicator = torch.empty(input_ids.size(0), (self._voc_size - self._n_relation - 2)).to(self._device)
        torch.nn.init.constant_(entity_indicator,1)              
        type_indicator = torch.cat((relation_indicator, entity_indicator), dim=1).to(self._device)
        type_indicator = torch.mul(type_indicator, mask_type)
        type_indicator = torch.cat([special_indicator, type_indicator], dim=1)
        type_indicator=torch.nn.functional.relu(type_indicator)
        fc_out_mask=1000000.0*(type_indicator-1.0)
        fc_out = torch.add(fc_out, fc_out_mask)        
        one_hot_labels = torch.nn.functional.one_hot(mask_label, self._voc_size)
        type_indicator = torch.sub(type_indicator, one_hot_labels)
        num_candidates = torch.sum(type_indicator, dim=1)
        #get soft label
        soft_labels = ((1 + mask_type) * self._e_soft_label +
                       (1 - mask_type) * self._r_soft_label) / 2.0
        soft_labels=soft_labels.expand(-1,self._voc_size)       
        soft_labels = soft_labels * one_hot_labels + (1.0 - soft_labels) * \
                      torch.mul(type_indicator, 1.0/torch.unsqueeze(num_candidates,1))       
        #get loss
        mean_mask_lm_loss = self.myloss(
              logits=fc_out, label=soft_labels)      
        """
        mean_mask_lm_loss=self.myloss(fc_out,mask_label)
        """
        return  mean_mask_lm_loss,fc_out



    def _hypergraph_attention(self, input_ids):

        output_embeddings=torch.nn.Dropout(self._prepostprocess_dropout)(self.global_layer_norm(self.node_embedding.weight))
        output_hyperedge_embeddings=torch.nn.Dropout(self._prepostprocess_dropout)(self.global_hyperedge_layer_norm(self.hyperedge_embedding.weight))

        self.output = list()  # reset

        # self.output[0]: layernorm and then dropout result of node_embedding
        self.output.append(output_embeddings[input_ids].squeeze())

        # get unique vocab id in batch
        unq_input_ids, inv_input_ids = torch.unique(input_ids, sorted=True, return_inverse=True)
        batch_output_count = unq_input_ids.shape[0]
        output_embeddings = output_embeddings[unq_input_ids]
        
        self.output.append(output_embeddings)


        # map unique vocab id to tensor id
        id_map = -torch.ones([self._voc_size], dtype=torch.int64)
        id_range = torch.arange(start=0, end=batch_output_count, dtype=torch.int64)
        id_map[unq_input_ids[id_range]] = id_range

        # get unique hyperedge id and hyperedge embeddings
        adj2_ids = self.adj_mat2._indices()
        hyperedge_start_ids = adj2_ids[0]
        hyperedge_end_ids = adj2_ids[1]
        hyperedge_in_batch = np.isin(hyperedge_start_ids.cpu(), unq_input_ids.cpu())
        batch_hyperedge_ids = hyperedge_end_ids[hyperedge_in_batch]
        unq_hyperdege_ids = torch.unique(batch_hyperedge_ids, sorted=True, return_inverse=False)
        batch_hyperedge_count = unq_hyperdege_ids.shape[0]        
        output_hyperedge_embeddings = output_hyperedge_embeddings[unq_hyperdege_ids]
        
        # map unique hyperedge id to tensor id
        hid_map = -torch.ones([self._hyperedge_num], dtype=torch.int64)
        hid_range = torch.arange(start=0, end=batch_hyperedge_count, dtype=torch.int64)
        hid_map[unq_hyperdege_ids[hid_range]] = hid_range

        # construct batch adjacent matrix
        batch_hyperedge_start_ids = id_map[hyperedge_start_ids[hyperedge_in_batch]]
        batch_hyperedge_end_ids = hid_map[hyperedge_end_ids[hyperedge_in_batch]]
        batch_adj_ids = torch.vstack([batch_hyperedge_end_ids, batch_hyperedge_start_ids])
        batch_adj2_ids = torch.vstack([batch_hyperedge_start_ids, batch_hyperedge_end_ids])
        batch_adj12_vals = torch.ones(batch_hyperedge_start_ids.shape)
        batch_adj_shape = torch.Size([batch_hyperedge_count, batch_output_count])
        batch_adj2_shape = torch.Size([batch_output_count, batch_hyperedge_count])
        batch_adj = torch.sparse_coo_tensor(batch_adj_ids, batch_adj12_vals, batch_adj_shape).to(self._device)
        batch_adj2 = torch.sparse_coo_tensor(batch_adj2_ids, batch_adj12_vals, batch_adj2_shape).to(self._device)



        for i in range(self.layer_num):
            output_hyperedge_embeddings = getattr(self,"HyperGAT_layer{}".format(i))(output_hyperedge_embeddings,output_embeddings, batch_adj)
            output_embeddings = getattr(self,"HyperGAT2_layer{}".format(i))(output_embeddings,output_hyperedge_embeddings, batch_adj2)
            if i!=(self.layer_num-1):
                output_embeddings=torch.nn.functional.elu(output_embeddings)
            # output_embeddings = output_embeddings[inv_input_ids]
            output_embeddings = getattr(self,"hgnn_layer_norm{}".format(i))(torch.add(torch.nn.Dropout(self._prepostprocess_dropout)(output_embeddings), self.output[-1]))
            # self.output[-1]: updated unique embedding
            self.output.append(output_embeddings)
        
        inc_embeddings = self.output[-1][inv_input_ids].squeeze()
        self.output.append(torch.add(inc_embeddings, self.output[0]))
        
    

class softmax_with_cross_entropy(torch.nn.Module):
    def __init__(self):
        super(softmax_with_cross_entropy,self).__init__()

    def forward(self,logits, label):
        logprobs=torch.nn.functional.log_softmax(logits,dim=1)
        loss=-1.0*torch.sum(torch.mul(label,logprobs),dim=1).squeeze()
        loss=torch.mean(loss)
        return loss
