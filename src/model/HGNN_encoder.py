from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn
from model.graph_encoder import truncated_normal
torch.set_printoptions(precision=16)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class HGNNLayer(torch.nn.Module):
    def __init__(self,
                 adj,
                 input_dim,
                 output_dim,
                 bias=True,
                 act=None):
        super(HGNNLayer,self).__init__()
        self.bias = bias
        self.act = act
        self.adj = adj
        self.weight_mat = torch.nn.parameter.Parameter(torch.zeros([input_dim, output_dim]))
        self.weight_mat.data=truncated_normal(self.weight_mat.data,std=0.02)

        if bias:
            self.bias_vec = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.zeros([1, output_dim])),0.0)

    def forward(self, inputs, drop_rate=0.0):
        pre_sup_tangent = inputs
        if drop_rate > 0.0:
            pre_sup_tangent = torch.nn.Dropout(p=drop_rate)(pre_sup_tangent) * (1 - drop_rate)  # not scaled up
        output = torch.matmul(pre_sup_tangent, self.weight_mat)
        output = torch.sparse.mm(self.adj, output)
        if self.bias:
            bias_vec = self.bias_vec
            output = torch.add(output, bias_vec) 
        if self.act is not None:
            output = self.act(output)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class SpGAT(nn.Module):
    def __init__(self, adj,nfeat, nhid, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.adj=adj
        self.nheads=nheads

        self.attentions = SpGraphAttentionLayer(nfeat//self.nheads ,
                                                 nhid//self.nheads, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=False)

    def forward(self, x,y, adj=None):

        if adj==None:
            adj = self.adj

        xx=torch.chunk(x,self.nheads,1)
        yy=torch.chunk(y,self.nheads,1)
        output=list()
        for i in range(self.nheads):
            output.append(self.attentions(xx[i], yy[i], adj))
        x=torch.cat(output,dim=1)

        return x
        

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))####Wq
        truncated_normal(self.W.data, std=0.02)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        truncated_normal(self.a.data, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, y,adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        M = y.size()[0]
        edge =adj.coalesce().indices()

        h = torch.mm(input, self.W)
        hy=torch.mm(y, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], hy[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        edge_e = torch.where(torch.isnan(edge_e), torch.full_like(edge_e, 1e-8), edge_e)
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, M]), torch.ones(size=(M,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, M]), hy)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        h_prime=torch.where(torch.isnan(h_prime), torch.full_like(h_prime, 0), h_prime)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

'''

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b
    

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
'''
class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        a = torch.sparse_coo_tensor(indices, values, shape)
        return torch.sparse.mm(a, b)
        #return SpecialSpmmFunction.apply(indices, values, shape, b)
