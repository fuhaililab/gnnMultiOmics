"""
Jiarui Feng
construct GNN model given gnn layer
Modified by Zitian Tang 02/2023
"""

import torch
import torch
import torch.nn as nn
from layers import GCNLayer,GATLayer,clones, GNNLayer
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set,global_sort_pool
# from RFpooling import GlobalAttention
from torch_geometric.nn import BatchNorm,LayerNorm,InstanceNorm,PairNorm,GraphSizeNorm
from torch_geometric.utils import dropout_adj
from math import sqrt
import networkx as nx
import pandas as pd


class GNN(nn.Module):
    """A generalized GNN framework
    Args:
        input_size(int): the size of input feature
        hidden_size(int): the size of output feature
        num_layer(int): the number of GNN layer
        gnn_layer(nn.Module): gnn layer used in GNN model
        JK(str):method of jumping knowledge, last,concat,max or sum
        norm_type(str): method of normalization, batch or layer
        drop_prob (float): dropout rate
    """
    def __init__(self,input_size,hidden_size,num_layer,gnn_layer,JK="last",norm_type="batch",
                        edge_drop_prob=0.1,drop_prob=0.1):
        super(GNN, self).__init__()
        self.num_layer=num_layer
        self.hidden_size=hidden_size
        self.dropout=nn.Dropout(drop_prob)
        self.edge_drop_flag=edge_drop_prob>0
        self.edge_drop_prob=edge_drop_prob
        self.JK=JK

        if self.JK=="attention":
            self.attention_lstm=nn.LSTM(hidden_size,self.num_layer,1,batch_first=True,bidirectional=True,dropout=0.)
            for layer_p in self.attention_lstm._all_weights:
                for p in layer_p:
                    if 'weight' in p:
                        nn.init.xavier_uniform_(self.attention_lstm.__getattr__(p))

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.init_proj=nn.Linear(input_size,hidden_size)
        nn.init.xavier_uniform_(self.init_proj.weight.data)

        self.add_geneid = nn.Embedding(5000, hidden_size)

        #gnn layer list
        self.gnns=clones(gnn_layer,num_layer)
        #norm list
        if norm_type=="Batch":
            self.norms=clones(BatchNorm(hidden_size),num_layer)
        elif norm_type=="Layer":
            self.norms=clones(LayerNorm(hidden_size),num_layer)
        elif norm_type=="Instance":
            self.norms=clones(InstanceNorm(hidden_size),num_layer)
        elif norm_type=="GraphSize":
            self.norms=clones(GraphSizeNorm(),num_layer)
        elif norm_type=="Pair":
            self.norms=clones(PairNorm(),num_layer)
        else:
            raise ValueError("Not supported norm method")

    def forward(self, x,edge_index,path_attr,node_attr,edge_attr,ids):

        #initial projection
        x=self.init_proj(x) + self.add_geneid(ids)
        #initialize attention
        attention=None
        #forward in gnn layer
        h_list=[x]
        for l in range(self.num_layer):
            if self.edge_drop_flag and self.training:
                edge_index,edge_attr=dropout_adj(edge_index,edge_attr,p=self.edge_drop_prob)
            h=self.gnns[l](h_list[l],edge_index,path_attr,node_attr,edge_attr)

            #For GAT layer, it will return both embedding and attention weight
            #we only keep attention at the last layer
            if len(h)==2:
                h,attention=h
            #print((torch.isnan(h)).sum())
            h=self.norms[l](h)
            #if not the last gnn layer, add dropout layer
            if l!=self.num_layer-1:
                h=self.dropout(h)

            h_list.append(h)

        #JK connection
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze(-1) for h in h_list]
            node_representation = F.max_pool1d(torch.cat(h_list, dim = -1),kernel_size=self.num_layer+1).squeeze()
        elif self.JK == "sum":
            h_list = [h.unsqueeze(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)
        elif self.JK=="attention":
            h_list = [h.unsqueeze(0) for h in h_list]
            h_list=torch.cat(h_list, dim = 0).transpose(0,1) # N *num_layer * H
            self.attention_lstm.flatten_parameters()
            attention_score,_=self.attention_lstm(h_list) # N * num_layer * 2
            attention_score=torch.softmax(torch.sum(attention_score,dim=-1),dim=1).unsqueeze(-1) #N * num_layer  * 1
            node_representation=torch.sum(h_list*attention_score,dim=1)
        if attention is None:
            return node_representation
        else:
            return node_representation,attention


def make_gnn_layer(gnn_type,hidden_size,head=None,negative_slope=0.2,num_edge_type=0):
    """function to construct gnn layer
    Args:
        gnn_type(str):
    """
    if gnn_type=="GCN":
        return GCNLayer(hidden_size,hidden_size,"add",num_edge_type)
    elif gnn_type=="GAT":
        return GATLayer(hidden_size,hidden_size,head,negative_slope,"add",num_edge_type)
    elif gnn_type == "GNN":
        return GNNLayer(hidden_size,2,head,num_edge_type=num_edge_type,aggr="add")
    else:
        raise ValueError("Not supported GNN type")


class GraphRegression(nn.Module):
    """model for graph regression
    Args:
        embedding_model(nn.Module): node embedding model
        pooling_method(nn.Module): graph pooling method
    """
    def __init__(self,embedding_model,pooling_method="mean"):
        super(GraphRegression, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_size=embedding_model.hidden_size
        self.JK=embedding_model.JK
        self.num_gnn_layer=embedding_model.num_layer
        self.pooling_method=pooling_method

        # Different kind of graph pooling
        if pooling_method == "sum":
            self.pool = global_add_pool
        elif pooling_method == "mean":
            self.pool = global_mean_pool
        elif pooling_method == "max":
            self.pool = global_max_pool
        # elif pooling_method == "attention":
        #     if self.JK == "concat":
        #         self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_gnn_layer + 1) * self.hidden_size, 1))
        #
        #     else:
        #         self.pool = GlobalAttention(gate_nn = torch.nn.Linear(self.hidden_size, 1))

        # regressor
        if self.JK == "concat":
            self.regressor = nn.Sequential(nn.Linear((self.num_gnn_layer + 1) * self.hidde_size, self.hidden_size),
                                           nn.ELU(),
                                           nn.Linear(self.hidden_size, 2))
        else:
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                           nn.ELU(),
                                           nn.Linear(self.hidden_size, 2))

    def forward(self, data):

        x, edge_index,batch,mask,ids = data.x, data.edge_index, data.batch, data.mask, data.ids

        attention = None
        node_attr = None
        path_attr = mask
        edge_attr = None
        # node representation
        x=self.embedding_model(x,edge_index,path_attr,node_attr,edge_attr,ids)

        if len(x)==2:
            x, attention = x

        pool_x=self.pool(x,batch)

        # t_np = batch.cpu().numpy()  # convert to Numpy array
        # df = pd.DataFrame(t_np)  # convert to a dataframe
        # df.to_csv("testfile.csv", index=False)  # save to file
        out = self.regressor(pool_x).squeeze()
        if attention is not None:
            return out, attention
        else:
            return self.regressor(pool_x).squeeze()

