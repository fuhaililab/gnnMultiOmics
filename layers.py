"""
Jiarui Feng
definition of the GNN layers
"""

from torch_geometric.utils import add_self_loops, degree, softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import  deepcopy as c
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros,kaiming_uniform
from torch_geometric.utils import softmax
from torch_scatter.composite import scatter_softmax

def clones( module, N):
    """Layer clone function, used for concise code writing
    Args:
        module: the layer want to clone
        N: the time of clone
    """
    return nn.ModuleList(c(module) for _ in range(N))



class GCNLayer(MessagePassing):
    """
    Graph convolution layer with edge attribute
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute
    """
    def __init__(self,input_size,output_size,aggr="add",num_edge_type=0):
        super(GCNLayer, self).__init__()
        self.aggr=aggr
        self.proj=nn.Linear(input_size,output_size,bias=False)
        self.bias=nn.Parameter(torch.Tensor(output_size))
        if num_edge_type>0:
            #0 for padding index, therefore plus 1
            self.edge_embedding = torch.nn.Embedding(num_edge_type+1, output_size,padding_idx=0)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight.data)
        zeros(self.bias)

    def forward(self,x,edge_index,edge_attr=None):
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        x = self.proj(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if edge_attr is not None:
            #add features corresponding to self-loop edges, set as zeros.
            self_loop_attr = torch.zeros(x.size(0),dtype=torch.long)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

            edge_embeddings = self.edge_embedding(edge_attr)

            return self.propagate(edge_index, x=x, norm=norm,edge_attr=edge_embeddings)
        else:
            return self.propagate(edge_index, x=x,norm=norm, edge_attr=None)

    def message(self, x_j,edge_attr,norm):
        if edge_attr is not None:
            return norm.view(-1,1)*(x_j+edge_attr)
        else:
            return norm.view(-1,1)*x_j

    def update(self,aggr_out):
        return F.relu(aggr_out)



# GAT torch_geometric implementation
#Adapted from https://github.com/snap-stanford/pretrain-gnns and
#https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv
class GATLayer(MessagePassing):
    """Graph attention layer with edge attribute
    Args:
        input_size(int): the size of input feature
        output_size(int): the size of output feature
        head(int): the number of head in multi-head attention
        negative_slope(float): the slope in leaky relu function
        aggr(str): aggregation function in message passing network
        num_edge_type(int): number of edge type, 0 indicate no edge attribute.

    """
    def __init__(self, input_size,output_size,head, negative_slope=0.2, aggr = "add",num_edge_type=0):
        super(GATLayer, self).__init__(node_dim=0)
        assert output_size%head==0
        self.k=output_size//head
        self.aggr = aggr

        self.output_size = output_size
        self.head = head
        self.negative_slope = negative_slope

        self.proj = nn.Linear(input_size, output_size,bias=False)
        self.att_node = torch.nn.Parameter(torch.Tensor(1, head,  self.k))
        self.att_edge=torch.nn.Parameter(torch.Tensor(1,head,self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))

        if num_edge_type>0:
            #0 for padding index, therefore plus 1
            self.edge_embedding = torch.nn.Embedding(num_edge_type+1, output_size,padding_idx=0)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight.data)
        glorot(self.att_node) #glorot: commonly used in NN weight initialization
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index,edge_attr=None):

        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        #projection
        x = self.proj(x).view(-1, self.head, self.k) # N * head * k

        # Next, we compute node-level attention coefficients
        alpha=(x*self.att_node).sum(dim=-1) # N * head
        if edge_attr is not None:
            #add features corresponding to self-loop edges, set as zeros.
            self_loop_attr = torch.zeros(x.size(0),dtype=torch.long)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
            alpha=self.edge_updater(edge_index,alpha=alpha,edge_attr=edge_attr)# E * head
        else:
            alpha=self.edge_updater(edge_index,alpha=alpha,edge_attr=None)
        return self.propagate(edge_index, x=x, alpha=alpha),alpha


    def message(self, x_j, alpha) :
        return alpha.unsqueeze(-1) * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1,self.output_size)
        aggr_out = aggr_out + self.bias

        return F.relu(aggr_out)

    def edge_update(self,edge_index, alpha_j,alpha_i,edge_attr):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_embedding = self.edge_embedding(edge_attr)
            edge_embedding = edge_embedding.view(-1, self.heads, self.k)
            alpha_edge = (edge_embedding * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return alpha


class GNNLayer(MessagePassing):
    def __init__(self, hidden_size, K, head, num_encoding_type=0, num_edge_type=0, aggr="add"):
        """Proposed GNN layer
        Args:
            hidden_size: the hidden size of the model
            K: the max number of hop to consider
            head: number of head in the attention
            num_encoding_type: number of encoding type. 0 means no encoding module
            num_edge_type: number of edge type. 0 means no edge feature
            aggr: aggregation type
        """
        super(GNNLayer, self).__init__(node_dim=0)
        assert hidden_size % K == 0
        self.dk = hidden_size // K
        self.K = K  # number of hop
        self.head = head
        self.k = self.dk // head  # dimension of each head in each hop
        self.hidden_size = hidden_size
        self.aggr = aggr
        self.num_encoding_type = num_encoding_type
        self.num_edge_type = num_edge_type
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        if self.num_encoding_type > 0:
            # 0 for padding index
            self.key_path_embedding = nn.Embedding(self.num_encoding_type + 1, self.dk, padding_idx=0)
            self.query_path_embedding = nn.Embedding(self.num_encoding_type + 1, self.dk, padding_idx=0)
            self.value_path_embedding = nn.Embedding(self.num_encoding_type + 1, self.dk, padding_idx=0)
            nn.init.xavier_uniform_(self.key_path_embedding.weight.data)
            nn.init.xavier_uniform_(self.query_path_embedding.weight.data)
            nn.init.xavier_uniform_(self.value_path_embedding.weight.data)

            self.key_node_embedding = nn.Embedding(self.num_encoding_type + 1, self.dk, padding_idx=0)
            self.query_node_embedding = nn.Embedding(self.num_encoding_type + 1, self.dk, padding_idx=0)
            self.value_node_embedding = nn.Embedding(self.num_encoding_type + 1, self.dk, padding_idx=0)
            nn.init.xavier_uniform_(self.key_node_embedding.weight.data)
            nn.init.xavier_uniform_(self.query_node_embedding.weight.data)
            nn.init.xavier_uniform_(self.value_node_embedding.weight.data)

        if num_edge_type > 0:
            # embedding start from 1
            self.kee = nn.Embedding(num_edge_type + 1, self.dk, padding_idx=0)
            self.qee = nn.Embedding(num_edge_type + 1, self.dk, padding_idx=0)
            self.vee = nn.Embedding(num_edge_type + 1, self.dk, padding_idx=0)
            nn.init.xavier_uniform_(self.kee.weight.data)
            nn.init.xavier_uniform_(self.qee.weight.data)
            nn.init.xavier_uniform_(self.vee.weight.data)

        self.att = torch.nn.Parameter(torch.rand([1, self.K, head, 2 * self.k]))
        self.out_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.eps = nn.Parameter(torch.tensor([0.]))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight.data)
        zeros(self.bias)

    def forward(self, x, edge_index, path_attr_list=None, node_attr_list=None, edge_attr=None):
        # print((torch.isnan(x)).sum())
        batch_num_node = x.size(0)
        # transformation
        k = self.key_proj(x).view(batch_num_node, self.K, self.dk)  # N * K * dk
        q = self.query_proj(x).view(batch_num_node, self.K, self.dk)  # N * K * dk
        v = self.value_proj(x).view(batch_num_node, self.K, self.dk)  # N * K * dk



        if (edge_attr is not None) and (self.num_edge_type > 0):
            kee = self.kee(edge_attr).unsqueeze(0)  # 1 * E * dk
            qee = self.qee(edge_attr).unsqueeze(0)  # 1 * E * dk
            vee = self.vee(edge_attr).unsqueeze(0)  # 1 * E * dk
        else:
            kee = torch.zeros([1, edge_index.size(-1), self.dk], device=edge_index.device)
            qee = torch.zeros([1, edge_index.size(-1), self.dk], device=edge_index.device)
            vee = torch.zeros([1, edge_index.size(-1), self.dk], device=edge_index.device)

        if (path_attr_list is not None) and (self.num_encoding_type > 0):
            kpes = self.key_path_embedding(path_attr_list)  # E * k
            qpes = self.query_path_embedding(path_attr_list)
            vpes = self.value_path_embedding(path_attr_list)
            knes = self.key_node_embedding(node_attr_list)
            qnes = self.query_node_embedding(node_attr_list)
            vnes = self.value_node_embedding(node_attr_list)
        else:
            kpes = torch.cat([torch.zeros_like(kee) for _ in range(self.K - 1)], dim=0)
            qpes = torch.cat([torch.zeros_like(qee) for _ in range(self.K - 1)], dim=0)
            vpes = torch.cat([torch.zeros_like(vee) for _ in range(self.K - 1)], dim=0)
            knes = torch.zeros([self.K - 1, batch_num_node, self.dk], device=x.device)
            qnes = torch.zeros([self.K - 1, batch_num_node, self.dk], device=x.device)
            vnes = torch.zeros([self.K - 1, batch_num_node, self.dk], device=x.device)
        kpes = torch.cat([kee, kpes], dim=0).transpose(0, 1).contiguous()  # E * K * dk
        qpes = torch.cat([qee, qpes], dim=0).transpose(0, 1).contiguous()  # E * K * dk
        vpes = torch.cat([vee, vpes], dim=0).transpose(0, 1).contiguous()  # E * K * dk
        knes = torch.cat([torch.zeros_like(knes[0]).unsqueeze(0), knes], dim=0).transpose(0,
                                                                                          1).contiguous()  # N * K * dk
        qnes = torch.cat([torch.zeros_like(qnes[0]).unsqueeze(0), qnes], dim=0).transpose(0,
                                                                                          1).contiguous()  # N * K * dk
        vnes = torch.cat([torch.zeros_like(vnes[0]).unsqueeze(0), vnes], dim=0).transpose(0,
                                                                                          1).contiguous()  # N * K * dk
        k = k * (1 + knes)
        q = q * (1 + qnes)
        v = v * (1 + vnes)


        mask = path_attr_list # K * E
        # mask = mask.transpose(0, 1).contiguous()  # E * K
        v_n,alpha= self.edge_updater(edge_index.long(), k=k, q=q, v=v, kpes=kpes, qpes=qpes, vpes=vpes, mask=mask)  # N * H
        x_n = self.propagate(edge_index.long(), x=v_n,alpha=alpha, index=edge_index[0])
        x_n = self.out_proj(x_n + (1 + self.eps) * x)

        return x_n,alpha

    def message(self, x, alpha) :
        return  x * alpha.unsqueeze(-1)

    def edge_update(self, edge_index, k_i, q_j, v_j, kpes, qpes, vpes, mask):

        k_i = k_i * (1 + kpes)  # E* k * dk
        q_j = q_j * (1 + qpes)  # E* k * dk
        v_j = v_j + vpes  # E* k * dk

        E = v_j.size(0)
        v_j = v_j.view(E, self.K, self.head, -1)  # E * K * h * k
        k_i = k_i.view(E, self.K, self.head, -1)  # E * K * h * k
        q_j = q_j.view(E, self.K, self.head, -1)  # E * K * h * k
        alpha = F.leaky_relu(torch.cat([k_i, q_j], dim=-1), 0.2)  # E * K * head * 2k
        alpha = torch.sum(self.att * alpha, dim=-1)  # E * K * head
        mask = mask.unsqueeze(-1)
        alpha.masked_fill_(mask == 0, -1e30)
        alpha = scatter_softmax(alpha, edge_index[0], dim=self.node_dim)  # E * K * head
        # v_j = v_j * alpha.unsqueeze(-1)  # E * K * head * k
        return v_j,alpha

    def update(self, aggr_out):
        return aggr_out.view(-1, self.hidden_size)




