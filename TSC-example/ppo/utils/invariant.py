import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import Attention, PreNorm, Transformer, CrossAttention, FeedForward, PreNormDouble, CrossAttention_Querydimchanged
from einops import rearrange
import random

class Invariant(nn.Module):
    def __init__(self, input_dim, invariant_type = "attn_sum", hidden_dim=128, heads=4, dim_head=32, mlp_dim=128, dropout=0., depth=1, distance=0, neighbor_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.invariant_type = invariant_type
        self.depth = depth
        self.distance = distance
        if invariant_type == 'type_sigmoid_attn':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.type_attn_layer = nn.ModuleList([
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout, embedding_dimension_query=2),     ### 用type做query去提取信息
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, 8)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout, embedding_dimension_query=8+hidden_dim),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])               
            self.map_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.fc_sum = nn.Linear(2*hidden_dim, hidden_dim)
        elif invariant_type == 'type_direct_attn':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim*2)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim*2, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim*2, hidden_dim*2, dropout = dropout)
            ])
            self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim*2, heads = heads, dim_head = dim_head, dropout = dropout, embedding_dimension_query=hidden_dim),
                    FeedForward(hidden_dim*2, hidden_dim*2, dropout = dropout)]
            )
            self.fc_sum = nn.Linear(2*hidden_dim, hidden_dim)
        elif invariant_type == "self_cross":
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.spatial_attn_layer = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.agent_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, hidden_dim//4)
                self.map_net = nn.Linear(hidden_dim+hidden_dim//4, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention_Querydimchanged(hidden_dim, hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])   
            self.fc_sum = nn.Linear(hidden_dim, hidden_dim)
        elif invariant_type == 'cross':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.agent_attn_layer = nn.ModuleList([
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.fc_sum = nn.Linear(hidden_dim, hidden_dim)
        elif invariant_type == "attn_sum":
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.attn_net = nn.ModuleList([
                Attention(hidden_dim * 2, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim * 2, mlp_dim, dropout = dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ])
            self.fc_max = nn.Linear(hidden_dim, hidden_dim)
            self.fc_sum = nn.Linear(hidden_dim, hidden_dim)
        elif invariant_type == "attn_mean":
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.attn_net = nn.ModuleList([
                Attention(hidden_dim * 2, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim * 2, mlp_dim, dropout = dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ])
            self.fc_max = nn.Linear(hidden_dim, hidden_dim)
            self.fc_mean = PreNorm(hidden_dim, nn.Linear(hidden_dim, hidden_dim))
        elif invariant_type == "attn_rnn":
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.attn_net = nn.ModuleList([
                Attention(hidden_dim * 2, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim * 2, mlp_dim, dropout = dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ])
            self.attn_rnn = nn.ModuleList([
                nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, mlp_dim, dropout = dropout)
            ])
        elif invariant_type == 'attn_N':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.attn_net = Transformer(hidden_dim, depth = 1, heads = heads, dim_head = dim_head, mlp_dim = mlp_dim, dropout = 0.)
        elif invariant_type == "mean":
            pass
        elif invariant_type == 'type_attn_no_sigmoid':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.type_map = nn.Linear(2, hidden_dim)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.type_attn_layer = nn.ModuleList([
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, hidden_dim//4)
                self.map_net = nn.Linear(hidden_dim+hidden_dim//4, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention_Querydimchanged(hidden_dim, hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])               
            self.fc_sum = nn.Linear(2*hidden_dim, hidden_dim)
        elif invariant_type == 'type_attn':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.type_map = nn.Linear(2, hidden_dim)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.type_attn_layer = nn.ModuleList([
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, hidden_dim//4)
                self.map_net = nn.Linear(hidden_dim+hidden_dim//4, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention_Querydimchanged(hidden_dim, hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])               
            self.map_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.fc_sum = nn.Linear(2*hidden_dim, hidden_dim)
        elif invariant_type == 'type_attn_wo_twoheads':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.type_map = nn.Linear(2, hidden_dim)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.type_attn_layer = nn.ModuleList([
                CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, hidden_dim//4)
                self.map_net = nn.Linear(hidden_dim+hidden_dim//4, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention_Querydimchanged(hidden_dim, hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])               
            self.fc_sum = nn.Linear(hidden_dim, hidden_dim)
        elif self.invariant_type == 'type_attn_wo_type':
            self.encode_actor_net = nn.Linear(input_dim, hidden_dim)
            self.encode_other_net = nn.Linear(neighbor_dim, hidden_dim)
            self.type_map = nn.Linear(2, hidden_dim)
            self.spatial_attn_layer_x = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            self.spatial_attn_layer_y = nn.ModuleList([
                Attention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(hidden_dim, hidden_dim, dropout = dropout)
            ])
            if self.distance == 0:
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 1:
                self.encode_distance_net = nn.Linear(2, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])
            elif self.distance == 2:
                self.encode_distance_net = nn.Linear(2, hidden_dim//4)
                self.map_net = nn.Linear(hidden_dim+hidden_dim//4, hidden_dim)
                self.cross_attn_layer = nn.ModuleList([
                    CrossAttention_Querydimchanged(hidden_dim, hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                    FeedForward(hidden_dim, hidden_dim, dropout = dropout)
                ])               
            self.fc_sum = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise NotImplementedError
    
    def attn_type(self, x):
        x, x_type = x
        attn, ff = self.type_attn_layer
        x = attn(x, x_type) + x     ### residual的结构
        x = ff(x)       ## feed forward layer
        return x
    
    def attn_cross(self, x):
        x, y = x
        attn, ff = self.agent_attn_layer
        x = attn(x, y) + x
        # x = attn(x, y) 
        x = ff(x)
        # x = ff(x) + x
        return x

    def attn_cross_distance(self, x):
        x, main = x
        attn, ff = self.cross_attn_layer
        x = attn(x, main) + x
        # x = attn(x, main)
        # x = ff(x) + x 
        x = ff(x)
        return x
    
    def attn_self_x(self, x):
        attn, ff = self.spatial_attn_layer_x
        # x = attn(x)
        x = attn(x) + x
        x = ff(x)
        return x
    
    def attn_self_y(self, x):
        attn, ff = self.spatial_attn_layer_y
        x = attn(x) + x
        # x = attn(x)
        x = ff(x) 
        return x

    def forward(self, x, others, neighbor_masks=None, neighbor_types=None, neighbor_relations=None, neighbor_distances=None):
        B = x.shape[0]
        assert len(others) == len(neighbor_masks) == len(neighbor_types) == len(neighbor_relations) == len(neighbor_distances) == 4
        if self.invariant_type == 'type_attn':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            out_typea, out_typeb = torch.zeros(B, 2, self.hidden_dim).to(x.device), torch.zeros(B, 2, self.hidden_dim).to(x.device)
            coef_typea, coef_typeb = torch.zeros(B, 2, 1).to(x.device), torch.zeros(B, 2, 1).to(x.device)
            for i in range(2):
                y, neighbor_mask, neighbor_relation, neighbor_distance = others[i], neighbor_masks[i], neighbor_relations[i], neighbor_distances[i]
                y = self.encode_other(y)
                neighbor_relation = self.type_map(neighbor_relation)
                y = self.attn_type((y, neighbor_relation))  
                y = self.attn_self_y(y) 
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    x_distance = self.map_net(x_distance)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z_sig = self.map_layer(z)
                z_sig[neighbor_mask==0, :] = -np.inf
                out_typea[:, i, :] = z
                coef_typea[:, i, :] = z_sig
            coef_typea = torch.softmax(coef_typea, dim=1)
            coef_typea = torch.nan_to_num(coef_typea, nan=0.0)
            out_typea = torch.sum(out_typea * coef_typea, dim=1)
            for i in range(2, 4):
                y, neighbor_mask, neighbor_relation, neighbor_distance = others[i], neighbor_masks[i], neighbor_relations[i], neighbor_distances[i]
                y = self.encode_other(y)
                neighbor_relation = self.type_map(neighbor_relation)
                y = self.attn_type((y, neighbor_relation))  
                y = self.attn_self_y(y) 
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    x_distance = self.map_net(x_distance)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z_sig = self.map_layer(z)
                z_sig[neighbor_mask==0, :] = -np.inf
                out_typeb[:, i-2, :] = z
                coef_typeb[:, i-2, :] = z_sig
            coef_typeb = torch.softmax(coef_typeb, dim=1)
            coef_typeb = torch.nan_to_num(coef_typeb, nan=0.0)
            out_typeb = torch.sum(out_typeb * coef_typeb, dim=1)
            return x, self.fc_sum(torch.cat([out_typea, out_typeb], dim=-1))
        elif self.invariant_type == 'type_attn_no_sigmoid':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            out_sum_type0, out_sum_type1 = torch.zeros(B, self.hidden_dim).to(x.device), torch.zeros(B, self.hidden_dim).to(x.device)
            for _, (y, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance) in enumerate(zip(others, neighbor_masks, neighbor_types, neighbor_relations, neighbor_distances)):
                y = self.encode_other(y)
                neighbor_relation = self.type_map(neighbor_relation)
                y = self.attn_type((y, neighbor_relation))  
                y = self.attn_self_y(y) 
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    x_distance = self.map_net(x_distance)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z[neighbor_mask==0, :] = 0
                a = z.clone()
                z[neighbor_type==1, :] = 0
                out_sum_type0 = out_sum_type0 + z
                z = a
                z[neighbor_type==0, :] = 0
                out_sum_type1 = out_sum_type1 + z
            return x, self.fc_sum(torch.concat([out_sum_type0, out_sum_type1], dim=-1))
        elif self.invariant_type == 'type_sigmoid_attn':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            out_typea, out_typeb = torch.zeros(B, 2, self.hidden_dim).to(x.device), torch.zeros(B, 2, self.hidden_dim).to(x.device)
            coef_typea, coef_typeb = torch.zeros(B, 2, 1).to(x.device), torch.zeros(B, 2, 1).to(x.device)
            for i in range(2):
                y, neighbor_mask, neighbor_relation, neighbor_distance = others[i], neighbor_masks[i], neighbor_relations[i], neighbor_distances[i]
                y = self.encode_other(y)
                y = self.attn_type((y, neighbor_relation))
                y = self.attn_self_y(y)
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z_sig = self.map_layer(z)
                z_sig[neighbor_mask==0, :] = -np.inf
                out_typea[:, i, :] = z
                coef_typea[:, i, :] = z_sig
            coef_typea = torch.softmax(coef_typea, dim=1)
            coef_typea = torch.nan_to_num(coef_typea, nan=0.0)
            out_typea = torch.sum(out_typea * coef_typea, dim=1)
            for i in range(2, 4):
                y, neighbor_mask, neighbor_relation, neighbor_distance = others[i], neighbor_masks[i], neighbor_relations[i], neighbor_distances[i]
                y = self.encode_other(y)
                y = self.attn_type((y, neighbor_relation))
                y = self.attn_self_y(y)
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z_sig = self.map_layer(z)
                z_sig[neighbor_mask==0, :] = -np.inf
                out_typeb[:, i-2, :] = z
                coef_typeb[:, i-2, :] = z_sig
            coef_typeb = torch.softmax(coef_typeb, dim=1)
            coef_typeb = torch.nan_to_num(coef_typeb, nan=0.0)
            out_typeb = torch.sum(out_typeb * coef_typeb, dim=1)
            return x, self.fc_sum(torch.cat([out_typea, out_typeb], dim=-1))
        elif self.invariant_type == 'type_direct_attn':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            y = torch.cat(others, dim=1)
            y = self.encode_other(y)
            y = self.attn_cross_distance((y, x))
            # return self.fc_sum(torch.cat([x, y], dim=-1))     ## 之前这里有误
            return x, self.fc_sum(y)
        elif self.invariant_type == 'type_attn_wo_twoheads':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            out_sum = torch.zeros(B, self.hidden_dim).to(x.device)
            for _, (y, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance) in enumerate(zip(others, neighbor_masks, neighbor_types, neighbor_relations, neighbor_distances)):
                y = self.encode_other(y)
                neighbor_relation = self.type_map(neighbor_relation)
                y = self.attn_type((y, neighbor_relation))
                y = self.attn_self_y(y)
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    x_distance = self.map_net(x_distance)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z[neighbor_mask==0, :] = 0
                out_sum = out_sum + z
            return x, self.fc_sum(out_sum)
        elif self.invariant_type == 'type_attn_wo_type':
            x = self.encode_actor(x)
            x = self.attn_self_x(x)
            out_sum = torch.zeros(B, self.hidden_dim).to(x.device)
            for _, (y, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance) in enumerate(zip(others, neighbor_masks, neighbor_types, neighbor_relations, neighbor_distances)):
                y = self.encode_other(y)
                y = self.attn_self_y(y)
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    x_distance = self.map_net(x_distance)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross_distance((y, x))
                z[neighbor_mask==0, :] = 0
                out_sum = out_sum + z
            return x, self.fc_sum(out_sum)
        elif self.invariant_type == "self_cross":
            x = self.encode_actor(x)
            out_sum = torch.zeros(B, self.hidden_dim).to(x.device)
            for idx, (y, neighbor_mask, neighbor_type, neighbor_relation, neighbor_distance) in enumerate(zip(others, neighbor_masks, neighbor_types, neighbor_relations, neighbor_distances)):
                y = self.encode_other(y)
                z = self.attn_self(y)      ### self attention
                # z = self.attn_cross((z, x))    ### cross attention
                if self.distance == 1:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = x + neighbor_distance
                    z = self.attn_cross_distance((y, x_distance))
                elif self.distance == 2:
                    neighbor_distance = self.encode_distance_net(neighbor_distance)
                    x_distance = torch.cat([x, neighbor_distance], dim=1)
                    x_distance = self.map_net(x_distance)
                    z = self.attn_cross_distance((y, x_distance))
                else:
                    z = self.attn_cross((z, x))
                z[neighbor_mask==0, :] = 0
                out_sum = out_sum + z
            return x, self.fc_sum(out_sum)
        elif self.invariant_type == 'cross':
            x = self.encode_actor(x)
            out_sum = torch.zeros(B, self.hidden_dim).to(x.device)
            for _, (y, neighbor_mask) in enumerate(zip(others, neighbor_masks)):
                y = self.encode_other(y)
                z = self.attn_cross((y, x))
                z[neighbor_mask==0, :] = 0

                out_sum = out_sum + z
            return x, self.fc_sum(out_sum)
        elif self.invariant_type == "attn_sum":
            x = self.encode_actor(x) # B x D
            out_max = torch.zeros(B, self.hidden_dim).to(x.device)
            out_sum = torch.zeros(B, self.hidden_dim).to(x.device)
            for _, (y, neighbor_mask) in enumerate(zip(others, neighbor_masks)):
                y = self.encode_other(y) # B x D
                z = torch.cat([x,y], dim=-1)
                z = self.attn(z)
                z[neighbor_mask==0, :] = 0 
                
                out_max = torch.max(out_max, z)
                out_sum = out_sum + z
            return x, 0.0 * self.fc_max(out_max) + 1.0 * self.fc_sum(out_sum)
        elif self.invariant_type == "attn_mean":
            x = self.encode_actor(x) # B x D
            out_max = torch.zeros(B, self.hidden_dim).to(x.device)
            out_sum = torch.zeros(B, self.hidden_dim).to(x.device)
            for _, (y, neighbor_mask) in enumerate(zip(others, neighbor_masks)):
                y = self.encode_other(y) # B x D
                z = torch.cat([x,y], dim=-1)
                z = self.attn(z)
                z[neighbor_mask==0, :] = 0 
                
                out_max = torch.max(out_max, z)
                out_sum = out_sum + z
            if np.sum(neighbor_masks) > 0:
                out_sum = out_sum / np.sum(neighbor_masks)
            return x, 0.0 * self.fc_max(out_max) + 1.0 * self.fc_mean(out_sum)
        elif self.invariant_type == "attn_rnn":
            x = self.encode_actor(x) # B x 64 x D
            out = torch.zeros(B, self.hidden_dim).to(x.device)
            for y in others:
                y = self.encode_other(y) # B x 64 x D
                z = torch.cat([x,y], dim=-1)
                z = self.attn(z)

                norm1, norm2, to_kv, cross_attn, ff = self.attn_rnn
                out = norm1(out)
                k, v = to_kv(norm2(z)).chunk(2, dim=-1)
                out = out + cross_attn(out, k, v)
                out = out + ff(out) 
            return x, out
        elif self.invariant_type == "mean":
            for y in others:
                x = x+y
            return x / (len(others)+1)
        else:
            raise NotImplementedError
    
    def encode_actor(self, x):
        fc = self.encode_actor_net
        return fc(x)
    
    def encode_other(self, y):
        fc = self.encode_other_net
        return fc(y)
    
    def attn_self(self, x):
        attn, ff = self.spatial_attn_layer
        x = attn(x) + x
        x = ff(x)
        # x = attn(x)
        # x = ff(x) + x
        return x
    
    def attn(self, x):
        attn, ff, fc = self.attn_net
        x = attn(x) + x
        # x = attn(x)
        x = ff(x) + x
        x = fc(x)
        return x
