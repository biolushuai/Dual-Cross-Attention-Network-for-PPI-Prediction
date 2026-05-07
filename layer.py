import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GATv2Conv, TransformerConv
import torch.nn.functional as F
import numpy as np


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_dim):
        super(ChannelAttention, self).__init__()
        self.gat = GATConv(in_dim, 64, heads=2, concat=False)
        self.gru = nn.GRU(in_dim, 32, bidirectional=True, batch_first=True)


        self.channel_attention1 = nn.MultiheadAttention(64, num_heads=2)
        self.channel_attention2 = nn.MultiheadAttention(64, num_heads=2)

    def forward(self, p):
        node_feature, edge_index = p.x, p.edge_index

        x1 = self.gat(node_feature, edge_index)
        x2 = self.gru(node_feature)[0]
        out1 = self.channel_attention1(x1, x2, x2)[0]
        out2 = self.channel_attention2(x2, x1, x1)[0]
        out = torch.cat((out1, out2), 1)
        return out


class DualAttentionPPI(nn.Module):
    def __init__(self, feature_dim, n_heads):
        super().__init__()
        self.ch_att1 = ChannelAttention(feature_dim)
        self.ch_att2 = ChannelAttention(feature_dim)

        # self.linear1 = nn.Linear(feature_dim * 2, feature_dim)
        # self.linear2 = nn.Linear(feature_dim * 2, feature_dim)

        self.cross_attention = nn.MultiheadAttention(128, n_heads, batch_first=True)

    def cross_attention_func(self, splited_h):

        splited_h1 = splited_h[0]
        splited_h2 = splited_h[1]

        # splited_h1 = self.linear1(splited_h1)
        # splited_h2 = self.linear1(splited_h2)

        q_1, k_1, v_1 = splited_h1, splited_h1, splited_h1
        q_2, k_2, v_2 = splited_h2, splited_h2, splited_h2

        output1, attn_weights1 = self.cross_attention(query=q_1, key=k_2, value=v_2) # attn_weights1 L1*L2
        output2, attn_weights2 = self.cross_attention(query=q_2, key=k_1, value=v_1) # attn_weights1 L2*L1
        # np.savetxt('./case/a1.csv', attn_weights1.cpu().numpy(), delimiter=",")
        # np.savetxt('./case/a2.csv', attn_weights2.cpu().numpy(), delimiter=",")

        b1 = torch.mean(attn_weights1, 1) # L1
        # np.savetxt('./case/b1.csv', b1.cpu().numpy(), delimiter=",")
        p1 = torch.softmax(b1, 0) # L1
        # np.savetxt('./case/p1.csv', p1.cpu().numpy(), delimiter=",")

        s1 = torch.matmul(torch.t(splited_h1), p1).view(1, -1) #(1, 128)
        # np.savetxt('./case/s1.csv', s1.cpu().numpy(), delimiter=",")

        b2 = torch.mean(attn_weights2, 1) # L2
        # np.savetxt('./case/b2.csv', b2.cpu().numpy(), delimiter=",")
        p2 = torch.softmax(b2, 0) # L2
        # np.savetxt('./case/p2.csv', p2.cpu().numpy(), delimiter=",")
        s2 = torch.matmul(torch.t(splited_h2), p2).view(1, -1)#(1, 128)
        # np.savetxt('./case/s2.csv', s2.cpu().numpy(), delimiter=",")

        return s1, s2

    def mutual_attention(self, h1, h2):
        x1 = h1.x
        x2 = h2.x

        mark_h1 = list(torch.unique(h1.batch,return_counts=True)[1].cpu().tolist())
        mark_h2 = list(torch.unique(h2.batch, return_counts=True)[1].cpu().tolist())

        splited_h1 = torch.split(x1, mark_h1, dim=0)
        splited_h2 = torch.split(x2, mark_h2, dim=0)

        h1_total, h2_total = zip(*list(map(self.cross_attention_func,list(zip(splited_h1,splited_h2)))))
        h1_total, h2_total = torch.vstack(list(h1_total)),torch.vstack(list(h2_total))

        return h1_total,h2_total

    def forward(self, p1, p2):
        p1.x = self.ch_att1(p1)
        p2.x = self.ch_att1(p2)

        p1_out, p2_out = self.mutual_attention(p1, p2)
        # print('p1_out, p2_out', p1_out.shape, p2_out.shape)

        return p1_out, p2_out

