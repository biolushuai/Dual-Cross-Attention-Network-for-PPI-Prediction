import torch.nn as nn
from torch_geometric.nn import LayerNorm
import torch
from layer import DualAttentionPPI
import torch.nn.functional as F
class nnModel_multi_class(nn.Module):

    def __init__(self, feature_dim = 1280, heads=2):
        super(nnModel_multi_class,self).__init__()
        self.ln = LayerNorm(feature_dim)
        self.relu = nn.ReLU()
        # self.logsoftmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()

        self.block1 = DualAttentionPPI(feature_dim, heads)

        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(128 * 2, 512)  # 64
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 7)

        self.dense_go1 = nn.Linear(128,512)
        self.dense_go2 = nn.Linear(512,100)


        self.dense_go11 = nn.Linear(128, 512)
        self.dense_go22 = nn.Linear(512, 100)

        self.dense_location1 = nn.Linear(128,512)
        self.dense_location2 = nn.Linear(512,100)

        self.dense_location11 = nn.Linear(128, 512)
        self.dense_location22 = nn.Linear(512, 100)

    def forward(self, p1, p2):
        p1.x = self.ln(p1.x)
        p2.x = self.ln(p2.x)

        out1 = self.block1(p1,p2)
        p1_rep_layer1 = out1[0]
        p2_rep_layer1 = out1[1]

        rep_0 = torch.cat((p1_rep_layer1, p2_rep_layer1), 1)
        rep = self.dense1(rep_0)
        rep = self.relu(rep)
        rep = self.dropout(rep)
        rep = self.dense2(rep)
        rep = self.relu(rep)
        rep = self.dropout(rep)
        rep = self.dense3(rep)
        out = rep


        x_go1 = self.dense_go1(p1_rep_layer1)
        x_go1 = self.relu(x_go1)
        x_go1 = self.dropout(x_go1)
        x_go1 = self.dense_go2(x_go1)
        x_go1 = self.sigmoid(x_go1)

        x_go2 = self.dense_go11(p2_rep_layer1)
        x_go2 = self.relu(x_go2)
        x_go2 = self.dropout(x_go2)
        x_go2 = self.dense_go22(x_go2)
        x_go2 = self.sigmoid(x_go2)

        x_location1 = self.dense_location1(p1_rep_layer1)
        x_location1 = self.relu(x_location1)
        x_location1 = self.dropout(x_location1)
        x_location1 = self.dense_location2(x_location1)
        x_location1 = self.sigmoid(x_location1)

        x_location2 = self.dense_location11(p2_rep_layer1)
        x_location2 = self.relu(x_location2)
        x_location2 = self.dropout(x_location2)
        x_location2 = self.dense_location22(x_location2)
        x_location2 = self.sigmoid(x_location2)

        return out, x_go1, x_go2, x_location1, x_location2, rep_0
