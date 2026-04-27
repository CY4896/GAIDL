# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import rtdl

class Mine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        h = self.relu(self.fc1(z))
        out = self.fc2(h)
        return out

class Embedding(nn.Module):
    def __init__(self, input_dim1, hidden_size1, n_layers, num_emb, steps):
        super().__init__()

        self.input_dim1 = input_dim1
        self.steps = steps
        self.hidden_size1 = hidden_size1
        self.n_layers = n_layers
        self.num_emb = num_emb

        self.gru1 = torch.nn.GRU(768, 16, n_layers, batch_first=True, bidirectional=True)
        self.GRU_gpt = torch.nn.GRU(768, 16, n_layers, batch_first=True, bidirectional=True)

        self.v1_NFT_CON_MLP = rtdl.NumericalFeatureTokenizer(n_features=14, d_token=8, bias=True,
                                                             initialization='uniform')
        self.v_transformer = nn.TransformerEncoderLayer(num_emb, 4, dim_feedforward=num_emb, dropout=0)

        self.announcement_att_net = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        self.expert_att_net = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))

        self.out1 = nn.Linear(64, 32)
        self.out2 = nn.Linear(64, 32)

        self.Merge = nn.Sequential(
            nn.Linear(176, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def calculate_attention_weight(self, input1, att_weight_network):
        att_score = att_weight_network(input1)
        att_weight = F.softmax(att_score, dim=1)
        return att_weight

    def forward(self, x_data):
        x, hidden_cr = self.gru1(x_data[1], None)
        gpt, hidden_gpt = self.GRU_gpt(x_data[2], None)

        announcement_att_weight = self.calculate_attention_weight(x, self.announcement_att_net)
        expert_att_weight = self.calculate_attention_weight(gpt, self.expert_att_net)

        announcement_representation = torch.sum(x * announcement_att_weight, dim=1)
        expert_representation = torch.sum(gpt * expert_att_weight, dim=1)

        mlcomp = torch.cat((announcement_representation, expert_representation), dim=-1)

        output1 = self.out1(mlcomp)
        output2 = self.out2(mlcomp)

        h2 = self.v_transformer(self.v1_NFT_CON_MLP(x_data[0]))
        h2 = h2.view(h2.shape[0], -1)

        h4 = torch.cat((output1, output2, h2), 1)
        pred = torch.sigmoid(self.Merge(h4))

        return pred, output1, output2, announcement_representation, expert_representation

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Embedding = Embedding(768, 16, 1, 8, 97)
        self.mine = Mine()

    def forward(self, x_data):
        pred, output1, output2, announcement_representation, expert_representation = self.Embedding(x_data)
        total_loss, total_train = self.compute_mi_loss(output1, output2, announcement_representation,
                                                       expert_representation)
        return pred, total_loss, total_train

    def compute_mi_loss(self, output1, output2, hidden_cr, hidden_gpt):
        N = output1.size(0)
        perm = torch.randperm(N)

        hidden_cr_ = hidden_cr[perm]
        hidden_gpt_ = hidden_gpt[perm]

        pos_out1_cr = self.mine(output1, hidden_cr)
        neg_out1_cr = self.mine(output1, hidden_cr_)
        pos_out1_gpt = self.mine(output1, hidden_gpt)
        neg_out1_gpt = self.mine(output1, hidden_gpt_)

        loss_cr1 = torch.mean(pos_out1_cr) - torch.log(torch.mean(torch.exp(neg_out1_cr)))
        loss_gpt1 = torch.mean(pos_out1_gpt) - torch.log(torch.mean(torch.exp(neg_out1_gpt)))
        loss1 = loss_cr1 + loss_gpt1

        pos_out2_review = self.mine(output2, hidden_gpt)
        neg_out2_review = self.mine(output2, hidden_gpt_)
        pos_out2_report = self.mine(output2, hidden_cr)
        neg_out2_report = self.mine(output2, hidden_cr_)
        loss_gpt2 = torch.mean(pos_out2_review) - torch.log(torch.mean(torch.exp(neg_out2_review)))
        loss_cr2 = (torch.mean(pos_out2_report) - torch.log(torch.mean(torch.exp(neg_out2_report))))

        loss2 = loss_gpt2 - loss_cr2

        total_loss = - (loss1 + loss2)
        total_train = -(loss_cr1 + loss_gpt1 + loss_gpt2 + loss_cr2)
        return total_loss, total_train