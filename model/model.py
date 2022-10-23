import torch
from torch import nn
import torch.nn.functional as F
import os
import torchvision
# print(torchvision.__version__)
# import torchvision.ops.misc.MLP as MLP
import warnings
from typing import Callable, List, Optional
from torch.autograd import Variable

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, A, P, N):
        distance_pos = self.cos(A, P)
        distance_neg = self.cos(A, N)

        losses = torch.relu(distance_neg - distance_pos + self.margin)
        return losses.mean()

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder_layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims)-1):
            self.encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.encoder_layers.append(nn.ReLU())
            self.encoder_layers.append(nn.Dropout(0.1))
        
        self.encoder = nn.Sequential(*self.encoder_layers)
        # self.A = torch.rand((4000, 4000))
        # self.A = self.A.cuda()

    
    def forward(self, x):
        # x = torch.mm(x, self.A)
        return self.encoder(x)

class graphlayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        # self.adj_matrix = adj_matrix
    
    def forward(self, x, adj):
        x = torch.spmm(x, adj)
        output = self.relu(self.weight(x))
        return output

class graph(nn.Module):
    def __init__(self, input_dim, graph_dim, hidden_dims, num_layers=2):
        super().__init__()
        self.graph_layers = []
        self.g1 = graphlayer(input_dim, graph_dim)
        self.g2 = graphlayer(graph_dim, graph_dim)
        # self.graph_layers = nn.Sequential(*self.graph_layers)
        
        self.encoder_layers = []
        dims = [graph_dim] + hidden_dims

        for i in range(len(dims)-1):
            self.encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
            self.encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
            self.encoder_layers.append(nn.ReLU())
            self.encoder_layers.append(nn.Dropout(0.1))
        self.encoder = nn.Sequential(*self.encoder_layers)

    def forward(self, x, adj):
        # x = torch.mm(x, adj)
        # gemb = self.graph_layers(x, adj)
        g1 = self.g1(x, adj)
        g2 = self.g2(g1, adj)
        return self.encoder(g2)

class Encoder2(nn.Module):
    # input_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, out_dim
    def __init__(self, input_dim = 4000, channels=[200, 50, 20], out_dim=2):
        super().__init__()
        self.encoder_layers = []
        # print(input_dim)
        self.encoder= MLP(input_dim, channels)
        # self.encoder = nn.Sequential(*self.encoder_layers)

        # self.head = nn.Linear(channels[-1], out_dim)

    def forward(self, x1):
        # x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        z = self.encoder(x1)
        # print(z.shape)
        # out = self.head(z)
        return z

class GraphMlpModel(nn.Module):
    def __init__(self, input_dim, graph_dim, mlp_hid_dims, graph_hid_dims, graph_prop_layers=2):
        super().__init__()
        self.mlp_trunk = Encoder2(input_dim, mlp_hid_dims)
        self.graph_trunk = graph(input_dim, graph_dim, graph_hid_dims, num_layers=graph_prop_layers)

        # concat
        self.c1 = nn.Sequential(
            nn.Linear(mlp_hid_dims[-1] + graph_hid_dims[-1], mlp_hid_dims[-1]),
            nn.BatchNorm1d(mlp_hid_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, adj):
        z1 = self.mlp_trunk(x)
        z2 = self.graph_trunk(x, adj)
        combine = torch.cat([z1, z2], dim=1)
        z = self.c1(combine)
        return z


# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dims):
#         super().__init__()
#         self.encoder_layers = []
#         dims = [input_dim] + hidden_dims

#         for i in range(len(dims)-1):
#             self.encoder_layers.append(nn.Linear(dims[i], dims[i+1]))
#             self.encoder_layers.append(nn.BatchNorm1d(dims[i+1]))
#             self.encoder_layers.append(nn.ReLU())
#             self.encoder_layers.append(nn.Dropout(0.1))
        
#         self.encoder = nn.Sequential(*self.encoder_layers)
    
#     def forward(self, x, A):
#         x  = torch.mm(x,A) # x = B * N,  A=N*N -> B * N
#         return self.encoder(x)




# class Encoder(nn.Module):
#     # input_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5, out_dim
#     def __init__(self, input_dim = 4000, channels=[200, 50, 20], out_dim=2):
#         super().__init__()
#         self.encoder_layers = []
#         # print(input_dim)
#         self.encoder= MLP(input_dim, channels)
#         # self.encoder = nn.Sequential(*self.encoder_layers)

#         self.head = nn.Linear(channels[-1], out_dim)

#     def forward(self, x1, x2):
#         x = torch.cat([x1, x2], dim=1)
#         # print(x.shape)
#         z = self.encoder(x)
#         # print(z.shape)
#         out = self.head(z)
#         return out

class TripletGraphModel(nn.Module):
    def __init__(self, input_dim=4000,graph_dim=4000, mlp_channels=[200, 50, 20], graph_channels=[200, 50, 10], lr=1e-3,
                save_path='', device):
        super().__init__()
        self.device = device
        # only support single GPU for now
        # self.model = Encoder(input_dim, channels, out_dim)
        self.model = GraphMlpModel(input_dim, graph_dim, mlp_channels, graph_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = TripletLoss(margin=1.0)
        self.loss_stat = {}
        self.save_path = save_path

    def set_input(self, inputs, istrain=1):
        if istrain:
            self.A = Variable(inputs['A']).to(self.device)
            self.P = Variable(inputs['P']).to(self.device)
            self.N = Variable(inputs['N']).to(self.device)
            self.adj = Variable(inputs['adj']).to(self.device)
            # self.label = Variable(inputs['label']).cuda()
        else:
            self.x1 = Variable(inputs['x1']).to(self.device)
            self.x2 = Variable(inputs['x2']).to(self.device)
            self.adj = Variable(inputs['adj']).to(self.device)

    def forward(self):
        self.model.train()
        # self.z1, self.z2, self.out = self.model(self.x1, self.x2)
        # self.out = self.model(self.x1, self.x2)
        self.A_emb = self.model(self.A, self.adj)
        self.P_emb = self.model(self.P, self.adj)
        self.N_emb = self.model(self.N, self.adj)
    
    def inference(self):
        self.model.eval()
        # self.z1, self.z2, self.out = self.model(self.x1, self.x2)
        self.emb1 = self.model(self.x1, self.adj)
        self.emb2 = self.model(self.x2, self.adj)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        distance = cos(self.emb1, self.emb2)
        return distance

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def compute_loss(self):
        self.loss = self.criterion(self.A_emb, self.P_emb, self.N_emb)
        self.loss_stat['triplet loss'] = self.loss
    
    def get_cur_loss(self):
        return self.loss_stat
    
    def single_update(self):
        self.forward()
        self.compute_loss()
        self.backward()
    
    def save(self, name):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name+'.pth'))
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))


class TripletModel(nn.Module):
    def __init__(self, input_dim=4000, channels=[200, 50, 20], out_dim=2, lr=1e-3,
                save_path='', device='cuda:0'):
        super().__init__()
        self.device=device
        # only support single GPU for now
        # self.model = Encoder(input_dim, channels, out_dim)
        self.model = Encoder2(input_dim, channels, out_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = TripletLoss(margin=1.0)
        self.loss_stat = {}
        self.save_path = save_path

    def set_input(self, inputs, istrain=1):
        if istrain:
            self.A = Variable(inputs['A']).to(self.device)
            self.P = Variable(inputs['P']).to(self.device)
            self.N = Variable(inputs['N']).to(self.device)
            # self.label = Variable(inputs['label']).cuda()
        else:
            self.x1 = Variable(inputs['x1']).to(self.device)
            self.x2 = Variable(inputs['x2']).to(self.device)

    def forward(self):
        self.model.train()
        # self.z1, self.z2, self.out = self.model(self.x1, self.x2)
        # self.out = self.model(self.x1, self.x2)
        self.A_emb = self.model(self.A)
        self.P_emb = self.model(self.P)
        self.N_emb = self.model(self.N)
    
    def inference(self):
        self.model.eval()
        # self.z1, self.z2, self.out = self.model(self.x1, self.x2)
        self.emb1 = self.model(self.x1)
        self.emb2 = self.model(self.x2)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        distance = cos(self.emb1, self.emb2)
        return distance

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def compute_loss(self):
        self.loss = self.criterion(self.A_emb, self.P_emb, self.N_emb)
        self.loss_stat['triplet loss'] = self.loss
    
    def get_cur_loss(self):
        return self.loss_stat
    
    def single_update(self):
        self.forward()
        self.compute_loss()
        self.backward()
    
    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name, '.pth'))
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

class PairModel(nn.Module):
    def __init__(self, input_dim=4000, channels=[200, 50, 20], out_dim=2, lr=1e-3,
                save_path='', device='cuda:0'):
        super().__init__()
        self.device = device
        # only support single GPU for now
        # self.model = Encoder(input_dim, channels, out_dim)
        self.model = Encoder(input_dim * 2, channels, out_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_stat = {}
        self.save_path = save_path

    def set_input(self, inputs):
        self.x1 = Variable(inputs['x1']).to(self.device)
        self.x2 = Variable(inputs['x2']).to(self.device)
        self.label = Variable(inputs['label']).to(self.device)

    def forward(self):
        self.model.train()
        # self.z1, self.z2, self.out = self.model(self.x1, self.x2)
        self.out = self.model(self.x1, self.x2)
    
    def inference(self):
        self.model.eval()
        # self.z1, self.z2, self.out = self.model(self.x1, self.x2)
        self.out = self.model(self.x1, self.x2)
        return torch.argmax(self.out, dim=1)

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def compute_loss(self):
        # self.loss = self.criterion(self.out, self.label)
        # self.loss_stat['loss'] = self.loss
        idx = (self.label == 1)
        self.posloss = self.criterion(self.out[idx], self.label[idx])

        idx = (self.label == 0)
        self.negloss = self.criterion(self.out[idx], self.label[idx])

        self.loss = self.posloss * 100 + self.negloss
        self.loss_stat['posloss'] = self.posloss
        self.loss_stat['negloss'] = self.negloss
        self.loss_stat['loss'] = self.loss
    
    def get_cur_loss(self):
        return self.loss_stat
    
    def single_update(self):
        self.forward()
        self.compute_loss()
        self.backward()
    
    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, name, '.pth'))
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
