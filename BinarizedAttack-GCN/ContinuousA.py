import torch
import os
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import argparse
from GCN_model import GCN
from utils import load_anomaly_detection_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--dataset', default='BlogCatalog', choices=['Flickr', 'ACM', 'BlogCatalog'], help='dataset name: Flickr/ACM/BlogCatalog')
parser.add_argument('--hidden_size', type=float, default=32, help='hidden size')
parser.add_argument('--lam', type=float, default=0.5, help='Lambda')
parser.add_argument('--epochs', type=int, default=500, help='Training epoch')
parser.add_argument('--lr', type=float, default=300, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--trial', type=int, default=5, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--device', default='cuda:0', type=str, help='cuda/cpu')
torch.cuda.empty_cache()
torch.cuda.init()
args = parser.parse_args()

dirs = '/home/user/Desktop/attack_oddball_extension/src/black-box-attack'

save_dir = dirs+'/data/'+args.dataset+'/ContinuousA/'+str(args.trial)
try:
    os.makedirs(save_dir)
except:
    pass
np.random.seed(args.seed)
torch.manual_seed(args.seed)

adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
train_idx = np.loadtxt(dirs + '/data/' + args.dataset + '/train_idx'+str(args.trial)+'.txt', dtype='int32')
test_idx = np.loadtxt(dirs + '/data/' + args.dataset + '/test_idx'+str(args.trial)+'.txt', dtype='int32')
n_node = len(adj)
def adj_to_tri_all(adj):
    triple_all = np.concatenate((np.triu_indices(len(adj.todense()), k=1)[0].reshape(-1,1), \
                                 np.triu_indices(len(adj.todense()), k=1)[1].reshape(-1,1), \
                                 adj[np.triu_indices(len(adj.todense()), k=1)].T),1)
    return np.array(triple_all)

triple = adj_to_tri_all(sp.csr_matrix(adj_label)).astype('float32')
# =============================================================================
# triple = []
# for i in range(n_node):
#     for j in range(i+1,n_node):
#         triple.append([i,j,adj_label[i,j]])
# triple = np.array(triple)
# =============================================================================


adj = torch.FloatTensor(adj).to(args.device)

n_edges = (triple[:,2]==1).sum()

adj_label = torch.FloatTensor(adj_label).to(args.device)

attrs = torch.FloatTensor(attrs).to(args.device)

label = torch.FloatTensor(label).to(args.device)
#y_train = label[train_idx]
y_pred = torch.from_numpy(np.loadtxt(dirs+'/data/'+args.dataset+'/gcn_pred'+str(args.trial)+'.txt', dtype='float32')).to(args.device)

class Surrogate_GNN(nn.Module):
    def __init__(self, n_node, feat_dim, out_dim, train_idx, test_idx, tri, device):
        super().__init__()
        self.f_dim = feat_dim
        self.out_dim = out_dim
        self.n = n_node
        self.tri = tri
        self.device = device
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.edge = nn.Parameter(torch.Tensor(tri[:,2]).to(args.device))
    
    def sparse_matrix_mul(self, A1, A2):
        return torch.sparse.mm(A1.to_sparse(), A2.to_sparse()).to_dense()
    
    def adjacency_matrix(self):
        A = torch.sparse_coo_tensor(self.tri[:,:2].T, self.edge, size=[self.n,self.n]).to_dense()
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def normalize_adj(self, adj):
        adj_1 = adj + torch.eye(adj.shape[0]).to(self.device)
        D_05 = torch.diag(1/torch.sqrt(adj_1.sum(0)))
        #return torch.mm(torch.mm(D_05, adj_1), D_05)
        return D_05 @ adj_1 @ D_05
        #return self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, adj_1), D_05)
    
    def feat(self, x):
        self.A = self.adjacency_matrix()
        self.A_norm = self.normalize_adj(self.A)
        #A2 = self.sparse_matrix_mul(self.A_perturb_norm, self.A_perturb_norm)
        A2 = torch.linalg.matrix_power(self.A_norm, 2)
        #H = self.sparse_matrix_mul(A2, x)
        H = torch.mm(A2, x)
        return H
    
    # Ridge regression to penalize high dimensional attributes and prevent singular matrix in matrix inverse.
    def Ridge(self, x, y_train):
        lam=0.1
        H = self.feat(x)
        H_train = H[self.train_idx]
        H1 = torch.cat((torch.ones((len(H_train),1)).to(self.device), H_train), 1)
        I = torch.eye(x.shape[1]+1).to(self.device)
        theta = torch.linalg.inv((H1.T @ H1) + lam*I) @ H1.T @ (y_train.reshape(-1,1))
        self.bias = nn.Parameter(theta[:1].reshape(1))
        self.weight = nn.Parameter(theta[1:].reshape(self.f_dim,1))
        return torch.mm(H_train, self.weight) + self.bias
    
    # Weighted Ridge regression. beta_hat = (X^{T}WX+lam*I)^{-1}(X^{T}WY. I_{pp}.
    def forward(self, x, y_train):
        lam = 0.1
        H = self.feat(x)
        H_train = H[self.train_idx]
        H_test = H[self.test_idx]
        H1 = torch.cat((torch.ones((len(H_train),1)).to(self.device), H_train), 1)
        pos_weight = ((y_train==0).sum()/(y_train==1).sum()).item()
        weight_tensor = torch.ones_like(y_train.detach().clone())
        weight_tensor[y_train==1]=pos_weight
        weight_tensor = torch.diag(weight_tensor)
        I = torch.eye(x.shape[1]+1).to(self.device)
        theta = torch.linalg.inv((H1.T @ weight_tensor @ H1) + lam*I) @ H1.T @ weight_tensor @ (y_train.reshape(-1,1))
        bias = nn.Parameter(theta[:1].reshape(1))
        weight = nn.Parameter(theta[1:].reshape(self.f_dim,1))
        outputs_train = torch.sigmoid(torch.mm(H_train, weight) + bias)
        outputs_test = torch.sigmoid(torch.mm(H_test, weight) + bias)
        return outputs_train, outputs_test

model = Surrogate_GNN(n_node, attrs.shape[1], 1, train_idx, test_idx, triple, args.device).to(args.device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) 

loss_fn = nn.BCELoss()

loss_lst = []
y_train = label[model.train_idx]
for epoch in range(args.epochs):
    optimizer.zero_grad()
    outputs_train, outputs_test = model(attrs, y_train)
    atk_loss =  -((1-args.lam)*loss_fn(outputs_test, y_pred.reshape(-1,1)) + \
                     args.lam *loss_fn(outputs_train, y_train.reshape(-1,1)))
    atk_loss.backward()
    optimizer.step()
    model.edge.data.clamp_(0., 1.)
    print('epoch:', epoch, 'attack loss:', atk_loss.item())
    loss_lst.append(atk_loss.item())

plt.plot(loss_lst)

np.savetxt(save_dir+'/loss.txt', loss_lst)
torch.save(model.state_dict(),save_dir+"/ckt.pth")

