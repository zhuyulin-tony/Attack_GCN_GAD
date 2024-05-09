import torch
import os
import matplotlib.pyplot as plt 
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import argparse
from LGNN import Surrogate_GNN
from utils import load_anomaly_detection_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'BlogCatalog'])
parser.add_argument('--lam', type=float, default=0., help='Lambda')
parser.add_argument('--xi', type=float, default=0.1, help='Xi')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--device', default='cuda:0', type=str, help='cuda/cpu')
torch.cuda.empty_cache()
torch.cuda.init()
args = parser.parse_args()

dirs = '/home/user/Desktop/attack_oddball_extension_desktop/src/black-box-attack'

save_dir = dirs+'/data/'+args.dataset+'/RWLS1_greedy_lam='+str(args.xi)+'/'+str(args.trial)
try:
    os.makedirs(save_dir)
except:
    pass
np.random.seed(args.seed)
torch.manual_seed(args.seed)

adj, attrs, label, adj_label = load_anomaly_detection_dataset(args.dataset)
train_idx = np.loadtxt(dirs + '/data/' + args.dataset + '/train_idx'+str(args.trial)+'.txt', dtype='int32')
test_idx = np.loadtxt(dirs + '/data/' + args.dataset + '/test_idx'+str(args.trial)+'.txt', dtype='int32')

def adj_to_tri_all(adj):
    triple_all = np.concatenate((np.triu_indices(len(adj.todense()), k=1)[0].reshape(-1,1), \
                                 np.triu_indices(len(adj.todense()), k=1)[1].reshape(-1,1), \
                                 adj[np.triu_indices(len(adj.todense()), k=1)].T),1)
    return triple_all

triple = adj_to_tri_all(sp.csr_matrix(adj_label)).astype('float32')
adj = torch.FloatTensor(adj).to(args.device)
n_node = len(adj)
n_edges = (triple[:,2]==1).sum()

adj_label = torch.FloatTensor(adj_label).to(args.device)

attrs = torch.FloatTensor(attrs).to(args.device)

model = Surrogate_GNN(n_node, attrs.shape[1], 1, train_idx, test_idx, args.xi, args.device)

label = torch.FloatTensor(label).to(args.device)
#y_train = label[train_idx]
y_pred = torch.from_numpy(np.loadtxt(dirs+'/data/'+args.dataset+'/gcn_pred'+str(args.trial)+'.txt', dtype='float32')).to(args.device)

def BCE_loss_with_weight(output, label, pos_weight):
    cost = (1/len(label))*(pos_weight*((-label).t() @ torch.log(output.reshape(-1,)))-((1-label).t() @ torch.log(1-output.reshape(-1,))))
    return cost

class GradMaxSearch(nn.Module):
        def __init__(self, train_model, loss_fn, B, train_idx, test_idx, y_pred, device):
            super().__init__()
            self.model = train_model
            self.loss_fn = loss_fn
            self.B = B
            self.train_idx = train_idx
            self.y_pred = y_pred
            self.test_idx = test_idx
            self.device = device
            
        def inner_train(self, x, tri, y_train):
            self.model.reset_parameters()
            _ = self.model.RWLS(x, tri, y_train)
            #_ = self.model.OLS(x, tri, y_train)
    
        def get_meta_grad(self, x, triple_copy, label):
            y_train = label[self.train_idx]
            edges = Variable(triple_copy[:,2:], requires_grad = True)
            triple_torch = torch.cat((triple_copy[:,:2], edges),1)
            outputs = self.model(x, triple_torch, self.test_idx)
            outputs_train = self.model(x, triple_torch, self.train_idx)
            
            #pos_weight = ((label[self.train_idx]==0).sum()/(label[self.train_idx]==1).sum()).item()
            #atk_loss = - ((1-args.lam)*BCE_loss_with_weight(outputs, self.y_pred, pos_weight=pos_weight) + \
            #                 args.lam *BCE_loss_with_weight(outputs_train, y_train, pos_weight=pos_weight)) 
            
            atk_loss = - ((1-args.lam)*self.loss_fn(outputs, self.y_pred.reshape(-1,1)) + \
                          args.lam*self.loss_fn(outputs_train, y_train.reshape(-1,1)))
            atk_loss.backward()
            meta_grad = edges.grad.data.cpu().numpy()
            return np.concatenate((triple_copy[:,:2].cpu().data.numpy(), meta_grad), 1)
        
        def forward(self, x, triple, y):
            gcn_auc_lst = []
            surrogate_auc_lst = []
            triple_copy = torch.from_numpy(triple.copy()).to(self.device)
            perturb = []
            y_train = y[self.train_idx]
            y_test = y[self.test_idx].cpu().data.numpy()
            for i in range(self.B):
            #for i in tqdm(range(self.B), desc = 'Perturbing Graph'):
                st = time.time()
                if i != 0:
                    triple_copy = torch.from_numpy(triple_copy).to(self.device)
                self.inner_train(x, triple_copy, y_train)
                meta_grad = self.get_meta_grad(x, triple_copy, y) 
                v_grad = np.zeros((len(meta_grad),3))
                v_grad[:,0] = meta_grad[:,0]
                v_grad[:,1] = meta_grad[:,1]
                v_grad[np.where((triple_copy[:,2] == 0).cpu().data.numpy() * meta_grad[:,2] < 0)] = \
                meta_grad[np.where((triple_copy[:,2] == 0).cpu().data.numpy() * meta_grad[:,2] < 0)]
                v_grad[np.where((triple_copy[:,2] == 1).cpu().data.numpy() * meta_grad[:,2] > 0)] = \
                meta_grad[np.where((triple_copy[:,2] == 1).cpu().data.numpy() * meta_grad[:,2] > 0)]
        
                v_grad = v_grad[np.abs(v_grad[:,2]).argsort()]
                # attack w.r.t gradient information.
                K = -1
                while v_grad[K][:2].astype('int').tolist() in perturb:
                    K -= 1
                target_grad = v_grad[int(K)]
                #print(K, target_grad)
                target_index = np.where(np.all((triple[:,:2] == target_grad[:2]), axis = 1))[0][0]
                triple_copy = triple_copy.cpu().data.numpy()
                triple_copy[target_index,2] -= np.sign(target_grad[2])
                perturb.append([int(target_grad[0]),int(target_grad[1])])
                
                "Evaluation on surrogate model."
                self.model.reset_parameters()
                _ = self.model.RWLS(attrs, torch.from_numpy(triple_copy).to(self.device), y_train)
                #_ = self.model.OLS(attrs, torch.from_numpy(triple_copy).to(self.device), y_train)
                self.model.eval()
                preds = self.model(attrs, torch.from_numpy(triple_copy).to(self.device), self.test_idx).cpu().data.numpy()
                test_auc_s = roc_auc_score(y_test, preds)
                nt = time.time()
                print('perturb: {:04d}'.format(i+1), \
                      #'GCN AUC: {:.4f}'.format(auc_test), \
                      'Surrogate AUC: {:.4f}'.format(test_auc_s), \
                      'time: {:.4f}'.format(np.round(nt-st,3)))
                surrogate_auc_lst.append(test_auc_s)
                np.savetxt(save_dir+'/'+args.dataset+'_mtri_'+str(int(i+1))+'.txt',triple_copy[triple_copy[:,2]==1],fmt='%d')
            return gcn_auc_lst, surrogate_auc_lst
        
loss_fn = nn.BCELoss()
B = 2000
#B = int(0.01*n_edges)
gradmax = GradMaxSearch(model, loss_fn, B, train_idx, test_idx, y_pred, args.device)
gcn_auc, surrogate_auc = gradmax(attrs, triple, label)

np.savetxt(save_dir+'/RWLS1_'+str(args.xi)+'_surrogate_auc.txt', surrogate_auc)
#%%
plt.plot(surrogate_auc)

#np.savetxt(dirs+'/RWLS_0.1_gcn_auc.txt', gcn_auc)
np.savetxt(save_dir+'/RWLS1_'+str(args.xi)+'_surrogate_auc.txt', surrogate_auc)
#%%
gcn_auc_100 = np.loadtxt(dirs+'/RWLS_100_gcn_auc.txt')
gcn_auc_10 = np.loadtxt(dirs+'/RWLS_10_gcn_auc.txt')
gcn_auc_1 = np.loadtxt(dirs+'/RWLS_1_gcn_auc.txt')
gcn_auc_01 = np.loadtxt(dirs+'/RWLS_0.1_gcn_auc.txt')
gcn_auc_001 = np.loadtxt(dirs+'/RWLS_0.01_gcn_auc.txt')
gcn_auc_0001 = np.loadtxt(dirs+'/RWLS_0.001_gcn_auc.txt')
gcn_auc_00001 = np.loadtxt(dirs+'/RWLS_0.0001_gcn_auc.txt')
gcn_auc_ridge = np.loadtxt(dirs+'/ridge_gcn_auc.txt')
#%%
RWLS_lgnn_01 = np.loadtxt(dirs+'/RWLS_0.1_surrogate_auc.txt')
RWLS_gcn_01 = np.loadtxt(dirs+'/RWLS_0.1_gcn_auc.txt')
ridge_lgnn_01 = np.loadtxt(dirs+'/ridge_0.1_surrogate_auc.txt')
ridge_gcn_01 = np.loadtxt(dirs+'/ridge_0.1_gcn_auc.txt')
#%%
plt.title('GradMaxSearch with $\lambda$=0.1 GCN AUC')
plt.plot(RWLS_gcn_01, label='RWLS')
plt.plot(ridge_gcn_01, label='Ridge')
plt.xlabel('B')
plt.ylabel('testing AUC')
plt.legend()
plt.show()
#%%
plt.title('GradMaxSearch with $\lambda$=0.1 Surrogate AUC')
plt.plot(RWLS_lgnn_01, label='RWLS')
plt.plot(ridge_lgnn_01, label='Ridge')
plt.xlabel('B')
plt.ylabel('testing AUC')
plt.legend()
plt.show()
#%%
#atk_power = np.linspace(0,4999,4999).astype('int')
atk_power = np.array([0,10,100,1000,2000,4000,4999])

#plt.plot(atk_power, gcn_auc_ridge[atk_power], label='Ridge lam=0.01')
#plt.plot(atk_power, gcn_auc_10[atk_power], label='lam=10')
#plt.plot(atk_power, gcn_auc_1[atk_power], label='lam=1')
plt.plot(atk_power, gcn_auc_01[atk_power], label='lam=0.1')
#plt.plot(atk_power, gcn_auc_001[atk_power], label='RWLS lam=0.01')
#plt.plot(atk_power, gcn_auc_0001[atk_power], label='lam=0.001')
plt.legend()
plt.xlabel('B')
plt.ylabel('GCN AUC')
plt.show()
#%%
Bin_lgnn_100 = np.loadtxt(dirs+'/bin_gcn_auc_eta_100.txt')
Bin_lgnn_200 = np.loadtxt(dirs+'/bin_gcn_auc_eta_200.txt')
Bin_lgnn_250 = np.loadtxt(dirs+'/bin_gcn_auc_eta_250.txt')
Bin_lgnn_300 = np.loadtxt(dirs+'/bin_gcn_auc_eta_300.txt')
Bin_lgnn_350 = np.loadtxt(dirs+'/bin_gcn_auc_eta_350.txt')
Bin_lgnn_400 = np.loadtxt(dirs+'/bin_gcn_auc_eta_400.txt')
Bin_lgnn_1000 = np.loadtxt(dirs+'/bin_gcn_auc_eta_1000.txt')
#%%
plt.plot(Bin_lgnn_100, label='$\eta$=100')
plt.plot(Bin_lgnn_200, label='$\eta$=200')
plt.plot(Bin_lgnn_250, label='$\eta$=250')
plt.plot(Bin_lgnn_300, label='$\eta$=300')
plt.plot(Bin_lgnn_350, label='$\eta$=350')
plt.plot(Bin_lgnn_400, label='$\eta$=400')
plt.plot(Bin_lgnn_1000, label='$\eta$=1000')
plt.xlabel('iters')
plt.ylabel('Surrogate AUC')
plt.legend()
plt.show()



