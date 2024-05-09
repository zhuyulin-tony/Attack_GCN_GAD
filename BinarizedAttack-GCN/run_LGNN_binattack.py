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
from GCN_model import GCN
from BinAttack_model import BinAttack
import re
from shutil import copyfile
import shutil
from utils import load_anomaly_detection_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--dataset', default='BlogCatalog', choices=['cora', 'citeseer', 'BlogCatalog'], help='dataset name')
parser.add_argument('--hidden_size', type=float, default=32, help='hidden size')
parser.add_argument('--lam', type=float, default=0.5, help='Lambda')
parser.add_argument('--xi', type=float, default=0.1, help='xi.')
parser.add_argument('--epochs', type=int, default=800, help='Training epoch')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--trial', type=int, default=1, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--device', default='cuda:0', type=str, help='cuda/cpu')
torch.cuda.empty_cache()
torch.cuda.init()
args = parser.parse_args()

dirs = '/home/user/Desktop/attack_oddball_extension/src/black-box-attack'

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
triple_torch = torch.from_numpy(triple).to(args.device)
adj_label = torch.FloatTensor(adj_label).to(args.device)

attrs = torch.FloatTensor(attrs).to(args.device)

model = Surrogate_GNN(n_node, attrs.shape[1], train_idx, test_idx, args.xi, args.device)

label = torch.FloatTensor(label).to(args.device)
#y_train = label[train_idx]
y_pred = torch.from_numpy(np.loadtxt(dirs+'/data/'+args.dataset+'/gcn_pred'+str(args.trial)+'.txt', dtype='float32')).to(args.device)

loss_fn = nn.BCELoss()

model_dir = dirs + '/data/'+args.dataset+'/BinAttack/'+str(args.trial)+'/saved_ckt'
save_dir = dirs + '/data/'+args.dataset+'/BinAttack/'+str(args.trial)+'/sort_ckt'
try:
    os.makedirs(model_dir)
except:
    pass
try:
    os.makedirs(save_dir)
except:
    pass

def Pick_ckt(lst):
    table = np.zeros((len(lst), 2))
    for i in range(len(lst)):
        table[i,0] = int(re.findall(r'\d+', lst[i])[0])
        table[i,1] = float(re.findall(r'\d+', lst[i])[1] + '.' \
                         + re.findall(r'\d+', lst[i])[2])
        
    B_lst = list(set(table[:,0]))
    tmp = np.zeros((len(B_lst), 3))
    for j in range(len(B_lst)):
        tmp[j,0] = B_lst[j]
        tmp[j,1] = np.where(table[:,0] == B_lst[j])[0]\
                            [np.argmin(table[np.where(table[:,0] == B_lst[j])][:,1])]
        tmp[j,2] = np.min(table[np.where(table[:,0] == B_lst[j])][:,1])
        
    return tmp

def train(epochs, lr):
    #eta_lst = list(np.linspace(120, 1000, 89).astype('int32'))
    eta_lst = list(np.linspace(610, 1000, 40).astype('int32'))
    #eta_lst = list(np.linspace(50, 1000, 476).astype('int32'))
    for eta in eta_lst:
        surrogate_auc_lst = []
        attack_model = BinAttack(model, n_node, args.lam, train_idx, test_idx, args.device)
        for epoch in range(epochs):
            attack_model.train()
            outputs_train, outputs_test = attack_model(attrs, triple_torch, label)
            atk_loss = - ((1-args.lam)*loss_fn(outputs_test, y_pred.reshape(-1,1)) + \
                           args.lam*loss_fn(outputs_train, label[train_idx].reshape(-1,1)))
            atk_loss += eta * torch.mean(torch.abs(attack_model.Z_continue))
            atk_loss.backward()
            with torch.no_grad():
                attack_model.Z_continue -= lr * attack_model.Z_continue.grad.data
                # projection gradient descent.
                attack_model.Z_continue.data.clamp_(1e-3, 1.-1e-3)
                attack_model.Z_continue.grad.zero_()
            
            perturb = (attack_model.Z_continue >= 0.5).sum().item()
            
            "Evaluation on surrogate model."
            triple_lgnn = attack_model.perturb(triple_torch).detach().clone()
            attack_model.model.reset_parameters()
            _ = attack_model.model.RWLS(attrs, triple_lgnn, label[train_idx])
            #_ = self.model.OLS(attrs, torch.from_numpy(triple_copy).to(self.device), y_train)
            attack_model.model.eval()
            preds = attack_model.model(attrs, triple_lgnn, test_idx).cpu().data.numpy()
            test_auc_s = roc_auc_score(label[test_idx].cpu().data.numpy(), preds)
            surrogate_auc_lst.append(test_auc_s)
            print('eta: {:04d}'.format(eta), \
                  'epoch: {:04d}'.format(epoch+1), \
                  'perturb: {:04d}'.format(perturb), \
                  'GCN AUC: {:.4f}'.format(test_auc_s))
            if perturb <= 5000 and perturb % 2 == 0:
                torch.save(attack_model.state_dict(),model_dir+"/P="\
                                                              +str(perturb)+",AUC="+str(np.round(test_auc_s,3))+".pth",
                                                              _use_new_zipfile_serialization=False)
            
        #if eta_lst.index(eta) % 2 == 0:
        #print('lets save model!!!!!!')
        model_name_lst = os.listdir(model_dir)
        ckt = Pick_ckt(model_name_lst)
        for k in range(len(ckt)):
            idx = int(ckt[k,1])
            copyfile(model_dir + '/' + model_name_lst[idx], save_dir + '/' + model_name_lst[idx])
        
        shutil.rmtree(model_dir)
        #os.rmdir(model_dir)
        os.rename(save_dir, model_dir)
        os.makedirs(save_dir)
        
train(4000, 0.5)  
#%%
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
from GCN_model import GCN
from BinAttack_model import BinAttack
import re
from shutil import copyfile
import shutil
from utils import load_anomaly_detection_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666, help='seed')
parser.add_argument('--dataset', default='BlogCatalog', choices=['cora', 'citeseer', 'BlogCatalog'], help='dataset name')
parser.add_argument('--hidden_size', type=float, default=32, help='hidden size')
parser.add_argument('--lam', type=float, default=0.5, help='Lambda')
parser.add_argument('--epochs', type=int, default=800, help='Training epoch')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--trial', type=int, default=5, choices=[1,2,3,4,5], help='trial')
parser.add_argument('--device', default='cuda:0', type=str, help='cuda/cpu')
torch.cuda.empty_cache()
torch.cuda.init()
args = parser.parse_args()

#dirs = '/home/user/Desktop/attack_oddball_extension/src/black-box-attack'
dirs = '/media/user/ZHU Yulin Repository/polyu/attack_oddball_extension/src/black-box-attack'

model_dir = dirs + '/data/'+args.dataset+'/BinAttack/'+str(args.trial)+'/saved_ckt'
save_dir = dirs + '/data/'+args.dataset+'/BinAttack/'+str(args.trial)+'/sort_ckt'

def Pick_ckt(lst):
    table = np.zeros((len(lst), 2))
    for i in range(len(lst)):
        table[i,0] = int(re.findall(r'\d+', lst[i])[0])
        table[i,1] = float(re.findall(r'\d+', lst[i])[1] + '.' \
                         + re.findall(r'\d+', lst[i])[2])
        
    B_lst = list(set(table[:,0]))
    tmp = np.zeros((len(B_lst), 3))
    for j in range(len(B_lst)):
        tmp[j,0] = B_lst[j]
        tmp[j,1] = np.where(table[:,0] == B_lst[j])[0]\
                            [np.argmin(table[np.where(table[:,0] == B_lst[j])][:,1])]
        tmp[j,2] = np.min(table[np.where(table[:,0] == B_lst[j])][:,1])
        
    return tmp

model_name_lst = os.listdir(model_dir)
ckt = Pick_ckt(model_name_lst)
for k in range(len(ckt)):
    idx = int(ckt[k,1])
    copyfile(model_dir + '/' + model_name_lst[idx], save_dir + '/' + model_name_lst[idx])
#%%
shutil.rmtree(model_dir)
#os.rmdir(model_dir)
os.rename(save_dir, model_dir)
os.makedirs(save_dir)


