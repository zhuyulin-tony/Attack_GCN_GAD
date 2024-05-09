import torch.nn as nn
import torch

class Surrogate_GNN(nn.Module):
    def __init__(self, n_node, feat_dim, out_dim, train_idx, test_idx, xi, device):
        super().__init__()
        self.f_dim = feat_dim
        self.out_dim = out_dim
        self.n = n_node
        self.device = device
        self.xi = xi
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.weight = nn.Parameter(torch.FloatTensor(self.f_dim, self.out_dim).to(self.device))
        self.bias = nn.Parameter(torch.FloatTensor(self.out_dim).to(self.device))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def sparse_matrix_mul(self, A1, A2):
        return torch.sparse.mm(A1.to_sparse(), A2.to_sparse()).to_dense()
    
    def adjacency_matrix(self, tri):
        A = torch.sparse_coo_tensor(tri[:,:2].T, tri[:,2], size=[self.n,self.n]).to_dense()
        A = A + A.T - torch.diag(torch.diag(A)) # symmetric.
        return A
    
    def normalize_adj(self, adj):
        adj_1 = adj + torch.eye(adj.shape[0]).to(self.device)
        D_05 = torch.diag(1/torch.sqrt(adj_1.sum(0)))
        return torch.mm(torch.mm(D_05, adj_1), D_05)
        #return D_05 @ adj_1 @ D_05
        #return self.sparse_matrix_mul(self.sparse_matrix_mul(D_05, adj_1), D_05)
    
    def feat(self, x, tri):
        self.A_perturb = self.adjacency_matrix(tri)
        self.A_perturb_norm = self.normalize_adj(self.A_perturb)
        #A2 = self.sparse_matrix_mul(self.A_perturb_norm, self.A_perturb_norm)
        A2 = torch.linalg.matrix_power(self.A_perturb_norm, 2)
        #H = self.sparse_matrix_mul(A2, x)
        H = torch.mm(A2, x)
        return H
    
    # Ridge regression to penalize high dimensional attributes and prevent singular matrix in matrix inverse.
    def Ridge(self, x, tri, y_train):
        H = self.feat(x, tri)
        H_train = H[self.train_idx]
        H1 = torch.cat((torch.ones((len(H_train),1)).to(self.device), H_train), 1)
        I = torch.eye(x.shape[1]+1).to(self.device)
        theta = torch.linalg.inv((H1.T @ H1) + self.xi*I) @ H1.T @ (y_train.reshape(-1,1))
        self.bias = nn.Parameter(theta[:1].reshape(1))
        self.weight = nn.Parameter(theta[1:].reshape(self.f_dim,1))
        return torch.mm(H_train, self.weight) + self.bias
    
    def BCE_loss_with_weight(self, output, label, pos_weight):
        cost = (1/len(label))*(pos_weight*((-label).t() @ torch.log(output.reshape(-1,)+1e-10))-((1-label).t() @ torch.log(1-output.reshape(-1,)+1e-10)))
        return cost
    
    # Weighted Ridge regression. beta_hat = (X^{T}WX+xi*I)^{-1}(X^{T}WY. I_{pp}.
    def RWLS(self, x, tri, y_train):
        H = self.feat(x, tri)
        H_train = H[self.train_idx]
        H1 = torch.cat((torch.ones((len(H_train),1)).to(self.device), H_train), 1)
        pos_weight = ((y_train==0).sum()/(y_train==1).sum()).item()
        weight_tensor = torch.ones_like(y_train.detach().clone())
        weight_tensor[y_train==1]=pos_weight
        weight_tensor = torch.diag(weight_tensor)
        I = torch.eye(x.shape[1]+1).to(self.device)
        theta = torch.linalg.inv((H1.T @ weight_tensor @ H1) + self.xi*I) @ H1.T @ weight_tensor @ (y_train.reshape(-1,1))
        self.bias = nn.Parameter(theta[:1].reshape(1))
        self.weight = nn.Parameter(theta[1:].reshape(self.f_dim,1))
        return torch.mm(H_train, self.weight) + self.bias
    
    def forward(self, x, tri, idx):
        H = self.feat(x, tri)
        H_sub = H[idx]
        if self.out_dim == 2:
            return torch.nn.functional.log_softmax(torch.mm(H_sub, self.weight) + self.bias, dim=1) 
        elif self.out_dim == 1:
            return torch.sigmoid(torch.mm(H_sub, self.weight) + self.bias)
    
    def gradient_descent(self, x, tri, y, lr, epochs):
        pos_weight = ((y[self.train_idx]==0).sum()/(y[self.train_idx]==1).sum()).item()
        for epoch in range(epochs):
            outputs = self.forward(x, tri, self.train_idx)
            loss = self.BCE_loss_with_weight(outputs, y[self.train_idx], pos_weight=pos_weight)
            loss.backward()
            
            self.weight.data -= lr * self.weight.grad.data
            self.bias.data -= lr * self.bias.grad.data
            
            print('epoch:', epoch, 'loss:', loss.item())