import torch.nn as nn
import torch

class my_round_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

myroundfunction = my_round_func.apply
    
class BinAttack(nn.Module):
    def __init__(self, train_model, n_node, lam, train_idx, test_idx, device):
        super(BinAttack, self).__init__()
        self.model = train_model
        self.n = n_node
        self.lam= lam
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.device = device
        self.Z_continue = nn.Parameter(0.495 * torch.ones((int(0.5*self.n*(self.n-1)),1)).to(self.device))
    
    def sparse_matrix_mul(self, A1, A2):
        #n1 = len(A1)
        #k1 = A1.shape[1]
        A1_sp = A1.to_sparse()
        #A1_sp = torch.sparse_coo_tensor(A1_sp.indices(), A1_sp.values(), size=[n1,k1])
        #n2= len(A2)
        #k2 = A2.shape[1]
        A2_sp = A2.to_sparse()
        #A2_sp = torch.sparse_coo_tensor(A2_sp.indices(), A2_sp.values(), size=[n2,k2])
        return torch.sparse.mm(A1_sp, A2_sp).to_dense()
        
    def perturb(self, tri):
        Z = -2 * myroundfunction(self.Z_continue) + 1
        tri_perturb = tri.clone()
        tri_perturb[:,2:] = (tri[:,2:]-0.5) * Z + 0.5
        return tri_perturb
    
    def inner_train(self, x, tri, y_train):
        self.model.reset_parameters()
        #tri_mod = self.perturb(tri)
        _ = self.model.RWLS(x, tri, y_train)
    
    def forward(self, x, triple, label):
        tri_mod = self.perturb(triple)
        y_train = label[self.train_idx]
        
        self.inner_train(x, tri_mod, y_train)
        
        H = self.model.feat(x, tri_mod)
        outputs = torch.sigmoid(torch.mm(H, self.model.weight) + self.model.bias)
        outputs_train = outputs[self.train_idx]
        outputs_test = outputs[self.test_idx]
        return outputs_train, outputs_test