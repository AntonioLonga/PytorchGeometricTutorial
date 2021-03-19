from torch.nn import Parameter, Module, Sigmoid
import torch
import torch_scatter
import torch.nn.functional as F

class AbstractLAFLayer(Module):
    def __init__(self, **kwargs):
        super(AbstractLAFLayer, self).__init__()
        assert 'units' in kwargs or 'weights' in kwargs
        if 'device' in kwargs.keys():
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ngpus = torch.cuda.device_count()
        
        if 'kernel_initializer' in kwargs.keys():
            assert kwargs['kernel_initializer'] in [
                'random_normal',
                'glorot_normal',
                'he_normal',
                'random_uniform',
                'glorot_uniform',
                'he_uniform']
            self.kernel_initializer = kwargs['kernel_initializer']
        else:
            self.kernel_initializer = 'random_normal'

        if 'weights' in kwargs.keys():
            self.weights = Parameter(kwargs['weights'].to(self.device), \
                                     requires_grad=True)
            self.units = self.weights.shape[1]
        else:
            self.units = kwargs['units']
            params = torch.empty(12, self.units, device=self.device)
            if self.kernel_initializer == 'random_normal':
                torch.nn.init.normal_(params)
            elif self.kernel_initializer == 'glorot_normal':
                torch.nn.init.xavier_normal_(params)
            elif self.kernel_initializer == 'he_normal':
                torch.nn.init.kaiming_normal_(params)
            elif self.kernel_initializer == 'random_uniform':
                torch.nn.init.uniform_(params)
            elif self.kernel_initializer == 'glorot_uniform':
                torch.nn.init.xavier_uniform_(params)
            elif self.kernel_initializer == 'he_uniform':
                torch.nn.init.kaiming_uniform_(params)
            self.weights = Parameter(params, \
                                     requires_grad=True)
        e = torch.tensor([1,-1,1,-1], dtype=torch.float32, device=self.device)
        self.e = Parameter(e, requires_grad=False)
        num_idx = torch.tensor([1,1,0,0], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.num_idx = Parameter(num_idx, requires_grad=False)
        den_idx = torch.tensor([0,0,1,1], dtype=torch.float32, device=self.device).\
                                view(1,1,-1,1)
        self.den_idx = Parameter(den_idx, requires_grad=False)
        

class LAFLayer(AbstractLAFLayer):
    def __init__(self, eps=1e-7, **kwargs):
        super(LAFLayer, self).__init__(**kwargs)
        self.eps = eps
    
    def forward(self, data, index, dim=0, **kwargs):
        eps = self.eps
        sup = 1.0 - eps 
        e = self.e

        x = torch.clamp(data, eps, sup)
        x = torch.unsqueeze(x, -1)
        e = e.view(1,1,-1)        

        exps = (1. - e)/2. + x*e 
        exps = torch.unsqueeze(exps, -1)
        exps = torch.pow(exps, torch.relu(self.weights[0:4]))

        scatter = torch_scatter.scatter_add(exps, index.view(-1), dim=dim)
        scatter = torch.clamp(scatter, eps)

        sqrt = torch.pow(scatter, torch.relu(self.weights[4:8]))
        alpha_beta = self.weights[8:12].view(1,1,4,-1)
        terms = sqrt * alpha_beta

        num = torch.sum(terms * self.num_idx, dim=2)
        den = torch.sum(terms * self.den_idx, dim=2)
        
        multiplier = 2.0*torch.clamp(torch.sign(den), min=0.0) - 1.0

        den = torch.where((den < eps) & (den > -eps), multiplier*eps, den)

        res = num / den
        return res

