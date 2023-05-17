import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential

def get_activation(name='ReLU'):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        raise NotImplementedError("Acitivation {} not implemented!".format(name))
        

class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_name="gcn", act_name="ReLU", batchnorm=True) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.batchnorm = None

        self.layer = self.get_layer(layer_name)
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_dim)
        self.act = get_activation(act_name)

    def reset_parameters(self):
        self.layer.reset_parameters()
        if self.batchnorm is not None:
            self.batchnorm.reset_parameters()

    def get_layer(self, name="GCN"):
        if name == "GCN":
            return GCNConv(in_channels=self.in_dim, out_channels=self.out_dim)
        else:
            raise NotImplementedError("Layer {} not implemented!".format(name))
    
    def forward(self, x, egde_index):
        x = self.layer(x, egde_index)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        
        return self.act(x)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dims, args):
        super().__init__()

        dims = [in_dim] + hid_dims
        assert len(dims) >= 2

        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(ConvLayer(in_dim, out_dim, args.layer_name, args.act_name, args.batchnorm))

    def forward(self, x, edge_index):
        outputs = []
        for layer in self.layers:
            x = layer(x.detach(), edge_index)
            # x = layer(x, edge_index)
            outputs.append(x)

        return outputs
    
    @torch.no_grad()
    def embeds(self, x, edge_index):
        for layer in self.layers:
            x = layer(x.detach(), edge_index)

        return x
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    @torch.no_grad()
    def get_embeding(self, data):
        self.eval()
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
        return x