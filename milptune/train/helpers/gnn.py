from typing import Tuple
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch
import torch.nn
import torch_geometric as tg
import torch.nn.functional as F



class InstanceEmbeddor(torch.nn.Module):
    def __init__(self, config_dim, n_gnn_layers=4, gnn_hidden_dim=64):
        super(InstanceEmbeddor, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=n_gnn_layers,
            hidden_dim=(gnn_hidden_dim, gnn_hidden_dim),
        )
        self.config_emb = ConfigEmbedding(in_dim=config_dim, out_dim=8)
        self.regression_head = FCLayers(
            in_dim=4 * gnn_hidden_dim, hidden_dim=8 * gnn_hidden_dim, out_dim=config_dim
        )

    def forward(self, instance_batch):
        graph_embedding = self.milp_gnn(instance_batch)
        x = self.regression_head(graph_embedding)
        return x


class MilpGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: Tuple[int, int],
        n_gnn_layers,
    ):
        super(MilpGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers

        self.gnns = torch.nn.ModuleList(
            [
                GNNFwd(
                    in_dim=(9, 1),
                    out_dim=hidden_dim,
                    batch_norm=True,
                )
            ]
            + [
                GNNFwd(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    batch_norm=True,
                )
                for i in range(self.n_gnn_layers - 1)
            ]
        )

        self.max_pool = tg.nn.global_max_pool

        self.attention_layer_var = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim[0], hidden_dim[0] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[0] // 2, 1),
        )
        self.attention_pool_var = tg.nn.GlobalAttention(self.attention_layer_var)

        self.attention_layer_cstr = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim[1], hidden_dim[1] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[1] // 2, 1),
        )
        self.attention_pool_cstr = tg.nn.GlobalAttention(self.attention_layer_cstr)

    def forward(self, x):
        for l in self.gnns:
            x = l(x)

        # Max pooling
        x_var_emb = self.max_pool(x.var_feats, x.var_batch_el)
        x_cstr_emb = self.max_pool(x.cstr_feats, x.cstr_batch_el)
        # Attention pooling
        x_var_emb2 = self.attention_pool_var(x.var_feats, x.var_batch_el)
        x_cstr_emb2 = self.attention_pool_cstr(x.cstr_feats, x.cstr_batch_el)
        x = torch.cat([x_var_emb, x_cstr_emb, x_var_emb2, x_cstr_emb2], axis=-1)
        return x


class GNNFwd(torch.nn.Module):
    def __init__(
        self,
        in_dim: Tuple[int, int],
        out_dim: Tuple[int, int],
        residual=False,
        batch_norm=True,
        additional_dense=False,
    ):
        super(GNNFwd, self).__init__()
        self.Conv = tg.nn.GraphConv

        self.additional_dense = additional_dense
        if additional_dense:
            self.node_encoder = torch.nn.Sequential(torch.nn.Linear(in_dim[0], in_dim[0]), torch.nn.ReLU())
            self.cstr_encoder = torch.nn.Sequential(torch.nn.Linear(in_dim[1], in_dim[1]), torch.nn.ReLU())

        # We're using 'mean' aggregation, because I've seen one single comparison and 'mean' had better performance than 'add'
        # Table 8 (Appendix) of Bengio et al: Benchmarking GNNs: https://arxiv.org/pdf/2003.00982.pdf
        self.node_gnn = self.Conv(
            in_channels=in_dim[::-1],
            out_channels=out_dim[0],
            aggr="mean",
        )
        self.cstr_gnn = self.Conv(
            in_channels=in_dim,
            out_channels=out_dim[1],
            aggr="mean",
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.node_batch_norm = tg.nn.BatchNorm(in_channels=in_dim[0])
            self.cstr_batch_norm = tg.nn.BatchNorm(in_channels=in_dim[1])

        self.residual = residual
        if self.residual:
            assert in_dim == out_dim, "For residual layers, in_dim and out_dim have to match"

    def forward(self, data):
        x_node = data.var_feats
        x_cstr = data.cstr_feats

        if self.batch_norm:
            x_node = self.node_batch_norm(x_node)
            x_cstr = self.cstr_batch_norm(x_cstr)

        if self.additional_dense:
            x_node = self.node_encoder(x_node)
            x_cstr = self.cstr_encoder(x_cstr)

        edge_attr = data.edge_attr.unsqueeze(-1)

        x_node_ = self.node_gnn(x=(x_cstr, x_node), edge_index=data.edge_index, edge_weight=edge_attr)
        # x_node_ = checkpoint(self.node_gnn, {'x': (x_cstr, x_node), 'edge_index': data.edge_index, 'edge_weight':edge_attr})
        x_cstr_ = self.cstr_gnn(
            x=(x_node, x_cstr),
            edge_index=data.edge_index.flip(-2),
            edge_weight=edge_attr,
        )

        if self.residual:
            x_node_ = x_node_ + x_node
            x_cstr_ = x_cstr_ + x_cstr

        x_node_ = F.relu(x_node_)
        x_cstr_ = F.relu(x_cstr_)

        assert x_node.shape[-2] == x_node_.shape[-2]
        assert x_cstr.shape[-2] == x_cstr_.shape[-2]

        data.var_feats = x_node_
        data.cstr_feats = x_cstr_

        return data


class ConfigEmbedding(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=8):
        super(ConfigEmbedding, self).__init__()
        if not hidden_dim:
            hidden_dim = 64

        self.input_batch_norm = torch.nn.BatchNorm1d(in_dim)
        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        return x


class FCLayers(torch.nn.Module):
    def __init__(self, in_dim=2 * 8, hidden_dim=None, out_dim=2):
        super(FCLayers, self).__init__()
        if not hidden_dim:
            hidden_dim = 4 * in_dim

        self.out_dim = out_dim

        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)


    def forward(self, x):
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()

        return x
