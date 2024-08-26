import pandas as pd
import numpy as np
import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import DataLoader

class DIGCNConv(MessagePassing):
    r"""
    The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache the adj matrix on first execution,
                                and will use the cached version for further executions.
                                Please note that, all the normalized adj matrices (including undirected)
                                are calculated in the dataset preprocessing to reduce time comsume.
                                This parameter should only be set to :obj:`True` in transductive learning scenarios.
                                (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn an additive bias.
                                (default: :obj:`True`)
        **kwargs (optional): Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=False, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    def forward(self, x, edge_index, edge_attr):
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_attr is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please obtain the adj matrix in preprocessing.')
            else:
                norm = edge_attr
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class MeanTrainer:
    def __init__(self, logger, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
        self.logger = logger
        self.device = device
        self.model = model
        self.optimizer = optimizer

        ##--parameters for OCSVDD objectives----##
        self.center = None
        self.reg_weight = 0
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer

    def train(self, train_loader):
        self.model.to(self.device)
        self.model.train()
        if self.center == None:    # first iteration, define s list to store vectors for computing SVDD center
            F_list = []

        svdd_loss_accum, total_iters = 0, 0

        for batch in train_loader:
            batch = batch.cuda()
            train_embeddings = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            mean_train_embeddings = [torch.mean(emb, dim=0) for emb in train_embeddings]  # use mean Readout to obtain graph embeddings
            F_train = torch.stack(mean_train_embeddings)

            # if first iteration, store vectors for computing SVDD center, and do not perform any backprop
            if self.center == None:
                F_list.append(F_train)

            else:    # if not first iteration, perform backprop
                train_scores = torch.sum((F_train - self.center) ** 2, dim=1)
                # the second term in SVDD objective is controled by regularizer automatically
                svdd_loss = torch.mean(train_scores)

                # backpropagate
                self.optimizer.zero_grad()
                svdd_loss.backward()
                self.optimizer.step()
                svdd_loss_accum += svdd_loss
                total_iters += 1

        if self.center == None:  # first epoch only, compute SVDD center
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach()  # no backpropagation for center
            average_svdd_loss = -1
        else:    # if not first epoch, compute averaged SVDD loss ----##
            average_svdd_loss = svdd_loss_accum / total_iters

        return average_svdd_loss, self.center

    def test(self, test_loader):
        self.model.to(self.device)
        test_svdd_loss_accum, total_iters = 0, 0
        dist_list = []

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.cuda()
                test_embeddings = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                for batch_idx in range(len(test_embeddings)):
                    dist_list += [float(torch.sum((test_embeddings[batch_idx][i] - self.center) ** 2)) for i in range(len(test_embeddings[batch_idx]))]

                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings]
                F_test = torch.stack(mean_test_embeddings)
                batch_dists = torch.sum((F_test - self.center) ** 2, dim=1)

                test_svdd_loss = torch.mean(batch_dists)
                test_svdd_loss_accum += test_svdd_loss
                total_iters += 1

            average_svdd_loss = test_svdd_loss_accum / total_iters

        return average_svdd_loss, dist_list

    def calculate_score_threshold(self, test_dataset, log_df, g_idxs):
        test_df = pd.concat([log_df[:g_idxs[0]], log_df.iloc[g_idxs]])
        tidx_list = list(test_df['tidx'])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

        decom_df = pd.DataFrame(columns=['eventID', 'score_list'])
        decom_df['eventID'] = tidx_list
        idx = 0

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                try:
                    test_embeddings = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    '''graph decomposition'''
                    input_template_idxs = tidx_list[g_idxs[idx] - 29: g_idxs[idx] + 1]
                    node_list = list(dict.fromkeys(input_template_idxs))

                    score_dic = {}  # score_dic = {tidx_6: score_6, tidx_23: score_23, ...}
                    for i in range(len(node_list)):
                        score = torch.sum((test_embeddings[0][i] - self.center) ** 2)
                        score_dic[node_list[i]] = round(float(score), 5)

                    # stack score to input data
                    input_idxs = np.arange(g_idxs[idx] - 29, g_idxs[idx] + 1)
                    for i in range(len(input_idxs)):
                        if np.isnan(decom_df['score_list'].iloc[input_idxs[i]]).all():
                            decom_df['score_list'].iloc[input_idxs[i]] = [score_dic[input_template_idxs[i]]]
                        else:
                            decom_df['score_list'].iloc[input_idxs[i]].append(score_dic[input_template_idxs[i]])
                    idx += 1
                except Exception as e:
                    self.logger.debug(e)
                    idx += 1
        decom_df['decom_avg_score'] = decom_df['score_list'].apply(np.mean)

        sigma = 3
        anomaly_threshold = decom_df['decom_avg_score'].mean() + sigma * decom_df['decom_avg_score'].std()

        return round(anomaly_threshold, 5)


class DiGCN(nn.Module):
    def __init__(self, nfeat, nhid, nlayer, dropout=0, bias=False, **kwargs):
        ##two layers
        super(DiGCN, self).__init__()
        self.conv1 = DIGCNConv(nfeat, nhid)
        self.conv2 = DIGCNConv(nhid, nhid)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, edge_attr, g_node_list):
        num_graphs = len(set(g_node_list.tolist()))
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        emb_list = []
        for g in range(num_graphs):
            emb = x[g_node_list == g]
            emb_list.append(emb)
        return emb_list