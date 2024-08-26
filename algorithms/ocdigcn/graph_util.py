#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import warnings
warnings.filterwarnings("ignore")
from typing import Callable, List, Optional
import scipy
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_scatter import scatter_add
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import coalesce, add_self_loops

def cat(seq):
    ##define a function to combine items into sequences
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def split(data, batch):
    ##define a funtion to split data into batches
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    ##define a slices
    slices = {'edge_index': edge_slice}

    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()

    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice

    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    ##for second-order index
    if data.edge_index2 is not None:
        row2, _ = data.edge_index2
        edge_slice2 = torch.cumsum(torch.from_numpy(np.bincount(batch[row2])), 0)
        edge_slice2 = torch.cat([torch.tensor([0]), edge_slice2])

        # Edge indices should start at zero for every graph.
        data.edge_index2 -= node_slice[batch[row2]].unsqueeze(0)

        ##define a slices
        slices['edge_index2'] = edge_slice2
        slices['edge_attr2'] = edge_slice2

    return data, slices

##IMPORTANT function 1: define a function to read data from text files
def read_graph_data(folder):
    # read edge index from adj matrix
    edge_index = read_file(folder, 'A', torch.long).t() - 1    ##first order adj matrix
    edge_index2 = read_file(folder, 'A2', torch.long).t() - 1   ##second order adj matrix

    batch = read_file(folder, 'graph_indicator', torch.long) - 1   # read graph index
    node_attributes = torch.empty((batch.size(0), 0))   # read node attributes
    node_attributes = read_file(folder, 'node_attributes', torch.float32)

    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    # read edge attributes
    edge_attributes = torch.empty((edge_index.size(1), 0))    ##first-order edge attributes
    edge_attributes = read_file(folder, 'edge_attributes')
    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)

    edge_attributes2 = torch.empty((edge_index2.size(1), 0))     ##second-order edge attributes
    edge_attributes2 = read_file(folder, 'edge_attributes2')
    if edge_attributes2.dim() == 1:
        edge_attributes2 = edge_attributes2.unsqueeze(-1)

    # concategate node attributes
    x = cat([node_attributes])

    # concategate edge attributes and edge lables
    edge_attr = cat([edge_attributes])    ##first-order edge attributes
    edge_attr2 = cat([edge_attributes2])    ##second-order edge attributes

    y = read_file(folder, 'graph_labels', torch.long)   # read graph attributes or graph labels

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)    # get total number of nodes for all graphs

    # remove self-loops: we should not remove selfloops
    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)     ##first-order
    # edge_index2, edge_attr2 = remove_self_loops(edge_index2, edge_attr2)
    edge_index2, edge_attr2 = coalesce(edge_index2, edge_attr2, num_nodes)    ##second-order

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_index2 = edge_index2
    data.edge_attr2 = edge_attr2

    data, slices = split(data, batch)

    sizes = {
        'num_node_attributes': node_attributes.size(-1),
        'num_edge_attributes': edge_attributes.size(-1),
        'num_edge_attributes2': edge_attributes2.size(-1)
    }

    return data, slices, sizes

class ParseDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 cleaned: bool = False):
        self.root = root
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)  # processed() function called

        load_data = torch.load(self.processed_paths[0])

        self.data, self.slices, self.sizes = load_data

        num_node_attributes = self.num_node_attributes
        self.data.x = self.data.x[:, :num_node_attributes]

        num_edge_attrs = self.num_edge_attributes
        self.data.edge_attr = self.data.edge_attr[:, :num_edge_attrs]

        num_edge_attrs2 = self.num_edge_attributes2
        self.data.edge_attr2 = self.data.edge_attr2[:, :num_edge_attrs2]

    @property
    def raw_dir(self) -> str:
        name = f'/Raw{"_cleaned" if self.cleaned else ""}'
        return self.root+self.name+name

    @property
    def processed_dir(self) -> str:
        name = f'/processed{"_cleaned" if self.cleaned else ""}'
        return self.root+self.name+name

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def num_edge_attributes2(self) -> int:
        return self.sizes['num_edge_attributes2']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        folder = self.root + self.name + '/processed'
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.data, self.slices, sizes = read_graph_data(self.raw_dir)

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        torch.save((self.data, self.slices, sizes), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight == None:    #if edge_weight is not given
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    else:
        edge_weight = torch.flatten(edge_weight)

    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1.0)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes + 1, num_nodes + 1]))
    p_v[0:num_nodes, 0:num_nodes] = (1 - alpha) * p_dense
    p_v[num_nodes, 0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes, num_nodes] = alpha
    p_v[num_nodes, num_nodes] = 0.0
    p_ppr = p_v

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(), left=True, right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:, ind[0]]  # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi / pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi < 0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight == None:    # if edge_weight is not given
        edge_weight = torch.ones((edge_index.size(1)), dtype=dtype, device=edge_index.device)
    else:
        edge_weight = torch.flatten(edge_weight)

    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes, num_nodes])).to_dense()

    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())

    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L, as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def ConcatGraphs(ReadGraph, graph_count_index, my_node_accum, new_node_accum, raw_path):
    # 1. Adj Matrix
    fp_A = raw_path + "/A.txt"
    df_A = pd.DataFrame(ReadGraph.edge_index.numpy()).T
    df_A = df_A + my_node_accum
    with open(fp_A, "ab") as f:
        np.savetxt(f, df_A.values, fmt='%i', delimiter=', ')

    # 2. Edge-weight Matrix
    fp_edge_weight = raw_path + "/edge_attributes.txt"
    df_edge_weight = pd.DataFrame(ReadGraph.edge_attr.numpy())
    with open(fp_edge_weight, "ab") as f:
        np.savetxt(f, df_edge_weight.values, fmt='%f', delimiter=', ')

    # 3. Graph-indicator Matrix
    df_graph_indicator = pd.DataFrame(columns=["indicator"])
    df_graph_indicator["indicator"] = [graph_count_index + 1] * len(new_node_accum)
    fp_graph_indicator = raw_path + "/graph_indicator.txt"
    with open(fp_graph_indicator, "ab") as f:
        np.savetxt(f, df_graph_indicator.values, fmt='%i', delimiter=', ')

    # 4. Graph-labels Matrix
    ##use the anomaly_label.csv file to generate this matrix
    fp_graph_labels = raw_path + "/graph_labels.txt"
    df_graph_labels = pd.DataFrame([ReadGraph.y.numpy()])
    with open(fp_graph_labels, "ab") as f:
        np.savetxt(f, df_graph_labels.values, fmt='%i', delimiter=', ')

    # 5. Node-attributes Matrix
    fp_node_attributes = raw_path + "/node_attributes.txt"
    df_node_attributes = pd.DataFrame(ReadGraph.x.numpy())
    with open(fp_node_attributes, "ab") as f:
        np.savetxt(f, df_node_attributes.values, fmt='%f', delimiter=', ')

    # 6. Second-order Adj Matrix
    fp_A2 = raw_path + "/A2.txt"
    df_A2 = pd.DataFrame(ReadGraph.edge_index2.numpy()).T
    df_A2 = df_A2 + my_node_accum
    with open(fp_A2, "ab") as f:
        np.savetxt(f, df_A2.values, fmt='%i', delimiter=', ')

    # 7. Second-order Edge-weight Matrix
    fp_edge_weight2 = raw_path + "/edge_attributes2.txt"
    df_edge_weight2 = pd.DataFrame(ReadGraph.edge_attr2.numpy())
    with open(fp_edge_weight2, "ab") as f:
        np.savetxt(f, df_edge_weight2.values, fmt='%f', delimiter=', ')

def read_file(folder, name, dtype=None):
    path = os.path.join(folder, f'{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def read_and_valid_graph_data(folder, adj_type):
    edge_index = read_file(folder, 'A', torch.long).t() - 1
    batch = read_file(folder, 'graph_indicator', torch.long) - 1

    if batch.dim() == 0:  ## the batch looks like ->tensor(42), which is zero dimension
        node_attributes = torch.empty((1, 0))

    else:  ## the batch looks like ->tensor([41, 41, 41, 41, 41, 41]), which is one dimension
        node_attributes = torch.empty((batch.size(0), 0))
    node_attributes = read_file(folder, 'node_attributes')
    if node_attributes.dim() == 1:
        node_attributes = node_attributes.unsqueeze(-1)

    is_empty_index = 0

    if len(edge_index.shape) == 1:  ##some graph only have a single row
        is_empty_index = 1
        data = Data()
        return data, is_empty_index  ##if it is empty, we skip this dataset

    ##some graphs only have a single node, we should skip those graphs?
    edge_attributes = torch.empty((edge_index.size(1), 0))
    edge_attributes = read_file(folder, 'edge_attributes')

    if edge_attributes.dim() == 1:
        edge_attributes = edge_attributes.unsqueeze(-1)

    x = cat([node_attributes])       # concategate node attributes

    if edge_index.size(1) == 1:  ##some graph only have a single row, this causes tensor with 0 dimension
        edge_attr = torch.tensor([[edge_attributes.item()]])
    else:
        edge_attr = cat([edge_attributes])

    # read graph attributes or graph labels
    y = None
    y = read_file(folder, 'graph_labels', torch.long)

    # get total number of nodes for all graphs
    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)

    # remove self-loops: we should not remove selfloops
    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_attr is None:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
    else:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

    # use get_adj to preprocess data: we should do this for each graph saparately
    # Processing first and second-order adj matrix
    alpha = 0.1
    indices = edge_index
    features = x

    edge_index, edge_attr = get_appr_directed_adj(alpha=alpha,
                                                  edge_index=indices,
                                                  num_nodes=features.shape[0],
                                                  dtype=features.dtype,
                                                  edge_weight=edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    ##we should input approximate edge_index, edge_attr or the original edge_index, edge_attr?
    edge_index2, edge_attr2 = get_second_directed_adj(edge_index=edge_index,
                                                      num_nodes=features.shape[0],
                                                      dtype=features.dtype,
                                                      edge_weight=edge_attr)

    data.edge_index2 = edge_index2
    data.edge_attr2 = edge_attr2

    return data, is_empty_index

def GraphConstruction(my_example_df, graph_count_index, new_node_accum, template_df, tempRaw_path):
    # #write a function to generate a graph from each group of log events and store them as
    # 1. A.txt
    # 2. edge_attributes.txt
    # 3. graph_indicator.txt
    # 4. graph_labels.txt
    # 5. node_attributes.txt
    # =============================================================================
    G = nx.MultiDiGraph()
    tidx_list = list(my_example_df["tidx"])
    node_list = list(dict.fromkeys(tidx_list))
    num_features = len(template_df) + 1

    G.add_nodes_from(node_list)
    G.add_edges_from([(tidx_list[v], tidx_list[v + 1]) for v in range(len(tidx_list) - 1)])

    ##get adjacency matrix in the form of sparse matrix
    if G.nodes:
        A = nx.adjacency_matrix(G)
        # 1. Adj Matrix: done, by handling it with get_adj in DIGCN
        df_A = pd.DataFrame(columns=["row", "column"])
        row_vec = (list(A.nonzero())[0]).tolist()
        col_vec = (list(A.nonzero())[1]).tolist()
        row_vec = [a + 1 for a in row_vec]
        col_vec = [a + 1 for a in col_vec]
        df_A["row"] = row_vec
        df_A["column"] = col_vec

        fp_A = tempRaw_path + "/A.txt"
        np.savetxt(fp_A, df_A.values, fmt='%i', delimiter=', ')

        # 2. Edge-weight Matrix: done
        df_edge_weight = pd.DataFrame(columns=["edge_weight"])
        df_edge_weight["edge_weight"] = list(A.data)
        fp_edge_weight = tempRaw_path + "/edge_attributes.txt"
        np.savetxt(fp_edge_weight, df_edge_weight.values, fmt='%i', delimiter=', ')

        # 3. Graph-indicator Matrix: done
        df_graph_indicator = pd.DataFrame(columns=["indicator"])
        df_graph_indicator["indicator"] = [graph_count_index + 1] * len(new_node_accum)
        fp_graph_indicator = tempRaw_path + "/graph_indicator.txt"
        np.savetxt(fp_graph_indicator, df_graph_indicator.values, fmt='%i', delimiter=', ')

        # 4. Graph-labels Matrix: done, by modifing the train/test split code in GLAM
        ##use the anomaly_label.csv file to generate this matrix
        label_value = 0
        df_graph_labels = pd.DataFrame(columns=["labels"])
        # df_graph_labels["labels"] = [label_value]*len(list(A.data))
        df_graph_labels["labels"] = [label_value]
        fp_graph_labels = tempRaw_path + "/graph_labels.txt"
        np.savetxt(fp_graph_labels, df_graph_labels.values, fmt='%i', delimiter=', ')

        # 5. Node-attributes Matrix: by retrieving semantic embedding vec from embedding_df dataframe
        node_attr_list = []
        for node_id in node_list:  ##this is important, we must keep the order of nodes
            # one-hot template idxs
            arr_vec = [1 if i == node_id else 0 for i in range(1, num_features + 1)]
            node_attr_list.append(arr_vec)

        df_node_attributes = pd.DataFrame(node_attr_list)
        fp_node_attributes = tempRaw_path + "/node_attributes.txt"
        np.savetxt(fp_node_attributes, df_node_attributes.values, fmt='%f', delimiter=', ')
        return True
    else:
        False

def get_daily_samping_idxs(df, graph_limit):
    result_idxs = []
    df['day'] = df['time'].values.astype('<M8[D]')
    day_grouped = df.groupby('day')
    daily_selection_cnt = int(graph_limit/day_grouped.ngroups)
    for k, v in day_grouped.indices.items():
        result_idxs.extend(np.random.choice(v, daily_selection_cnt))

    return result_idxs

def generate_graph_data(raw_df, template_df, graph_con_num, model_dir, core_num, train=False):
    torch.set_num_threads(core_num)
    tempRaw_path, raw_path = model_dir+'/TempRaw', model_dir+'/Raw'
    if not os.path.exists(tempRaw_path) and not os.path.exists(raw_path):
        os.makedirs(model_dir+'/TempRaw')
        os.makedirs(model_dir+'/Raw')

    all_event_list, graph_idxs, count_index, graph_limit = [], [], 0, 30000
    max_g = min(len(raw_df) - graph_con_num + 1, graph_limit)
    if train:
        idx_list = get_daily_samping_idxs(raw_df, max_g)

    else:
        idx_list = list(range(max_g))

    for i in idx_list:
        example_df = raw_df[i: i + graph_con_num]
        node_accum = max(len(all_event_list) + 1, 1)
        new_event_list = list(dict.fromkeys(example_df["tidx"]))

        success_yn = GraphConstruction(my_example_df=example_df,
                                       graph_count_index=count_index,
                                       new_node_accum=new_event_list,
                                       template_df=template_df,
                                       tempRaw_path=tempRaw_path)

        ##after generating each graph, we get its appr adj matrix accordingly
        MyReadGraph, empty_index = read_and_valid_graph_data(folder=tempRaw_path, adj_type='ib')

        if empty_index == 0:
            ##oncatenate all appr graphs (only none-empty graphs)
            ConcatGraphs(ReadGraph=MyReadGraph,
                         graph_count_index=count_index,
                         my_node_accum=node_accum,
                         new_node_accum=new_event_list,
                         raw_path=raw_path)

            all_event_list += new_event_list
            count_index += 1
            if success_yn:
                graph_idxs.append(i + graph_con_num - 1)

    return graph_idxs