import os, shutil
import time
import joblib
import psutil
import warnings
warnings.filterwarnings("ignore")
from types import SimpleNamespace
import math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from algorithms import aimodel
from algorithms.ocdigcn.graph_util import generate_graph_data, ParseDataset
from algorithms.ocdigcn.DIGCNConv import MeanTrainer, DiGCN
from common import constants, aicommon

class OCDiGCN(aimodel.AIModel):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        #torch graph model
        self.algo_name = constants.MODEL_S_DIGCN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.center = None
        self.num_features = None
        self.h_dims = 128
        self.h_layers = 2
        self.batch_size = 64
        self.lr = 0.01
        self.weight_decay = 1e-4
        self.epochs = 100
        self.graph_con_num = 30

        drain_config = TemplateMinerConfig()
        drain_config.load('./algorithms/logseq/config/drain3.ini')
        self.template_miner = TemplateMiner(config=drain_config)
        self.template_df = pd.DataFrame(columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.score_threshold = None  # distance from center (float value)
        self.cnt_threshold = None

    def set_template_df(self):
        for cluster in self.template_miner.drain.clusters:
            self.logger.info(cluster)
            self.template_df = self.template_df.append(pd.Series([cluster.cluster_id, ' '.join(cluster.log_template_tokens), cluster.size], index=self.template_df.columns), ignore_index=True)
        self.num_features = len(self.template_df) + 1

    def mining_template_cluster(self, log_df):
        msg_lines = list(log_df['msg'].values)
        self.logger.info(f"[DiGCN] drain3 Mining start")
        time_mining_s = time.time()
        for i, line in enumerate(msg_lines):
            self.template_miner.add_log_message(line.strip())
        self.logger.info(f"[DiGCN] Mining end (elapsed = {time.time() - time_mining_s:.2f}s)")

        self.set_template_df()

    def matching_template_idxs(self, log_df):
        self.logger.info(f"[DiGCN] drain3 Matching start")
        time_matching_s = time.time()
        msg_lines = list(log_df['msg'].values)
        tidx_list = [self.template_miner.match(msg).cluster_id if self.template_miner.match(msg) else self.num_features for msg in msg_lines]
        self.logger.info(f"[DiGCN] Matching end (elapsed = {time.time() - time_matching_s:.2f}s)")
        log_df['tidx'] = tidx_list
        self.logger.info(f"Matching Result >>>>>  \n{log_df}")
        return log_df

    def calculate_cnt_threshold(self, log_df, thre_3, dist_list):
        log_df['time_clean'] = log_df['time'].values.astype('<M8[m]')
        # daily_cnt = 해당 타겟의 일일 로그 발생량
        daily_cnt = math.ceil(log_df.groupby('time_clean')['msg'].count().values.mean())
        dist_df = pd.DataFrame(dist_list, columns=['dist'])
        dist_df['anomaly'] = dist_df['dist'] > thre_3
        try:
            total_anomaly_cnt = dist_df['anomaly'].value_counts().to_dict()[True]
            self.cnt_threshold = math.ceil(total_anomaly_cnt / (len(dist_df) / daily_cnt))
        except KeyError:
            self.cnt_threshold = math.ceil(daily_cnt/100)

    def fit(self, log_df, train_progress=None):
        if not os.path.isdir(os.path.join(self.config['model_dir'], f"{self.algo_name}")):
            os.makedirs(os.path.join(self.config['model_dir'], f"{self.algo_name}"))

        self.mining_template_cluster(log_df)
        log_df = self.matching_template_idxs(log_df)

        self.del_graph_data(train=True)

        time_graph_data_s = time.time()
        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, constants.MODEL_B_DIGCN, 10)

        core_num = max(int(psutil.cpu_count()/20), 1)
        self.logger.info(f"[DiGCN] start graph data generating. torch.set_num_threads({core_num})")
        generate_graph_data(log_df, self.template_df, self.graph_con_num, self.config['model_dir']+'/digcn/train', core_num, train=True)
        self.logger.info(f"[DiGCN] generating end (elapsed = {time.time() - time_graph_data_s:.2f}s)")
        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, constants.MODEL_B_DIGCN, 40)

        graph_dataset = ParseDataset(root=self.config['model_dir'], name='/digcn/train')
        train_dataset, test_dataset = train_test_split(graph_dataset, test_size=0.2, shuffle=False)
        self.logger.info(f"num_dataset ===== train: {len(train_dataset)}, test:{len(test_dataset)}")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0)

        self.model = DiGCN(nfeat=self.num_features, nhid=self.h_dims, nlayer=self.h_layers)

        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        early_stop = EarlyStopping(patience=5)
        trainer = MeanTrainer(logger=self.logger, model=self.model, optimizer=optimizer, device=self.device)

        epochinfo, dist_list = [], []
        t_start = time.time()
        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, constants.MODEL_B_DIGCN, 50)
        for epoch in range(1, self.epochs + 1):
            train_svdd_loss, self.center = trainer.train(train_loader=train_loader)
            test_svdd_loss, dist_list = trainer.test(test_loader=test_loader)
            dist_list += dist_list

            if epoch % 10 == 0:
                self.logger.info(f"Epoch: {epoch} train loss: {train_svdd_loss:.5f}, test loss: {test_svdd_loss:.5f}")
                aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, f"train {constants.MODEL_B_DIGCN} model", epoch)

            TEMP = SimpleNamespace()   # set a temporary object to store important information
            TEMP.epoch_no, TEMP.svdd_loss = epoch, train_svdd_loss
            epochinfo.append(TEMP)

            if train_svdd_loss > 0 and early_stop.early_stop(train_svdd_loss):
                break

        train_duration_t = round(time.time() - t_start, 3)
        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, constants.MODEL_B_DIGCN, 100, train_duration_t)

        time_cal_s = time.time()
        threshold_3 = round(np.array(dist_list).mean() + 3 * np.array(dist_list).std(), 5)
        threshold_4 = round(np.array(dist_list).mean() + 4 * np.array(dist_list).std(), 5)
        threshold_5 = round(np.array(dist_list).mean() + 5 * np.array(dist_list).std(), 5)
        self.calculate_cnt_threshold(log_df, threshold_3, dist_list)
        self.logger.info(f"cnt_thre: {self.cnt_threshold}, score_thre 3: {threshold_3}, 4: {threshold_4}, 5: {threshold_5}, elapsed time: {time.time() - time_cal_s}")

        best_svdd_idx = np.argmin([e.svdd_loss.detach().cpu().numpy() for e in epochinfo[1:]]) + 1

        train_metrics = {
            constants.MSG: {"svdd_loss": round(float(epochinfo[best_svdd_idx].svdd_loss), 5),
                            "deploy_policy": -1,
                            "hyper_params": {
                                'batch_size': self.batch_size,
                                'epochs': epoch,
                                'hidden_unit_size': self.h_dims,
                                'window_size': self.graph_con_num,
                                }
                            }
        }

        report = {
            "duration_time": train_duration_t,
            "mse": -1,
            "rmse": -1,
            "pre_mse": -1,
            "pre_rmse": -1,
            "deploy_policy": -1,
        }

        train_result = {"from_date": self.config['date'][0],
                        "to_date": self.config['date'][-1],
                        "train_metrics": train_metrics,
                        'anomaly_threshold': self.cnt_threshold,
                        'score_threshold': {'thre_3': threshold_3, 'thre_4': threshold_4, 'thre_5': threshold_5},
                        "except_failure_date_list": None,
                        "except_business_list": None,
                        "business_list": None,
                        "train_business_status": None,
                        "train_mode": -1,
                        "outlier_mode": -1,
                        "results": report
                        }
        self.logger.info(f"hypersphere's center vectors(dims: {len(self.center)}):  {self.center[:3]}...")
        self.logger.info(f"train result epoch:{epoch}, {constants.MSG},  elapsed: {time.time() - t_start:.2f}s")

        return train_result

    def save(self, model_dir):
        onnx_file_path = model_dir + f"/{self.algo_name}/digcn_model.onnx"
        dummy_data = (torch.randn(1, self.num_features), torch.randint(0, 1, (2, 2)),
                      torch.randn(1, 2), torch.randint(0, 1, (1,)))
        self.model = self.model.to('cpu')
        torch.onnx.export(self.model, dummy_data, onnx_file_path, opset_version=16,
                          input_names=['x', 'edge_index', 'edge_attr', 'node_list'],
                          output_names=['output'],
                          dynamic_axes={'x': {0: 'node_num'},
                                        'edge_index': {0: 'start_node', 1: 'end_node'},
                                        'edge_attr': {0: 'edge_weight', 1: 'dims'},
                                        'node_list': {0: 'node_num'}})

        torch.save(self.model.state_dict(), model_dir + f"/{self.algo_name}/DIGCN.pt")
        self.logger.info(f"[DiGCN] graph model saved (1/3)")

        self.center = self.center.to('cpu')
        joblib.dump(self.center, model_dir + f"/{self.algo_name}/center.pkl")
        self.logger.info(f"[DiGCN] hypersphere's center vectors saved (2/3)")

        joblib.dump(self.template_miner, model_dir + f"/{self.algo_name}/template_miner.pkl")
        if not os.path.exists(os.path.join(model_dir, f"{constants.MODEL_S_SPARSELOG}")):
            os.makedirs(os.path.join(model_dir, f"{constants.MODEL_S_SPARSELOG}"))
        sparse_miner_path = os.path.join(model_dir, f"{constants.MODEL_S_SPARSELOG}/template_miner.pkl")
        joblib.dump(self.template_miner, sparse_miner_path)
        self.logger.info(f"[DiGCN] template miner saved (3/3)")

    def del_graph_data(self, train=False):
        file_path = os.path.join(self.config['model_dir'], f"{constants.MODEL_S_DIGCN}", "serving")
        if train:
            file_path = os.path.join(self.config['model_dir'], f"{constants.MODEL_S_DIGCN}", "train")
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
            self.logger.info(f"[DiGCN] graph data deleted. path:{file_path}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False