import copy
import time
import random
import pickle
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from algorithms import aimodel
from algorithms.gdn.model import GDN
from algorithms.gdn.data import TimeDataset, data_cleaning, remove_outlier_func
from algorithms.gdn.utils import construct_data, build_loc_net, get_fc_graph_struc
from algorithms.gdn.train import loss_func, model_test, minmax_scal
from algorithms.gdn.config import Conf

from common.aicommon import Query
from common.module_exception import ModuleException


class GraphDetectionNetwork(aimodel.AIModel):

    def __init__(self, id, config, logger):
        ####################
        self.logger = logger
        self.config = copy.deepcopy(config)

        self.algo_name = 'gdn'
        self.model_desc = 'graph based anomaly detection network'
        self.progress_name = 'GDN'

        ####################
        cfg = config['parameter']['train']['gdn']
        self.batch = Conf.batch_size
        self.epoch = Conf.epoch
        self.edge_index_sets = None
        self.node_num = None
        self.slide_win = Conf.window_size
        self.slide_stride = Conf.slide_stride
        self.dim = Conf.dim
        self.topk = Conf.topk
        self.out_layer_inter_dim = Conf.out_layer_inter_dim
        self.out_layer_num = Conf.out_layer_num
        self.val_ratio = Conf.val_ratio
        self.decay = Conf.decay
        self.report = Conf.report
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.window_size = Conf.window_size
        self.confidence_level = Conf.threshold

        ####################
        removal_features = ["socket_count"]  # 훈련시 passive remove feature
        self.train_meta_feature_list = \
            list(filter(lambda x: x not in removal_features, config['parameter']['train']['gdn'].get('features')))

        self.train_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.feature_map = []
        self.save_path = config.get('model_dir')
        self.scaler = None
        self.score_table = None
        self.threshold = None
        self.model = None
        self.best_model = None

        self.logger.info("Success init GDN !!")

    def init_config(self, config: Dict):
        self.config = config

        self.params = config['parameter']['train'][self.algo_name]
        self.pred_feats = self.params['features']
        # self.window_size = self.parmas['window_size']

    def fit(self, df, train_progress=None):
        self.logger.info("====================== Start training ====================== ")

        train_orig = copy.deepcopy(df)

        df = df[self.train_meta_feature_list]  # removal feature

        if len(df.columns) < 2:
            self.logger.error("There is only one feature")
            raise ModuleException('E832')
        if any(columns not in train_orig.columns for columns in self.train_meta_feature_list):
            self.logger.error(f"The feature requested from meta does not exist in the training data.")
            raise ModuleException('E831')

        # df = remove_outlier_func(df)
        # train = data_cleaning(df)
        # self.logger.info("[gdn] Success Data Cleaning ")

        train, scaler = minmax_scal(df)
        self.scaler = scaler
        self.logger.info("[gdn] Success Data Normalizing")

        feature_map = list(train.columns)
        fc_struc = get_fc_graph_struc(feature_map)  # full connect graph
        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=list(feature_map))  # 타겟 노드와 이웃 노드 모두 엣지가 있는 그래프
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map
        self.topk = len(feature_map)
        self.logger.info(f"[gdn] {len(self.feature_map)} features, feature_map : {self.feature_map}")

        train_dataset_indata = construct_data(train, feature_map, labels=0)

        cfg = {
            'slide_win': self.slide_win,
            'slide_stride': self.slide_stride,
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        self.logger.info(f"[gdn] Success preprocessing TimeDataset : {cfg}")

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, self.batch, val_ratio=self.val_ratio)
        self.logger.info(f"[gdn] Success split train, test = ({1-self.val_ratio} / {self.val_ratio})")

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        self.edge_index_sets = edge_index_sets
        self.node_num = len(feature_map)

        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=self.dim,
                         input_dim=self.slide_win,
                         out_layer_num=self.out_layer_num,
                         out_layer_inter_dim=self.out_layer_inter_dim,
                         topk=self.topk
                         ).to(self.device)
        self.logger.info(f"[gdn] Success Build the model [running device: {self.device}]")

        self.logger.info("start model training.")
        _, elapse = self.run(train_progress)
        self.logger.info(f"[gdn] Training ElapsedTime : {elapse} s")

        score_list, result_train_metrics = self._get_thresholds()
        self.score_table = score_list
        self.logger.info(f"[gdn] Success to calculate threshold")

        results = {
            "results": {
                "duration_time": elapse,
                "hyper_params": {
                    "batch_size": self.batch,
                    "epochs": self.epoch,
                    "window_size": self.window_size,
                    "confidence_level": self.confidence_level
                }
            },
            "features": self.feature_map,
            "model_info": {  # 나중에 추가 정보필요하다면 사용
                "mean": 0,
                "std": 0,
                "max": 0,
                "min": 0,
                "threshold": 0
            },
            "train_metrics": result_train_metrics
        }
        self.logger.info("[gdn] ================= Start to save =================")
        self._save()
        self.logger.info("[gdn] Success save model & bundle")
        self.logger.info("[gdn] ================= End training =================")

        return results

    def get_loaders(self, train_dataset, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    def run(self, train_progress=None):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=self.decay)

        train_loss_list = []

        min_loss = 1e+8

        i = 0
        early_stop_win = 5
        device = self.device

        model.train()
        _start = time.time()

        stop_improve_count = 0
        dataloader = self.train_dataloader
        torch.set_num_threads(1)

        for i_epoch in range(self.epoch):
            Query.update_module_status_by_training_id(
                db_conn_str=self.config['db_conn_str'],
                train_history_id=self.config['train_history_id'],
                data=train_progress,
                train_process=self.progress_name,
                progress=int(i_epoch / self.epoch * 100)
            )
            acu_loss = 0
            model.train()

            for x, labels, _, _ in dataloader:
                x = x.float().to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()
                out, _, _ = model(x)
                out = out.float().to(device)
                loss = loss_func(out, labels)

                loss.backward()
                optimizer.step()

                train_loss_list.append(loss.item())
                acu_loss += loss.item()
                i += 1

            if self.val_dataloader is not None:
                val_loss, val_result = model_test(model, self.val_dataloader, self.device)

                # each epoch
                self.logger.info(f'epoch ({i_epoch + 1} / {self.epoch}) '
                                 f'train_Loss:{round((acu_loss / len(dataloader)), 6)}, '
                                 f'valid_Loss:{round(val_loss, 6)}, '
                                 f'ACU_Loss:{round(acu_loss, 4)} ')

                if val_loss < min_loss:
                    self.best_model = model.state_dict()

                    min_loss = val_loss
                    stop_improve_count = 0
                else:
                    stop_improve_count += 1

                if stop_improve_count >= early_stop_win:
                    break

            else:
                if acu_loss < min_loss:
                    self.best_model = model.state_dict()
                    min_loss = acu_loss
        _finished = time.time()
        elapse = round(_finished - _start, 3)

        Query.update_module_status_by_training_id(
            db_conn_str=self.config['db_conn_str'],
            train_history_id=self.config['train_history_id'],
            data=train_progress,
            train_process=self.progress_name,
            progress=100,
            process_start_time=elapse
        )
        return train_loss_list, elapse

    def _get_thresholds(self):
        column_len = len(self.feature_map)
        y_pred = []
        y_real = []

        self.model.load_state_dict(self.best_model)
        best_model = self.model.to(self.device)
        best_model.eval()
        torch.set_num_threads(1)

        for x, y, _, _ in self.val_dataloader:
            x, y = [item.to(self.device).float() for item in [x, y]]

            with torch.no_grad():
                predicted, _, _ = best_model(x)
                predicted = predicted.float().to(self.device)

                y_pred.append(predicted.cpu().detach().numpy())
                y_real.append(y.cpu().detach().numpy())

        temp_y_pred = [[] for _ in range(column_len)]
        temp_y_real = [[] for _ in range(column_len)]

        for i in range(column_len):
            for yy in y_pred:
                temp_y_pred[i].extend(yy[:, i])
            for yy in y_real:
                temp_y_real[i].extend(yy[:, i])

        y_pred = np.array(temp_y_pred)
        y_real = np.array(temp_y_real)
        train_metrics = self._get_mse_train_metrics(y_real, y_pred)
        cal = abs(y_real - y_pred)

        self.logger.info(f"[gdn] get anomaly threshold. shape={cal.shape}")

        score_list = []
        for i in range(column_len):
            loss = np.array(cal[i])
            th_value = np.percentile(loss, self.confidence_level)
            x_data = loss[loss > th_value]

            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(x_data.reshape(-1, 1))
            threshold = (gmm.means_[1][0] + gmm.means_[0][0])/2

            normal_range_loss = np.linspace(0, threshold, 100).reshape(-1, 1)
            anomaly_range_loss = np.linspace(threshold, loss.max(), 100).reshape(-1, 1)

            normal_range_score = np.linspace(100, 50, 100).reshape(-1, 1)
            anomaly_range_score = np.linspace(50, 0, 100).reshape(-1, 1)

            total_range_loss = np.r_[normal_range_loss, anomaly_range_loss]
            total_range_score = np.r_[normal_range_score, anomaly_range_score]

            # score table 생성
            df1 = pd.DataFrame(total_range_loss, columns=["loss"])
            df2 = pd.DataFrame(total_range_score, columns=["score"])
            score_table = pd.concat([df1, df2], axis=1)

            score_list.append(score_table)

        self.logger.info(f"[gdn] End of creating score_table")

        return score_list, train_metrics

    def _get_mse_train_metrics(self, real, pred):
        train_metrics_mse_results = {}
        for idx, key in enumerate(self.feature_map):
            train_metrics_mse_results[key] = {}

            error = real[idx] - pred[idx]
            mse = np.mean(np.square(error))

            train_metrics_mse_results[key]["mse"] = round(float(mse), 3)

        return train_metrics_mse_results

    def _save(self):
        save_dir = self.save_path + "/" + self.algo_name  # ~~~/model_dir/gdn
        self.logger.info(f"[gdn] save path = {save_dir}")
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        model_save_file = save_dir + "/gdn_model.onnx"
        bundle_save_file = save_dir + "/gdn_bundle.pkl"

        self.model.load_state_dict(self.best_model)
        best_model = self.model.to('cpu')
        best_model.eval()
        dummy_data = torch.randn(1, len(self.feature_map), self.window_size).to('cpu')
        torch.onnx.export(
            best_model, dummy_data, model_save_file, input_names=['input'], output_names=['output'], opset_version=16
        )

        bundle = {
            "edge_index_sets": self.edge_index_sets,
            "node_num": self.node_num,
            "dim": self.dim,
            "input_dim": self.slide_win,
            "out_layer_num": self.out_layer_num,
            "out_layer_dim": self.out_layer_inter_dim,
            "topk": self.topk,
            "scaler": self.scaler,
            "score_table": self.score_table
        }
        with open(str(bundle_save_file), "wb") as f:
            pickle.dump(bundle, f)

    def _get_cdf(self, train_anomaly, threshold):
        kde = KernelDensity(kernel='gaussian', bandwidth=.5).fit(train_anomaly.reshape(-1, 1))
        x = np.linspace(0, max(train_anomaly), 1000).reshape(-1, 1)  # 0, 0.01, 0.02, 0.03, ...
        log_pdf = kde.score_samples(x)
        pdf = np.exp(log_pdf)

        # threshold 보다 큰 구간의 분포
        anomaly_pdf = pdf[(x > threshold).reshape(-1)]
        anomaly_cdf = np.cumsum(anomaly_pdf) / np.sum(anomaly_pdf)

        # threshold 보다 작은 구간의 분포
        normal_pdf = pdf[(x <= threshold).reshape(-1)]
        normal_cdf = np.cumsum(normal_pdf) / np.sum(normal_pdf)

        total_cdf = np.r_[normal_cdf, anomaly_cdf]

        df1 = pd.DataFrame(x, columns=["loss"])
        df2 = pd.DataFrame(total_cdf, columns=["cdf"])
        cdf = pd.concat([df1, df2], axis=1)
        loss_threshold = x[len(normal_cdf)]

        return cdf, loss_threshold

