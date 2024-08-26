from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset

import common.constants as bc
import pandas as pd
import numpy as np
import time
import math

class TimeSeriesClustering(object):
    def __init__(self, config, logger):
        self.clustering_score = 0.6
        self.cluster_percent = 10
        self.logger = logger
        self.config = config

    def clustering(self, df, date_list, multivariate = False):
        """
        @param df: 지표별 데이터 index : 시간별 index, columns: date
        @return: data : 정제된 데이터, total_exclude_date : 제거된 날짜
        """
        exclude_date = list()

        if multivariate:
            ts_df = to_time_series_dataset(df)
        else:
            ts_df = to_time_series_dataset(df)

        ts_df = TimeSeriesScalerMinMax().fit_transform(ts_df)

        clustering_results = list()
        scores = list()

        for i in range(2, math.ceil(self.training_date/10) + 2):
            euclidean_cluster_labels = TimeSeriesKMeans(n_clusters=i, metric='euclidean', max_iter=10, n_jobs=-1, random_state=0).fit_predict(ts_df)
            if len(set(euclidean_cluster_labels)) == 1:
                clustering_results.append(euclidean_cluster_labels)
                scores.append(0)
                continue
            score = silhouette_score(ts_df, euclidean_cluster_labels, metric="euclidean")
            clustering_results.append(euclidean_cluster_labels)
            scores.append(score)
        self.logger.info(scores)

        if len(set(scores)) == 1:
            return exclude_date

        score = max(scores)
        euclidean_cluster_labels = clustering_results[scores.index(score)]

        score = silhouette_score(ts_df, euclidean_cluster_labels, metric="euclidean")
        self.logger.info(f"clustering score {score}")

        cluster_count = {}
        for i in euclidean_cluster_labels:
            try:
                cluster_count[i] += 1
            except:
                cluster_count[i] = 1

        self.logger.info(f"cluster : {len(set(cluster_count))}")

        if score < self.clustering_score:
            return exclude_date

        for k, v in cluster_count.items():  # 또는 클러스터링 개수만큼 for loop
            for j in range(len(euclidean_cluster_labels)):
                if euclidean_cluster_labels[j] == k and score >= self.clustering_score and (v / len(euclidean_cluster_labels)) * 100 <= self.cluster_percent / 2.2:
                    exclude_date.append(date_list[j])

        return exclude_date

    def training(self, data: pd.DataFrame, algorithm: str, multivariate = False):
        """
        @param data: anls_inst에서 concat 한 뒤에 사용 될 dataframe in list
        @return: 클러스터링 후 학습에 불필요 한 날짜가 제거된 데이터

        Parameters
        ----------
        algorithm
        """
        self.logger.info(f"start clustering")
        self.logger.info(f"Enable multivariate clustering : {multivariate}")

        features = self.config['parameter']['train'][algorithm]['features']
        self.logger.info(features)
        self.logger.info(len(features))

        start = time.time()
        data.reset_index(inplace=True, drop=True)
        data.reset_index(inplace=True)

        data_copy = data.copy()
        data_copy['yyyy-mm-dd'] = [dt[:10] for dt in data_copy['time']]
        data_copy['time'] = pd.to_datetime(data_copy['time'].values, format=bc.INPUT_DATETIME_FORMAT)
        data_copy['dmin'] = data_copy['time'].map(lambda x: int(x.hour * 60 + x.minute))

        self.training_date = len(set(data_copy['yyyy-mm-dd']))

        if self.training_date <= 5:
            self.logger.info(f"clustering canceled because data is less than 6days ")
            return data, []

        total_exclude_date = list()
        date_list = data_copy['yyyy-mm-dd'].unique()

        if not multivariate:

            for feature in features:
                self.logger.info(f"start {feature}")
                df = data_copy.pivot_table(index='dmin', columns='yyyy-mm-dd', values=feature)
                df.interpolate(limit_area="inside", inplace=True)
                df.interpolate(limit_direction="both", inplace=True)

                exclude_date = self.clustering(df.values.T, date_list)

                total_exclude_date.extend(exclude_date)

                self.logger.info(f"Done : {feature}")
            total_exclude_date = list(set(total_exclude_date))
        else:
            self.logger.info(f"start multivariate clustering")

            data_temp_list = list()

            for feature in features:
                data_pivot = data_copy.pivot(index='dmin', columns='yyyy-mm-dd', values=feature)
                data_pivot.interpolate(limit_area="inside", inplace=True)
                data_pivot.interpolate(limit_direction="both", inplace=True)
                data_temp_list.append(data_pivot.values)

            tensor_data = np.array(data_temp_list).transpose(2, 1, 0)

            total_exclude_date = list(self.clustering(tensor_data, date_list, multivariate))

        self.logger.info(f"total_exclude_date -> {total_exclude_date}, length:{len(total_exclude_date)}")

        for date in total_exclude_date:
            data = data.drop(data[data['time'].str.contains(date) == True]['index'])

        self.logger.info(f"elapse_time: {time.time() - start}")

        return data.drop(columns=['index']), total_exclude_date