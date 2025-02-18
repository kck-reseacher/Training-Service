import datetime
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from algorithms import aimodel
from common import aicommon
from common import constants as bc
from common.timelogger import TimeLogger
from common.module_exception import ModuleException


class DynamicBaseline(aimodel.AIModel):
    # def __new__(cls, *args, **kwargs):  # class에 속성을 binding 하는 단순한 singleton 패턴
    #     if not hasattr(cls, "_instance"):
    #         cls._instance = super().__new__(cls)  # binding _instance
    #     return cls._instance
    def __init__(self, id, config, logger):

        self.model_id = f"dbsln_{id}"
        self.model_desc = bc.MODEL_F_DBSLN

        self.config = config
        self.logger = logger

        # set default interval_width instead of KeyError
        self.interval_width = self.config.get("interval_width", 0.9972)
        self.interval_coef = norm.ppf(self.interval_width / 2 + 0.5)

        # set default window_size instead of KeyError
        self.window_size = self.config.get("window_size", bc.DBSLN_N_WINDOW)

        # set replace outliers
        self.outlier_mode = self.config.get("outlier_mode", False)

        # set a business calendar paramater for training
        self.business_list = self.config.get("business_list", [])

        # set BusinessDate for except training
        self.except_business_list = self.config.get("except_business_list", [])

        # except failure date
        self.except_failure_date_list = self.config.get("except_failure_date_list", [])

        # define a variable to set training mode
        self.training_mode = bc.TRAINING_MODE_INIT
        self.wday_map = None

        # 학습 모델 및 학습 결과
        self.training_model = {}
        self.total_bizday_status = {}
        self.train_result = None

        # 테스트 데이터 비율
        self.test_data_proportion = self.config["parameter"]["data_set"]["test"] if 'data_set' in self.config["parameter"].keys() else None

        # set module parameter
        self.parameter = None
        self.init_param(config)

        # cls._init = True

    def init_config(self, config, logger=None):
        self.config = config

        if logger is not None:
            self.logger = logger

        # set default interval_width instead of KeyError
        self.interval_width = self.config.get("interval_width", 0.9972)
        self.interval_coef = norm.ppf(self.interval_width / 2 + 0.5)

        # set default window_size instead of KeyError
        self.window_size = self.config.get("window_size", bc.DBSLN_N_WINDOW)

        # set replace outliers
        self.outlier_mode = self.config.get("outlier_mode", False)

        # set a business calendar paramater for training
        self.business_list = self.config.get("business_list", [])

        # set BusinessDate for except training
        self.except_business_list = self.config.get("except_business_list", [])

        # except failure date
        self.except_failure_date_list = self.config.get("except_failure_date_list", [])

        # set module parameter
        self.init_param(config)

    def init_param(self, config):
        # set module parameter
        self.parameter = config.get("parameter", None)
        self.features = self.parameter["train"]["dbsln"]["features"]
        self.logger.info("Success dynamic_baseline init_param !!")


    def outlier_iqr(self, data):
        q1, q3 = np.nanpercentile(data, [25, 75])  # q1, q3 :  각각 25번째, 75번째 백분위수 지정
        iqr = q3 - q1
        lower_bound = q1 - (iqr * bc.SIGMA)  # q1 - (iqr x 2.2) 를 lower_bound로 설정
        upper_bound = q3 + (iqr * bc.SIGMA)  # q3 + (iqr x 2.2) 를 upper_bound로 설정
        return np.where(
            (data > upper_bound) | (data < lower_bound)
        )  # Outlier가 있는 index를 return

    def replace_outlier(self, wday_df, divided_dmin):
        wday_df = wday_df.sort_values(by="minute")
        wday_df = wday_df.reset_index(drop=True)

        # make a wday_df vetorization
        wday_data = []
        for i in range(len(divided_dmin)):
            from_index = wday_df[wday_df["minute"] == divided_dmin[i][0]].index[0]
            to_index = wday_df[wday_df["minute"] == divided_dmin[i][-1]].index[-1] + 1
            wday_data.append(wday_df.loc[from_index:to_index]["value"].values)

        # detect and replace outliers
        for i in wday_data[0: len(wday_data)]:
            outlier = self.outlier_iqr(i)[0]
            mask_arr = np.ma.array(i, mask=False)
            mask_arr.mask[outlier] = True
            for out in outlier:
                i[out] = mask_arr.mean()

        return wday_df

    def preprocess_train_df(self, train_df, bizdata=False):
        train_df = train_df[~train_df.index.duplicated()]
        time_diff = train_df.index[-1] - train_df.index[-2]
        if not bizdata and time_diff != datetime.timedelta(minutes=5):
            new_idx = pd.date_range(train_df.index[0], train_df.index[-1], freq="1min")
            train_df = train_df.reindex(new_idx)
        train_df["weekday"] = train_df.index.map(lambda x: x.weekday())
        train_df["default_weekday"] = train_df.index.map(lambda x: x.weekday())
        train_df["minute"] = train_df.index.map(lambda x: x.hour * 60 + x.minute)
        train_df["yymmdd"] = train_df.index.map(lambda x: x.strftime(bc.INPUT_DATE_YMD))

        return train_df

    def train_data_drop_failure_and_business(self, train_df):
        failure_dates = []
        if len(self.except_failure_date_list) != 0:
            train_df, failure_dates = aicommon.Utils.drop_failure_date(self.except_failure_date_list, train_df)

        except_biz_dates = []
        if len(self.except_business_list) != 0:
            train_df, except_biz_dates = aicommon.Utils.drop_except_business_list(self.except_business_list, train_df)

        return train_df, failure_dates, except_biz_dates

    def preprocessing_for_business_train_data(self, biz_dict, weekday_map):
        biz_train_dict = {}
        bizday_train_result = []
        if biz_dict is not None:
            for idx in list(biz_dict.keys()):
                self.total_bizday_status[idx] = False
                biz_train_df = biz_dict[idx]
                biz_train_df.index = pd.to_datetime(biz_train_df.index, format=bc.INPUT_DATETIME_FORMAT)
                if biz_train_df.shape[0] == 0:
                    self.logger.info(f"[DBSLN]\tbizidx = {idx} => no data")
                    continue

                # preprocessing
                features = list(set(self.features) & set(biz_train_df.columns))
                biz_train_df = self.preprocess_train_df(biz_train_df, bizdata=True)
                training_from_date, training_to_date = biz_train_df.index[[0, -1]].strftime(bc.INPUT_DATE_FORMAT)
                self.logger.info(f"[DBSLN]\tbizidx = {idx}, training period : {training_from_date} ~ {training_to_date}")

                # imputation
                self.remove_spike_outliers(biz_train_df, features)
                self.impute_nan_by_mean(biz_train_df, features)
                biz_train_dict[idx] = biz_train_df

            if self.business_list:
                # check trainable bizidx
                bizidx2dates = {idx: np.unique(biz_train_dict[idx]['yymmdd'].values) if idx in biz_train_dict.keys() else [] for idx in biz_dict.keys()}
                df_biz = pd.DataFrame(self.business_list)
                df_biz['trainable_date'] = [[] for _ in range(df_biz.shape[0])]
                df_biz['trainable_date'] = [sorted(set(df_biz.loc[df_biz['index'] == idx, 'date'].squeeze()).intersection(bizidx2dates[idx])) if idx in biz_dict.keys() else [] for idx in df_biz['index']]
                self.total_bizday_status.update({idx: bool(len(df_biz.loc[df_biz['index'] == idx, 'trainable_date'].item())) for idx in biz_dict.keys()})

                # extend weekday_map
                trainable_bizidx = [idx for idx, is_data_available in self.total_bizday_status.items() if is_data_available]
                for idx in trainable_bizidx:
                    biz_train_dict[idx].loc[biz_train_dict[idx]['yymmdd'].isin(bizidx2dates[idx]), 'weekday'] = idx
                weekday_map.extend(trainable_bizidx)

                n_trained_days = lambda x: len(df_biz[df_biz['index'] == x]['trainable_date'].squeeze())
                bizday_train_result = [{'index': idx, 'biztype_name': df_biz[df_biz["index"] == idx]["biztype_name"].values[0],
                                        'result': bc.BIZDAY_TRAINED if n_trained_days(idx) > 1 else bc.BIZDAY_LESS_THAN_TWO_DAYS if n_trained_days(idx) == 1 else bc.BIZDAY_NOT_IN_DATA} for idx in df_biz['index'].unique()]

        return biz_train_dict, [result['index'] for result in bizday_train_result], bizday_train_result

    def remove_duplicate_bizdays(self, train_df, biz_train_dict):
        if not len(biz_train_dict.keys()):
            return train_df

        total_biz_df = pd.concat([biz_df for biz_df in biz_train_dict.values()])
        total_train_df = train_df[~train_df['yymmdd'].isin(total_biz_df['yymmdd'].unique())]
        return pd.concat([total_train_df if total_train_df.shape[0] > 0 else train_df, total_biz_df])

    def impute_nan_by_mean(self, input_df, features = None):
        features = features if features is not None else self.features
        tmp_df = input_df.copy().drop('yymmdd', axis=1).set_index(['weekday', 'minute'])
        mean_df = input_df.groupby(['weekday', 'minute']).mean().interpolate()
        tmp_df.update(mean_df, overwrite=False)
        tmp_df.set_index(input_df.index, inplace=True)
        input_df[features] = tmp_df[features]

    def remove_spike_outliers(self, train_df, features = None, m=30):
        features = features if features is not None else self.features
        self.logger.debug(f"[DBSLN]\t\t training feature({len(features)}) : {features}")
        feats_data = train_df[features].values

        trend = np.zeros_like(feats_data)
        for i in range(trend.shape[0]):
            trend[i] = np.mean(feats_data[np.max([0, i - m]):i + 1], axis=0)

        noise = np.maximum(0, feats_data - trend)

        for i, feat in enumerate(features):
            feat_data = feats_data[:, i].astype(float)
            noise_feat = noise[:, i]
            upper_bound = np.nanmean(noise_feat) + 3 * np.nanstd(noise_feat)

            feat_data[np.argwhere(noise_feat > upper_bound)] = np.nan
            train_df[feat] = feat_data

    def fit(self, df, biz_dict, train_progress=None):
        self.logger.info(f"model {self.model_id} start training")

        elapse = time.time()
        train_df = df.copy()
        train_df.index = pd.to_datetime(train_df.index, format=bc.INPUT_DATETIME_FORMAT)

        train_df, failure_dates, except_biz_dates = self.train_data_drop_failure_and_business(train_df)
        if len(train_df) == 0:
            raise ModuleException("E800")

        # preprocessing - train_df
        self.logger.info(f"[DBSLN] data preprocessing start")
        time_start = time.time()
        train_df = self.preprocess_train_df(train_df)

        training_from_date, training_to_date = train_df.iloc[[0, -1]].index.strftime(bc.INPUT_DATE_FORMAT)
        self.logger.info(f"[DBSLN]\ttraining period : {training_from_date} ~ {training_to_date}")

        # remove outliers
        self.remove_spike_outliers(train_df)

        # imputation
        self.impute_nan_by_mean(train_df)
        time_end = time.time()
        self.logger.info(f"[DBSLN] => data preprocessing end (elapsed = {time_end - time_start:.3f} sec)")

        # training_mode 설정
        self.training_mode = aicommon.Utils.calc_training_mode(len(np.unique(train_df["yymmdd"].values)))
        if self.training_mode < 0:
            raise ModuleException("E801")

        weekday_map = bc.TRAINING_WDAY_MAP[self.training_mode]

        # preprocessing - biz_df
        self.logger.info(f"[DBSLN] business day preprocessing start")
        time_start = time.time()
        biz_train_dict, trainable_bizidx, bizday_train_result = self.preprocessing_for_business_train_data(biz_dict,
                                                                                                           weekday_map)

        # concat - train_df + biz_df
        total_train_df = self.remove_duplicate_bizdays(train_df, biz_train_dict)
        time_end = time.time()
        self.logger.info(f"[DBSLN] => business day preprocessing end (elapsed = {time_end - time_start:.3f} sec)")

        self.logger.info(f"[DBSLN] model training start")
        time_start = time.time()
        feats_training_result = self.training_features(total_train_df, weekday_map)
        time_end = time.time()
        self.logger.info(f"[DBSLN] => model training end (elapsed = {time_end - time_start:.3f} sec)")

        self.logger.info("[DBSLN] test data prediction start")
        time_start = time.time()
        _, test_data = train_test_split(train_df, test_size=self.test_data_proportion / 100, shuffle=False)
        dummy_served_bizinfo = pd.DataFrame(columns=["priority", "index"])

        try:
            with TimeLogger("[predict] dbsln prediction takes ", self.logger):
                wday_map = self.wday_map if self.wday_map is not None else bc.TRAINING_WDAY_MAP[self.training_mode]

                predicted_result = pd.DataFrame(index=pd.to_datetime(list(test_data.index.values), format=bc.INPUT_DATE_FORMAT))
                predicted_result["minute"] = predicted_result.index.map(lambda x: x.hour * 60 + x.minute)
                predicted_result["weekday"] = predicted_result.index.map(lambda x: wday_map[x.weekday()])
                self.change_wday_if_bizday_serving(predicted_result, dummy_served_bizinfo)
                predicted_result.set_index(['weekday', 'minute'], inplace=True)
                self.forecast_features(predicted_result, None)

                cols_by_untrained_feats = [col for col, cnt in predicted_result.isnull().sum(axis=0).items() if cnt == predicted_result.shape[0]]
                predicted_result.drop(cols_by_untrained_feats, axis=1, inplace=True)

                predicted_result.set_index(pd.to_datetime(list(test_data.index.values), format=bc.INPUT_DATE_FORMAT), inplace=True)
                predicted_result.index.set_names('time', inplace=True)
                predicted_result[predicted_result < 0.0] = 0.0
        except Exception as e:
            ex_type, ex_value, ex_traceback = sys.exc_info()
            error_type, error_value = ex_type.__name__, ex_value.args[0]
            self.logger.exception(e)
            if error_type == "KeyError":
                if error_value == -1:
                    self.logger.warning("Check data volume in training directory because this error might be caused by lack of data")
                    raise ModuleException("E910")
                else:
                    raise ModuleException("E910")
            else:
                raise ModuleException("E910")

        predicted_result.index = test_data.index

        self.calculate_mean_square_error(test_data, predicted_result, feats_training_result)
        time_end = time.time()
        self.logger.info(f"[DBSLN] => test data prediction end (elapsed = {time_end - time_start:.3f} sec)")

        elapse = np.round(time.time() - elapse, 3)
        self.logger.info(f"[DBSLN] TOTAL ELAPSED = {elapse:.3f}sec")

        aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'],
                                                           train_progress, self.model_desc, 100, elapse)

        report = {"mse": -1, "rmse": -1, "duration_time": elapse, "hyper_params": None}

        results = {
            "from_date": total_train_df.index[0].strftime(bc.INPUT_DATE_FORMAT),
            "to_date": total_train_df.index[-1].strftime(bc.INPUT_DATE_FORMAT),
            "except_failure_date_list": failure_dates, "except_business_list": except_biz_dates,
            "business_list": [] if bizday_train_result is None else bizday_train_result,
            "train_business_status": {str(key): bool(value) for key, value in
                                      self.total_bizday_status.items()} if not len(
                self.total_bizday_status) == 0 else None,
            "train_mode": self.training_mode, "outlier_mode": self.outlier_mode, "train_metrics": feats_training_result,
            "results": report
        }

        # using in save function
        self.train_result = results

        self.logger.debug(f"[{bc.MODEL_B_DBSLN}] model training - report: {report}")
        self.logger.debug(f"[{bc.MODEL_B_DBSLN}] model training - result: {results}")

        return results, None, 0, None

    # funstion used in training
    def training_features(self, train_df, wday_map):
        self.logger.info(f"[DBSLN]\ttraining features = {self.features}: training_mode={self.training_mode}")
        df = train_df.copy()

        train_start = time.time()
        if self.outlier_mode:
            result_df = pd.DataFrame()
            result_df[['yymmdd', 'weekday', 'minute', 'default_weekday']] = df[
                ['yymmdd', 'weekday', 'minute', 'default_weekday']]
            for feat in [feat for feat in self.features if feat in df.columns]:
                feat_df = df[[feat, 'yymmdd', 'weekday', 'minute']].rename(columns={feat: 'value'})
                feat_result = []
                for i in np.unique(df["weekday"].values):
                    wday_df = feat_df[feat_df["weekday"] == i].copy()
                    wday_df = wday_df.reset_index(drop=True)
                    dmin_arr = np.unique(wday_df["minute"].values)
                    if len(np.unique(wday_df["yymmdd"].values)) < 4:  # 4일 미만의 wday는 window_size를 크게 적용
                        window_size = round(len(dmin_arr) / len(np.unique(wday_df["yymmdd"].values)))
                    else:
                        window_size = bc.OUTLIER_N_WINDOW
                    divided_dmin = np.array_split(dmin_arr, round(len(dmin_arr) / window_size))
                    outlier_processed_feat_df = self.replace_outlier(wday_df, divided_dmin)
                    feat_result.extend(outlier_processed_feat_df['value'].values.reshape(-1))
                result_df[feat] = feat_result
            result_df = result_df.set_index(df.index)
            df = result_df.copy()

        # default - weekly training mode
        window_data = [[] * len(self.features) for _ in range(len(wday_map) * bc.DBSLN_DMIN_MAX)]
        dt_cnt = np.zeros((len(wday_map) * bc.DBSLN_DMIN_MAX,), dtype=int)

        for dmin, wday, d_wday, *feats_data in df[['minute', 'weekday', 'default_weekday'] + self.features].values:
            if wday > 6:
                get_widx = lambda m, w, d_w: int(((d_w - 1) % 7 if m < 0 else (
                                                                                          d_w + 1) % 7 if m > 1439 else wday_map.index(
                    w)) * bc.DBSLN_DMIN_MAX + m) % len(window_data)
                window_idxs = [get_widx(midx, wday, d_wday) for midx in
                               np.arange(dmin - self.window_size // 2, dmin + self.window_size // 2, dtype=int)]
            else:
                get_widx = lambda m, w: wday_map[(w - 1) % 7 if m < 0 else (
                                                                                       w + 1) % 7 if m > 1439 else w] * bc.DBSLN_DMIN_MAX + m % 1440
                window_idxs = [get_widx(midx, int(wday)) for midx in
                               np.arange(dmin - self.window_size // 2, dmin + self.window_size // 2, dtype=int)]

            _ = [window_data[w_idx].append(feats_data) for w_idx in window_idxs]
            dt_cnt[window_idxs] += 1

        dt_avg = np.array([np.mean(window_data[i], axis=0) if dt_cnt[i] > 0 else np.zeros(len(self.features)) for i in
                           range(len(window_data))])
        dt_std = np.array([np.std(window_data[i], axis=0) if dt_cnt[i] > 0 else np.zeros(len(self.features)) for i in
                           range(len(window_data))])

        std_lower_bound = dt_avg * 0.1  # experiments
        dt_std = np.where(std_lower_bound > dt_std, std_lower_bound, dt_std)

        train_end = time.time()
        # index : [(wday_0, dmin_0), (wday_0, dmin_1), ..., (wday_0, dmin_1439), (wday_1, dmin_0), ..., (wday_n, dmin_0), ..., (wday_n, dmin_1438), (wday_n, dmin_1439)]
        df_wday_map = wday_map.copy()
        df_wday_map[:7] = np.arange(7)
        multi_index = pd.MultiIndex.from_arrays([np.repeat(df_wday_map, bc.DBSLN_DMIN_MAX), np.ravel(
            [list(range(bc.DBSLN_DMIN_MAX)) for _ in range(len(df_wday_map))])], names=['weekday', 'minute'])
        self.training_model = {'df_avg': pd.DataFrame(dt_avg, index=multi_index,
                                                      columns=[f"{feat}_avg" for feat in self.features]).interpolate(
            limit_direction='both').round(2),
                               'df_std': pd.DataFrame(dt_std, index=multi_index,
                                                      columns=[f"{feat}_std" for feat in self.features]).interpolate(
                                   limit_direction='both').round(2),
                               'dt_cnt': pd.DataFrame(dt_cnt, index=multi_index, columns=['cnt'])}

        return {
            feat: {"duration_time": np.round((train_end - train_start) / len(self.features), 3), "hyper_params": None}
            for feat in self.features}

    def calculate_mean_square_error(self, test_data, predicted_result, feats_training_result):
        for feature_name in self.features:
            if feature_name in predicted_result.columns:
                real_value = test_data[feature_name]
                prediction = predicted_result[[f"{feature_name}_lower", f"{feature_name}_upper"]]
                df = pd.concat([real_value, prediction], axis=1)
                df["upper_deviation"] = df[feature_name] - df[f"{feature_name}_upper"]
                df["lower_deviation"] = df[feature_name] - df[f"{feature_name}_lower"]
                df["deviation"] = np.where(df["upper_deviation"] > 0, df["upper_deviation"],
                                           np.where(df["lower_deviation"] < 0, df["lower_deviation"], 0))

                mse = np.mean(np.square(df["deviation"]))
                feats_training_result[feature_name]["mse"] = mse.round(3)
                feats_training_result[feature_name]["rmse"] = np.sqrt(mse).round(3)

    def change_wday_if_bizday_serving(self, result, served_bizinfo):
        served_bizinfo["status"] = None
        if len(served_bizinfo) != 0:
            served_bizinfo["status"] = [bizidx in self.wday_map for bizidx in served_bizinfo['index']]
            # if bizidx in serving data is in trained_bizidx_list => change wday to wday_bizidx
            if served_bizinfo[served_bizinfo['status']].shape[0] > 0:
                wday_today = result["weekday"].values[0]
                wday_bizidx = served_bizinfo[served_bizinfo['status']].sort_values('priority')['index'].iloc[0].astype(int)
                result.loc[result['weekday'] == wday_today, 'weekday'] = wday_bizidx

    def forecast_features(self, forecast_df, fail_type):
        feats_ = lambda postfix: [f"{feat}_{postfix}" if len(postfix) > 0 else feat for feat in self.features]

        forecast_df[np.ravel([feats_('avg'), feats_('std')])] = np.nan
        forecast_df.update(self.training_model['df_avg'])
        forecast_df.update(self.training_model['df_std'])

        scale_config = {feat: {'percent': 0, 'scale_out': True} for feat in self.features}
        if self.parameter['service'] is not None:
            scale_config.update(self.parameter["service"][bc.MODEL_S_DBSLN]["range"])
            if fail_type is not None:
                self.logger.info(f'fail_type data = {self.parameter["service"]["fail_condition"]}')
                fail_condition_dict = pd.DataFrame(self.parameter["service"]["fail_condition"][fail_type], columns=['stat', 'range_percent', 'scale_out']).set_index('stat').T.to_dict()
                scale_config.update(fail_condition_dict)
        scale_coef = np.array([1 + ((-1 + (2 * scale_config[feat]['scale_out'])) * (scale_config[feat]['percent'] / 100)) for feat in scale_config.keys()])
        # => (-1 + (2*scale_out)) : scale_out == True => 1, False => -1

        forecast_df[feats_('')] = forecast_df[feats_('avg')]
        forecast_df[feats_('lower')] = forecast_df[feats_('')].values - self.interval_coef * scale_coef * forecast_df[feats_('std')].values
        forecast_df[feats_('upper')] = forecast_df[feats_('')].values + self.interval_coef * scale_coef * forecast_df[feats_('std')].values

    def check(self, model_value, metric_name, metric_value, fail_type=None):
        threshold = bc.DBSLN_CHECK_MIN  # default 값은 DBSLN_CHECK_MIN
        upper = model_value[f"{metric_name}_upper"]
        lower = model_value[f"{metric_name}_lower"]

        # 장애 탐지 threshold 적용
        fail_stats = []
        if self.parameter["service"] is not None and fail_type is not None:
            fail_condition = pd.DataFrame(
                self.parameter["service"]["fail_condition"][fail_type]
            )
            if len(fail_condition) > 0:
                fail_stats = list(fail_condition["stat"])

            if len(fail_stats) > 0 and metric_name in fail_stats:
                threshold = int(
                    fail_condition[fail_condition["stat"] == metric_name]["threshold"]
                )

        if type(metric_value) == str:
            metric_value = float(metric_value)

        if metric_value >= threshold and (metric_value > upper or metric_value < lower):
            avg = model_value[metric_name]
            std = model_value[f"{metric_name}_std"]
            std = np.max((std, 0.0000001))
            dev = metric_value - avg
            zscore = (metric_value - avg) / std
            anomaly = {
                "name": metric_name,
                "value": metric_value,
                "avg": avg,
                "std": std,
                "lower": lower,
                "upper": upper,
                "deviation": dev,
                "zscore": zscore,
            }
            return anomaly
        else:
            return None

    def save(self, model_dir):
        serving_dir = Path(model_dir) / bc.MODEL_S_DBSLN

        try:
            if not os.path.exists(serving_dir):
                os.makedirs(serving_dir)

            self.logger.info(f"[DBSLN] start saving model to {serving_dir}")

            for file in os.listdir(serving_dir):
                if os.path.isfile(serving_dir):
                    os.remove(file)

            file_path = os.path.join(serving_dir, f"{self.model_id}.pkl")
            self.logger.info(f"[DBSLN]\ttraining_mode = {self.training_mode}")
            self.logger.info(f"[DBSLN]\ttrained_bizday_status = {self.total_bizday_status}")
            self.logger.info(f"[DBSLN]\tmodel = (avg_model.shape = {self.training_model['df_avg'].shape}, std_model.shape = {self.training_model['df_std'].shape})")

            with open(file_path, "wb") as f:
                pickle.dump(self.training_mode, f)
                pickle.dump(self.total_bizday_status, f)
                pickle.dump(self.training_model, f)

            self.logger.info(f"[DBSLN] saving model finished")

        except Exception as e:
            self.logger.exception(f"[DBSLN] model saving failed : {e}")
            return False

        return True
