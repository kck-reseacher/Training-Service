import datetime
import os
import pickle

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from copy import deepcopy

from algorithms import aimodel
from common import aicommon
from common import constants as bc
from common.memory_analyzer import MemoryUtil
from common.timelogger import TimeLogger
from common.module_exception import ModuleException


class Config:
    interval_width = 0.9972
    interval_coef = norm.ppf(interval_width / 2 + 0.5)
    window_size = 10
    test_proportion = 30


class DynamicBaseline(aimodel.AIModel):
    def __init__(self, id, config, logger):
        self.mu = MemoryUtil(logger)

        self.model_id = f"dbsln_{id}"
        self.model_desc = bc.MODEL_F_DBSLN

        self.config = config
        self.logger = logger
        self.db_conn_str = config["db_conn_str"]

        # set train hyper parameter
        self.interval_width = Config.interval_width
        self.interval_coef = Config.interval_coef
        self.test_data_proportion = Config.test_proportion

        # set default window_size instead of KeyError
        self.window_size = Config.window_size

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

        # set module parameter
        self.parameter = None
        self.init_param(config)

        # biz day 학습 완료 여부 return 목적
        self.train_business_status = dict()

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
                if biz_train_df.shape[0] == 0:
                    self.logger.info(f"[DBSLN]\tbizidx = {idx} => no data")
                    continue
                biz_train_df.index = pd.to_datetime(biz_train_df["time"], format=bc.INPUT_DATETIME_FORMAT)

                # preprocessing
                features = list(set(self.features) & set(biz_train_df.columns))
                biz_train_df = self.preprocess_train_df(biz_train_df, bizdata=True)
                training_from_date, training_to_date = biz_train_df.index[[0, -1]].strftime(bc.INPUT_DATE_FORMAT)
                self.logger.info(
                    f"[DBSLN]\tbizidx = {idx}, training period : {training_from_date} ~ {training_to_date}")

                # imputation
                self.remove_spike_outliers(biz_train_df, features)
                self.impute_nan_by_mean(biz_train_df, features)
                biz_train_dict[idx] = biz_train_df

            if self.business_list:
                # check trainable bizidx
                bizidx2dates = {
                    idx: np.unique(biz_train_dict[idx]['yymmdd'].values) if idx in biz_train_dict.keys() else [] for idx
                    in biz_dict.keys()}
                df_biz = pd.DataFrame(self.business_list)
                df_biz['trainable_date'] = [[] for _ in range(df_biz.shape[0])]
                df_biz['trainable_date'] = [sorted(
                    set(df_biz.loc[df_biz['index'] == idx, 'date'].squeeze()).intersection(
                        bizidx2dates[idx])) if idx in biz_dict.keys() else [] for idx in df_biz['index']]
                self.total_bizday_status.update(
                    {idx: bool(len(df_biz.loc[df_biz['index'] == idx, 'trainable_date'].item())) for idx in
                     biz_dict.keys()})

                # extend weekday_map
                trainable_bizidx = [idx for idx, is_data_available in self.total_bizday_status.items() if
                                    is_data_available]
                for idx in trainable_bizidx:
                    biz_train_dict[idx].loc[biz_train_dict[idx]['yymmdd'].isin(bizidx2dates[idx]), 'weekday'] = idx
                weekday_map.extend(trainable_bizidx)

                n_trained_days = lambda x: len(df_biz[df_biz['index'] == x]['trainable_date'].squeeze())
                bizday_train_result = [
                    {'index': idx, 'biztype_name': df_biz[df_biz["index"] == idx]["biztype_name"].values[0],
                     'result': bc.BIZDAY_TRAINED if n_trained_days(
                         idx) > 1 else bc.BIZDAY_LESS_THAN_TWO_DAYS if n_trained_days(
                         idx) == 1 else bc.BIZDAY_NOT_IN_DATA} for idx in df_biz['index'].unique()]

        return biz_train_dict, [result['index'] for result in bizday_train_result], bizday_train_result

    def remove_duplicate_bizdays(self, train_df, biz_train_dict):
        if not len(biz_train_dict.keys()):
            return train_df

        total_biz_df = pd.concat([biz_df for biz_df in biz_train_dict.values()])
        return pd.concat([train_df, total_biz_df])

    def impute_nan_by_mean(self, input_df, features=None):
        features = features if features is not None else self.features
        tmp_df = input_df.copy().drop('yymmdd', axis=1).set_index(['weekday', 'minute'])
        mean_df = input_df.groupby(['weekday', 'minute']).mean().interpolate()
        tmp_df.update(mean_df, overwrite=False)
        tmp_df.set_index(input_df.index, inplace=True)
        input_df[features] = tmp_df[features]

    def remove_spike_outliers(self, train_df, features=None, m=30):
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

        weekday_map = deepcopy(bc.TRAINING_WDAY_MAP[self.training_mode])

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
        feats_training_result = self.training_features(total_train_df, weekday_map, trainable_bizidx)
        time_end = time.time()
        self.logger.info(f"[DBSLN] => model training end (elapsed = {time_end - time_start:.3f} sec)")

        self.logger.info("[DBSLN] test data prediction start")
        time_start = time.time()
        _, test_data = train_test_split(train_df, test_size=self.test_data_proportion / 100, shuffle=False)
        dummy_served_bizinfo = pd.DataFrame(columns=["priority", "index"])
        predicted_result = self.predict(list(test_data.index.values), dummy_served_bizinfo, mode='train')
        predicted_result.index = test_data.index

        self.calculate_mean_square_error(test_data, predicted_result, feats_training_result)

        time_end = time.time()
        self.logger.info(f"[DBSLN] => test data prediction end (elapsed = {time_end - time_start:.3f} sec)")

        elapse = np.round(time.time() - elapse, 3)
        self.logger.info(f"[DBSLN] TOTAL ELAPSED = {elapse:.3f}sec")

        report = {"mse": -1, "rmse": -1, "duration_time": elapse, "hyper_params": None}

        results = {
            "from_date": total_train_df.index[0].strftime(bc.INPUT_DATE_FORMAT),
            "to_date": total_train_df.index[-1].strftime(bc.INPUT_DATE_FORMAT),
            "except_failure_date_list": failure_dates, "except_business_list": except_biz_dates,
            "business_list": [] if bizday_train_result is None else bizday_train_result,
            "train_business_status": self.train_business_status,
            "train_mode": self.training_mode, "outlier_mode": self.outlier_mode, "train_metrics": feats_training_result,
            "results": report
        }

        # using in save function
        self.train_result = results

        self.logger.debug(f"[{bc.MODEL_B_DBSLN}] model training - report: {report}")
        self.logger.debug(f"[{bc.MODEL_B_DBSLN}] model training - result: {results}")

        return results, None, 0, None

    # function used in training
    def training_features(self, train_df, wday_map, learning_bizday_index, print_stdout=False):
        n_half_window = int(self.window_size / 2)
        bc.DBSLN_WDAY_MAX = len(wday_map)

        self.logger.info(f"[DBSLN]\ttraining features = {self.features}: training_mode={self.training_mode}")

        '''
        window_data : columns는 지표, index가 7*1440 -> columns는 요일, index가 minute

        '''
        train_start = time.time()
        target_model = {}
        for feature_name in self.features:
            df = train_df.rename(columns={feature_name: "value"})
            df = df[["value", "yymmdd", "weekday", "minute"]]

            if self.outlier_mode:
                result_df = pd.DataFrame()
                for i in np.unique(df["weekday"].values):
                    wday_df = df[df["weekday"] == i].copy()
                    wday_df = wday_df.reset_index(drop=True)
                    dmin_arr = np.unique(wday_df["minute"].values)
                    if (
                            len(np.unique(wday_df["yymmdd"].values)) < 4
                    ):  # 4일 미만의 wday는 window_size를 크게 적용
                        window_size = round(
                            len(dmin_arr) / len(np.unique(wday_df["yymmdd"].values))
                        )
                    else:
                        window_size = bc.OUTLIER_N_WINDOW
                    divided_dmin = np.array_split(
                        dmin_arr, round(len(dmin_arr) / window_size)
                    )
                    outlier_processed_df = self.replace_outlier(wday_df, divided_dmin)
                    result_df = result_df.append(outlier_processed_df, ignore_index=True)
                result_df = result_df.set_index(df.index)
                df = result_df.copy()

            raw_data = []
            for i in range(bc.DBSLN_WDAY_MAX * bc.DBSLN_DMIN_MAX):
                raw_data.append([])

            lcount = 0
            for row_dmin, row_wday, row_value in zip(
                    df["minute"], df["weekday"], df["value"]
            ):
                if row_wday not in learning_bizday_index:
                    wday_index = wday_map[int(row_wday)]
                else:
                    wday_index = wday_map.index(int(row_wday))
                for i in range(self.window_size):
                    dmin_index = (row_dmin + i - n_half_window) % bc.DBSLN_DMIN_MAX

                    raw_index = bc.DBSLN_DMIN_MAX * wday_index + int(dmin_index)
                    raw_data[raw_index].append(row_value)

                if print_stdout and (lcount % 1000 == 0):
                    print(".", end="", flush=True)

                lcount = lcount + 1

            tmp_avg_data = np.full((bc.DBSLN_DMIN_MAX, bc.DBSLN_WDAY_MAX), np.nan)
            tmp_std_data = np.full((bc.DBSLN_DMIN_MAX, bc.DBSLN_WDAY_MAX), np.nan)

            for j in range(bc.DBSLN_WDAY_MAX):
                for i in range(bc.DBSLN_DMIN_MAX):
                    raw_index = bc.DBSLN_DMIN_MAX * j + i
                    if len(raw_data[raw_index]) > 0:
                        window_data = raw_data[raw_index]
                        tmp_avg_data[i][j] = np.mean(window_data)
                        tmp_std_data[i][j] = np.std(window_data)

            tmp_avg = pd.DataFrame(
                tmp_avg_data, index=range(0, bc.DBSLN_DMIN_MAX), columns=wday_map
            )
            tmp_std = pd.DataFrame(
                tmp_std_data, index=range(0, bc.DBSLN_DMIN_MAX), columns=wday_map
            )

            # std_lower_bound = tmp_avg * 0.1  # experiments
            # tmp_std_mod = np.where(std_lower_bound > tmp_std, std_lower_bound, tmp_std)

            tmp_avg_mean = np.repeat(tmp_avg.mean(axis=0).values.reshape(1, -1), bc.DBSLN_DMIN_MAX, axis=0)
            tmp_std_mod_data = np.maximum(tmp_std.values, tmp_avg_mean)
            tmp_std_mod = pd.DataFrame(
                tmp_std_mod_data, index=range(0, bc.DBSLN_DMIN_MAX), columns=wday_map
            )

            tmp_avg = tmp_avg.fillna(method='bfill')
            tmp_avg = tmp_avg.fillna(method='ffill')
            tmp_std_mod = tmp_std_mod.fillna(method='bfill')
            tmp_std_mod = tmp_std_mod.fillna(method='ffill')
            tmp_avg = tmp_avg.dropna(axis=1)
            tmp_std_mod = tmp_std_mod.dropna(axis=1)

            # model = {"df_avg": df_avg, "df_std": df_std_mod}

            target_model[feature_name] = {}
            df_lower, df_upper, df_avg, df_std = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            for wday in np.unique(tmp_avg.columns):
                df_lower.loc[:, f"{wday}"] = tmp_avg[wday].values - self.interval_coef * \
                                             tmp_std_mod[wday].values
                df_upper.loc[:, f"{wday}"] = tmp_avg[wday].values + self.interval_coef * \
                                             tmp_std_mod[wday].values
                df_avg.loc[:, f"{wday}"] = tmp_avg[wday].values
                df_std.loc[:, f"{wday}"] = tmp_std_mod[wday].values
            df_lower[df_lower < 0] = 0
            df_lower = df_lower.round(2)
            df_upper = df_upper.round(2)
            df_avg = df_avg.round(2)
            df_std = df_std.round(2)
            target_model[feature_name]["lower"] = df_lower
            target_model[feature_name]["upper"] = df_upper
            target_model[feature_name]["avg"] = df_avg
            target_model[feature_name]["std"] = df_std

            # combine [lower, upper, avg, std]
            combined_data = []
            for col in target_model[feature_name]["lower"].columns:
                combined_data.append(
                    pd.concat([target_model[feature_name][key][col] for key in target_model[feature_name].keys()],
                              axis=1).values.tolist())

            combined_df = pd.DataFrame(combined_data).T
            combined_df = combined_df.rename(
                columns={idx: value for idx, value in enumerate(target_model[feature_name]["lower"].columns)})
            target_model[feature_name] = combined_df

        """
        common_index, 모든 모델이 공통으로 가지고 있는 index
        WAS1201 tps 모델의 경우 biz index 7번 있지만
        WAS1201 heap_usage 모델의 경우 biz index 7번 없다면(수집 등의 이슈로)
        tps 모델의 7번 index 제거 목적
        """
        common_index = None
        for df_model in target_model.values():
            if common_index is None:
                common_index = set(df_model.columns)
            else:
                common_index = common_index & set(df_model.columns)

        # train meta business day 정보
        request_biz_index_list = []
        for biz_info_list in self.business_list:
            request_biz_index_list.append(biz_info_list["index"])

        # make business_status
        for index in request_biz_index_list:
            self.train_business_status[str(index)] = True if str(index) in common_index else False

        for feature, df_model in target_model.items():
            # common_biz_day 남김
            df_model = df_model[common_index]
            # 0-6 보장, 없을 경우 채워 넣을 series 생성
            frame = None
            index_cnt = 0
            for col in df_model.columns:
                if 0 <= int(col) and int(col) <= 6:  # biz day frame 구성 시 제외
                    index_cnt += 1
                    if frame is None:
                        frame = pd.DataFrame(df_model[col].tolist(), columns=["lower", "upper", "avg", "std"])
                    else:
                        frame += pd.DataFrame(df_model[col].tolist(), columns=["lower", "upper", "avg", "std"])
            # avg
            frame = (frame / index_cnt).round(2)
            avg_series = frame.apply(lambda row: row.tolist(), axis=1)
            # 0-6 채우기
            for i in range(7):
                if str(i) in df_model.columns:
                    pass
                else:  # 해당 컬럼 없는 경우
                    df_model.loc[:, str(i)] = avg_series

            target_model[feature] = df_model

        self.training_model = target_model
        train_end = time.time()

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

    # X: list of dates(%Y-%m-%d %H:%M:%S)
    def predict(self, datetime_list, served_bizinfo, fail_type=None, mode='serving'):
        try:
            with TimeLogger("[predict] dbsln prediction takes ", self.logger):
                wday_map = self.wday_map if self.wday_map is not None else bc.TRAINING_WDAY_MAP[self.training_mode]

                result = pd.DataFrame(index=pd.to_datetime(datetime_list, format=bc.INPUT_DATE_FORMAT))
                result["minute"] = result.index.map(lambda x: x.hour * 60 + x.minute)
                result["weekday"] = result.index.map(lambda x: wday_map[x.weekday()])
                self.change_wday_if_bizday_serving(result, served_bizinfo)
                result.set_index(['weekday', 'minute'], inplace=True)

                cols_by_untrained_feats = [col for col, cnt in result.isnull().sum(axis=0).items() if
                                           cnt == result.shape[0]]
                result.drop(cols_by_untrained_feats, axis=1, inplace=True)

                result.set_index(pd.to_datetime(datetime_list, format=bc.INPUT_DATE_FORMAT), inplace=True)
                result.index.set_names('time', inplace=True)
                # self.logger.debug(list(result.columns))
                result[result < 0.0] = 0.0
        except Exception as e:
            ex_type, ex_value, ex_traceback = sys.exc_info()
            error_type, error_value = ex_type.__name__, ex_value.args[0]
            self.logger.exception(e)
            if error_type == "KeyError":
                if error_value == -1:
                    self.logger.warning(
                        "Check data volume in training directory because this error might be caused by lack of data")
                    raise ModuleException("E910")
                else:
                    raise ModuleException("E910")
            else:
                raise ModuleException("E910")

        return result

    def change_wday_if_bizday_serving(self, result, served_bizinfo):
        served_bizinfo["status"] = None
        if len(served_bizinfo) != 0:
            served_bizinfo["status"] = [bizidx in self.wday_map for bizidx in served_bizinfo['index']]
            # if bizidx in serving data is in trained_bizidx_list => change wday to wday_bizidx
            if served_bizinfo[served_bizinfo['status']].shape[0] > 0:
                wday_today = result["weekday"].values[0]
                wday_bizidx = served_bizinfo[served_bizinfo['status']].sort_values('priority')['index'].iloc[0].astype(
                    int)
                result.loc[result['weekday'] == wday_today, 'weekday'] = wday_bizidx

    def save(self, model_dir):
        serving_dir = Path(model_dir) / bc.MODEL_S_DBSLN

        try:
            if not os.path.exists(serving_dir):
                os.makedirs(serving_dir)

            self.logger.info(f"[DBSLN] start saving model to {serving_dir}")

            for file in os.listdir(serving_dir):
                if os.path.isfile(serving_dir):
                    os.remove(file)

            file_path_model = os.path.join(serving_dir, f"{self.model_id}.pkl")
            self.logger.info(f"[DBSLN]\ttraining_mode = {self.training_mode}")
            self.logger.info(f"[DBSLN]\ttrained_bizday_status = {self.total_bizday_status}")
            # self.logger.info(f"[DBSLN]\tmodel = (avg_model.shape = {self.training_model['df_avg'].shape}, std_model.shape = {self.training_model['df_std'].shape})")

            with open(file_path_model, "wb") as f:
                pickle.dump(self.training_model, f)
            self.logger.info(f"[DBSLN] saving model finished")

        except Exception as e:
            self.logger.exception(f"[DBSLN] model saving failed : {e}")
            return False

        return True