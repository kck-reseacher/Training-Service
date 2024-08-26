import datetime
import logging
import os
import pickle
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pathos
from scipy.stats import norm

from algorithms import aimodel
from common import aicommon
from common import constants as bc
from common.error_code import Errors
from common.module_exception import ModuleException


class Config:
    interval_width = 0.9972
    interval_coef = norm.ppf(interval_width / 2 + 0.5)
    window_size = 10
    test_proportion = 30


class DynamicBaseline(aimodel.AIModel):
    def __init__(self, id, config, logger):
        self.model_id = f"dbsln_{id}"
        self.model_desc = "Dynamic baseline"

        self.config = config
        self.logger = logger

        # set default interval_width instead of KeyError
        self.interval_width = Config.interval_width
        self.interval_coef = Config.interval_coef

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

        # 학습 모델 및 학습 결과
        self.training_model = {}
        self.total_bizday_status = {}
        self.train_result = {}

        # set module parameter
        self.parameter = None
        self.init_param(config)

        # biz day 학습 완료 여부 return 목적
        self.train_business_status = dict()

    def init_param(self, config):
        # set module parameter
        self.parameter = config.get("parameter", None)
        if self.parameter is not None:
            self.training_features = self.parameter["train"]["dbsln"]["features"]
        else:
            self.training_features = None
        #self.logger.info("Success dynamic_baseline_for_service init_param !!")

    # detect outlier
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
        for i in wday_data[0 : len(wday_data)]:
            outlier = self.outlier_iqr(i)[0]
            mask_arr = np.ma.array(i, mask=False)
            mask_arr.mask[outlier] = True
            for out in outlier:
                i[out] = mask_arr.mean()

        return wday_df

    def interpolate_data_using_weekday_minute(self, null_df, not_null_df):
        """
        요일별 1분단위 대표값을 이용해서 결측치를 전처리한다.
        전체 학습 데이터에서 결측된 Row와 결측되지 않은 Row를 분리한 다음, 결측되지 않은 Row들을 통해 요일별 분단위 대표값을 구한다.
        그리고 결측된 Row들을 요일별 분단위 대표값을 대체하고, 남은 값들은 linear interpolation한다.
        interpolated된 학습 데이터와 결측되지 않은 학습 데이터를 합쳐서 최종 학습 데이터를 만든다.
        요일별 분단위 대표값으로 대체되는 Row는 기존에 부하가 발생하는 시간대에 장애가 발생한 경우이다.
        그리고 요일별 분단위 대표값으로도 대체되지 않는 Row는 기존에 부하가 발생하지 않았던 시간대(은행 점검 시간 등등...)이다.
        :param train_df: 학습 데이터
        :return: 보간된 학습 데이터(train_df)
        """
        null_df.reset_index(inplace=True)
        null_df.set_index(["weekday", "minute"], inplace=True)
        null_df.rename(columns={"index": "tstmp"}, inplace=True)
        mean_value_df = not_null_df.groupby(["weekday", "minute"]).mean()
        common_index = null_df.index.intersection(mean_value_df.index, sort=False)

        interpolated_df = null_df.combine_first(mean_value_df.loc[common_index])
        interpolated_df.sort_values(by="yymmdd", inplace=True)
        interpolated_df["yymmdd"] = interpolated_df["tstmp"].map(
            lambda x: x.strftime(bc.INPUT_DATE_FORMAT)
        )
        interpolated_df.reset_index(inplace=True)
        interpolated_df.set_index("tstmp", inplace=True)

        train_df = pd.concat([interpolated_df, not_null_df])
        train_df.sort_index(inplace=True)
        train_df.fillna(0, inplace=True)

        return train_df

    def check_if_bizday_is_missing_index(self, train_days, idx):
        """
        1. self.business_list에 담긴 bizday가 실제로 학습데이터에서 결측됐는지 확인
        2. bizday_status에는 결측되지 않은 bizday 중 이틀 이상 포함된 날짜를 담고, df_biz에는 결측되지 않은 모든 bizday 정보를 담는다.
        :param train_days:학습 데이터
        :return: df_biz(pandas.DataFrame)
        """

        df_biz = pd.DataFrame()
        bizday_status = dict()
        if len(self.business_list) == 0:
            df_biz_idx = []
        elif len(self.business_list) != 0:
            for i in range(len(self.business_list)):
                df_biz = df_biz.append(
                    pd.DataFrame(self.business_list[i]), ignore_index=True
                )
            df_biz["date"] = df_biz["date"].map(
                lambda x: datetime.datetime.strptime(x, bc.INPUT_DATE_YMD).strftime(
                    bc.INPUT_DATE_FORMAT
                )
            )
            df_biz["included_train"] = df_biz["date"].map(lambda x: x in train_days)
            df_biz = df_biz[df_biz["index"] == idx]
            df_biz_idx = np.unique(df_biz["index"].values)
            for idx in df_biz_idx:
                bizday_status[idx] = (
                    sum(df_biz[df_biz["index"] == idx]["included_train"].values) >= 1
                )

        return df_biz, df_biz_idx, bizday_status

    def pre_model_preprocessing(self, x_test_data, y_test_data=None, feat=None):

        return list(x_test_data.index.values), y_test_data

    def train_data_drop_failure_and_business(self, train_df, except_failure_date_list, except_business_list):
        failure_dates = list()
        if len(except_failure_date_list) != 0:
            train_df, failure_dates = aicommon.Utils.drop_failure_date(
                except_failure_date_list, train_df
            )

        except_biz_dates = list()
        if len(except_business_list) != 0:
            train_df, except_biz_dates = aicommon.Utils.drop_except_business_list(
                except_business_list, train_df
            )

        return train_df, failure_dates, except_biz_dates

    def train_data_preprocessing(self, train_df, multiprocessing, biz_data=False):
        train_df = train_df.fillna(0)
        train_df = train_df[~train_df.index.duplicated()]

        if not biz_data:
            new_idx = pd.date_range(train_df.index[0], train_df.index[-1], freq="1min")
            train_df = train_df.reindex(new_idx)

        not_null_tstmp = train_df[train_df.notnull().values].index.drop_duplicates()
        training_from_date = train_df.index[0].strftime(bc.INPUT_DATE_FORMAT)
        training_to_date = train_df.index[-1].strftime(bc.INPUT_DATE_FORMAT)

        if not multiprocessing:
            self.logger.info(
                f"[fit] training from date : training to date - {training_from_date} : {training_to_date}"
            )

        if "yymmdd" not in train_df.columns:
            train_df["yymmdd"] = train_df.index.map(
                lambda x: x.strftime(bc.INPUT_DATE_FORMAT)
            )
        train_df["weekday"] = train_df.index.map(lambda x: x.weekday())
        train_df["minute"] = train_df.index.map(lambda x: x.hour * 60 + x.minute)
        train_df["notnull"] = train_df.index.map(lambda x: x.isin(not_null_tstmp))
        null_df = train_df.query(f"notnull == {False}")
        not_null_df = train_df.query(f"notnull == {True}")
        null_df.drop(columns="notnull", inplace=True)
        not_null_df.drop(columns="notnull", inplace=True)

        train_df = self.interpolate_data_using_weekday_minute(null_df, not_null_df)

        return train_df

    def preprocessing_for_business_train_data(self, biz_df_xcd_dict, weekday_map, multiprocessing):
        biz_train_dict = None
        learning_bizday_index_dict = None
        total_bizday_train_dict = None
        if biz_df_xcd_dict is not None:
            learning_bizday_index_dict = dict()
            biz_train_dict = dict()
            total_bizday_train_dict = dict()
            for idx in list(biz_df_xcd_dict.keys()):
                self.total_bizday_status[idx] = False
                biz_train_df = biz_df_xcd_dict[idx]
                biz_train_df["tstmp"] = pd.to_datetime(
                    biz_train_df.index, format=bc.INPUT_DATETIME_FORMAT
                )
                biz_train_df = biz_train_df.set_index("tstmp")
                biz_train_df, failure_dates, except_biz_dates = self.train_data_drop_failure_and_business(
                    biz_train_df, self.except_failure_date_list, self.except_business_list
                )
                if biz_train_df.shape[0] == 0:
                    continue
                biz_train_df = self.train_data_preprocessing(biz_train_df, multiprocessing, biz_data=True)
                biz_train_df = biz_train_df[~biz_train_df.index.duplicated()]
                biz_train_days = np.unique(biz_train_df["yymmdd"].values)

                if len(self.business_list) != 0:
                    df_biz, df_biz_idx, bizday_status = self.check_if_bizday_is_missing_index(biz_train_days, idx)
                    bizidx_status_list = list(bizday_status.items())
                    learning_bizday_index = list(
                        dict(filter(lambda x: x[1] == True, bizidx_status_list)).keys()
                    )
                    biz_train_df, weekday_map = aicommon.Utils.change_weekday_of_learning_bizday(
                        df_biz, learning_bizday_index, biz_train_df, weekday_map
                    )
                    bizday_train = aicommon.Utils.process_bizday_training_result(
                        self.business_list, df_biz, df_biz_idx
                    )
                else:
                    learning_bizday_index = list()
                    bizday_status = dict()
                    bizday_train = list()
                learning_bizday_index_dict[idx] = learning_bizday_index
                if bizday_status[idx]:
                    self.total_bizday_status[idx] = True
                total_bizday_train_dict[idx] = bizday_train
                biz_train_dict[idx] = biz_train_df
        else:
            return biz_train_dict, list(), None

        if learning_bizday_index_dict is not None:
            total_learning_bizday_index = sum(list(learning_bizday_index_dict.values()), [])

        if total_bizday_train_dict is not None:
            total_bizday_train = sum(list(total_bizday_train_dict.values()), [])

        return biz_train_dict, total_learning_bizday_index, total_bizday_train

    def remove_duplicated_train_data(self, train_df, biz_train_dict):
        if biz_train_dict is None:
            return train_df

        if biz_train_dict:
            total_biz_df = pd.concat([biz_df for biz_df in biz_train_dict.values()])
            return pd.concat([train_df, total_biz_df])
        else:
            return train_df

    def remove_spike_outliers(self, train_df, m=30):
        # by trend_noise_decomposition
        feats_data = train_df[self.training_features].values
        trend = np.zeros_like(feats_data)

        # m = 30
        for i in range(trend.shape[0]):
            trend[i] = np.mean(feats_data[np.max([0, i - m]):i + 1], axis=0)

        noise = np.maximum(0, feats_data - trend)

        for i, feat in enumerate(self.training_features):
            feat_data = feats_data[:, i].astype(float)
            noise_feat = noise[:, i]
            upper_bound = np.nanmean(noise_feat) + 3 * np.nanstd(noise_feat)

            feat_data[np.argwhere(noise_feat > upper_bound)] = np.nan
            train_df[feat] = feat_data

    def fit(self, target, X, biz_df_xcd_dict, multiprocessing=True):

        if not multiprocessing:
            self.logger.info(f"model {self.model_id} start training")

        elapse = time.time()
        train_df = X.copy()
        train_df["tstmp"] = pd.to_datetime(
            train_df.index, format=bc.INPUT_DATETIME_FORMAT
        )
        train_df = train_df.set_index("tstmp")

        if not multiprocessing:
            if len(train_df) == 0:
                self.logger.warn(f"[fit] train_df of xcode '{target}' is empty")
                return None, None, Errors.E800.value, Errors.E800.desc
            self.logger.info(
                f"[fit] model {self.model_id}/{target} start serial training"
            )
        else:
            fomatter = logging.Formatter(
                "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s"
            )
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(fomatter)

            process_id = os.getpid()
            process_logger = pathos.logger(level=logging.INFO, handler=streamHandler)
            if len(train_df) == 0:
                process_logger.error(f"[fit] train_df of xcode '{target}' is empty")
                return (
                    None,
                    None,
                    None,
                    Errors.E800.value,
                    Errors.E800.desc,
                    target,
                    {}
                )
            else:
                process_logger.info(
                    f"[fit] model {self.model_id}/{target} start multiprocessing training by pid {process_id}"
                )

        self.remove_spike_outliers(train_df)

        train_df, failure_dates, except_biz_dates = self.train_data_drop_failure_and_business(
            train_df, self.except_failure_date_list, self.except_business_list
        )
        train_df = self.train_data_preprocessing(train_df, multiprocessing)
        length_of_training_days = len(np.unique(train_df["yymmdd"]))

        # training_mode 설정
        self.training_mode = aicommon.Utils.calc_training_mode_for_service(length_of_training_days)
        if self.training_mode < 0:
            if not multiprocessing:
                raise ModuleException("E801")
            else:
                return (
                    None,
                    None,
                    None,
                    Errors.E800.value,
                    Errors.E800.desc,
                    target,
                    {}
                )
        weekday_map = copy.deepcopy(bc.TRAINING_WDAY_MAP[self.training_mode])

        biz_train_dict, total_learning_bizday_index, total_bizday_train = self.preprocessing_for_business_train_data(
            biz_df_xcd_dict, weekday_map, multiprocessing
        )
        total_train_df = self.remove_duplicated_train_data(train_df, biz_train_dict)

        # train each feature
        amodel = {}
        for feature_name in self.training_features:
            amodel[feature_name] = self.training_a_feature(
                total_train_df,
                feature_name,
                weekday_map,
                self.training_mode,
                total_learning_bizday_index,
                multiprocessing
            )

        target_model = {}
        for feature in self.training_features:
            target_model[feature] = {}
            df_lower = pd.DataFrame()
            df_upper = pd.DataFrame()
            df_avg = pd.DataFrame()
            df_std = pd.DataFrame()
            for wday in np.unique(amodel[feature]['df_avg'].columns):
                df_lower.loc[:, f"{wday}"] = amodel[feature]['df_avg'][wday].values - self.interval_coef * \
                                             amodel[feature]['df_std'][wday].values
                df_upper.loc[:, f"{wday}"] = amodel[feature]['df_avg'][wday].values + self.interval_coef * \
                                             amodel[feature]['df_std'][wday].values
                df_avg.loc[:, f"{wday}"] = amodel[feature]['df_avg'][wday].values
                df_std.loc[:, f"{wday}"] = amodel[feature]['df_std'][wday].values
            df_lower[df_lower < 0] = 0
            df_lower = df_lower.round(2)
            df_upper = df_upper.round(2)
            df_avg = df_avg.round(2)
            df_std = df_std.round(2)
            target_model[feature]["lower"] = df_lower
            target_model[feature]["upper"] = df_upper
            target_model[feature]["avg"] = df_avg
            target_model[feature]["std"] = df_std

            # combine [lower, upper, avg, std]
            combined_data = []
            for col in target_model[feature]["lower"].columns:
                combined_data.append(
                    pd.concat([target_model[feature][key][col] for key in target_model[feature].keys()],
                              axis=1).values.tolist())

            combined_df = pd.DataFrame(combined_data).T
            combined_df = combined_df.rename(
                columns={idx: value for idx, value in enumerate(target_model[feature]["lower"].columns)})
            target_model[feature] = combined_df

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

        self.training_model[target] = target_model

        train_elapsed_time = int((time.time() - elapse))
        report = {
            "mse": -1,
            "rmse": -1,
            "duration_time": train_elapsed_time,
            "hyper_params": None,
        }
        results = {
            "from_date": total_train_df.index[0].strftime(bc.INPUT_DATE_FORMAT),
            "to_date": total_train_df.index[-1].strftime(bc.INPUT_DATE_FORMAT),
            "except_failure_date_list": failure_dates,
            "except_business_list": except_biz_dates,
            "business_list": [] if total_bizday_train is None else total_bizday_train,
            "train_business_status": self.train_business_status,
            "train_mode": self.training_mode,
            "outlier_mode": self.outlier_mode,
            "train_metrics": dict.fromkeys(self.training_features),
            "results": report,
        }

        self.train_result[target] = results
        if not multiprocessing:
            self.logger.info("[fit] return training result and report")

        # 싱글 프로세스 학습
        if not multiprocessing:
            self.logger.debug(
                f"model {self.model_id}/{target} training result : {results}"
            )
            return results, None, 0, None

        # 멀티 프로세스 학습
        process_logger.info(
            f"model {self.model_id}/{target} training result by pid {process_id} : {results}"
        )
        return (
            self.training_model,
            self.train_result[target],
            self.training_mode,
            0,
            None,
            target,
            self.total_bizday_status,
        )

    def training_a_feature(
        self,
        train_df,
        feature_name,
        wday_map,
        training_mode,
        learning_bizday_index,
        multiprocessing_mode=False,
        print_stdout=True,
    ):
        n_half_window = int(self.window_size / 2)
        bc.DBSLN_WDAY_MAX = len(wday_map)

        if not multiprocessing_mode:
            self.logger.info(
                f"training feature {feature_name}: training_mode={training_mode}"
            )
        else:
            pass

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

        # default - weekly training mode
        raw_data = []
        for i in range(bc.DBSLN_WDAY_MAX * bc.DBSLN_DMIN_MAX):
            raw_data.append([])

        if print_stdout:
            print(f"> Processing {feature_name} data", end="", flush=True)

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

        if print_stdout:
            print("", flush=True)

        df_avg_data = np.full((bc.DBSLN_DMIN_MAX, bc.DBSLN_WDAY_MAX), np.nan)
        df_std_data = np.full((bc.DBSLN_DMIN_MAX, bc.DBSLN_WDAY_MAX), np.nan)

        if print_stdout:
            print(f"> Summarizing {feature_name} data", end="", flush=True)

        for j in range(bc.DBSLN_WDAY_MAX):
            for i in range(bc.DBSLN_DMIN_MAX):
                raw_index = bc.DBSLN_DMIN_MAX * j + i
                if len(raw_data[raw_index]) > 0:
                    window_data = raw_data[raw_index]
                    df_avg_data[i][j] = np.mean(window_data)
                    df_std_data[i][j] = np.std(window_data)

            if print_stdout:
                print(".", end="", flush=True)

        if print_stdout:
            print("", flush=True)

        df_avg = pd.DataFrame(
            df_avg_data, index=range(0, bc.DBSLN_DMIN_MAX), columns=wday_map
        )
        df_std = pd.DataFrame(
            df_std_data, index=range(0, bc.DBSLN_DMIN_MAX), columns=wday_map
        )

        df_avg_mean = np.repeat(df_avg.mean(axis=0).values.reshape(1, -1), bc.DBSLN_DMIN_MAX, axis=0)
        df_std_mod_data = np.maximum(df_std.values, df_avg_mean)
        df_std_mod = pd.DataFrame(
            df_std_mod_data, index=range(0, bc.DBSLN_DMIN_MAX), columns=wday_map
        )

        df_avg = df_avg.fillna(method='bfill')
        df_avg = df_avg.fillna(method='ffill')
        df_std_mod = df_std_mod.fillna(method='bfill')
        df_std_mod = df_std_mod.fillna(method='ffill')
        df_avg = df_avg.dropna(axis=1)
        df_std_mod = df_std_mod.dropna(axis=1)

        model = {"df_avg": df_avg, "df_std": df_std_mod}

        return model

    def _get_model_file_path(self, model_dir, target_id=None):
        if target_id is None:
            return str(Path(model_dir) / f"{self.model_id}.pkl")
        else:
            return str(Path(model_dir) / f"{self.model_id}_{target_id}.pkl")

    def save(self, model_dir):
        model_dir = Path(model_dir) / "dbsln"
        model_dir.mkdir(exist_ok=True, parents=True)

        for target_id in self.training_model.keys():
            file_path = self._get_model_file_path(model_dir, target_id=target_id)
            Path(file_path).write_bytes(pickle.dumps(self.training_model[target_id]))
            self.logger.info(f"model {target_id} / {self.model_id} saved to {file_path}")

    def save_multi_mode(self, model_dir, training_model, train_result):
        model_dir = Path(model_dir) / "dbsln"
        model_dir.mkdir(exist_ok=True, parents=True)

        self.logger.info(f"train_result : {train_result}")

        for target_id in training_model.keys():
            file_path = self._get_model_file_path(model_dir, target_id=target_id)
            Path(file_path).write_bytes(pickle.dumps(training_model[target_id]))
            self.logger.info(
                f"multiprocessed model {target_id} / {self.model_id} saved to {file_path}"
            )