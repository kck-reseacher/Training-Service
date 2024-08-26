import datetime
import json
import os
import timeit
from copy import deepcopy
import numpy as np
import pandas as pd
import psycopg2 as pg2

from algorithms.prophet import akprophet
from analyzer import aimodule
from common import aicommon, constants
from common.error_code import Errors

class ExemAiopsLngtrmFcst(aimodule.AIModule):
    def __init__(self, config, logger):
        """
         장기부하예측: PROPHET

        Parameters
        ----------
        config
        logger
        """

        self.config = config
        self.logger = logger

        self.module_id = self.config["module"]
        self.inst_type = self.config['inst_type']
        self.target_id = str(self.config['target_id'])

        self.db_conn_str = self.config["db_conn_str"]
        self.config['predict_month'] = self.config.get('predict_month', 0)

        self.logger.info(f"{self.config}")

    def train(self, train_logger):
        start_time = timeit.default_timer()

        self.logger.info("[training] validate business day and failure date information")
        self.logger.info(f"[training] module start training")
        self.logger.info("[training] load dataframe")

        except_col_for_recent_length = list()
        self.insert_train_information()
        body = {
            "results": False,
            "master_id": self.master_id
        }

        try:
            df = pd.read_csv(os.path.join(self.config['train_dir'], "data.csv"))
        except Exception as e:
            self.logger.error(f"file {self.config['train_dir']} not exists", e)
            return None, body, Errors.E806.value, Errors.E806.desc

        self.logger.info("[training] validate training dataframe and features")
        if len(df) < 30:
            return None, body, Errors.E801.value, Errors.E801.desc

        dataframe = deepcopy(df[['time', 'target_id'] + self.config['collect_metric']])
        dataframe_using_insert = deepcopy(df[['time', 'target_id'] + self.config['collect_metric']])
        temp_collect_metric = deepcopy(self.config['collect_metric'])

        if dataframe.isnull().any().any():
            for col in temp_collect_metric:
                if np.isnan(df[col].iloc[-1]) or df[col].value_counts().sum() < 30:
                    dataframe.drop(columns=[col], inplace=True)
                    dataframe_using_insert.drop(columns=[col], inplace=True)
                    df.drop(columns=[col], inplace=True)
                    except_col_for_recent_length.append(col)
                    self.config['collect_metric'].remove(col)
        self.logger.info(f'[training] feat : {self.config["collect_metric"]}, except feature : {except_col_for_recent_length}')

        df_all = dataframe.copy()

        #학습으로 사용된 데이터 db에 저장
        self.insert_train_data(dataframe_using_insert, self.master_id)

        df_all['target_id'] = df_all['target_id'].astype(str)
        df_all['time'] = pd.to_datetime(df_all['time'], format=constants.INPUT_DATETIME_FORMAT)
        df_all['yyyy-mm-dd'] = df_all['time'].dt.strftime(constants.INPUT_DATE_FORMAT)

        self.logger.info(f"[training] inst_type is {self.inst_type}")
        try:
            if self.config['algorithm'] == 'prophet':
                self.trend_forecaster = akprophet(self.config, self.logger)
                self.logger.info(f"df_all.head() = {df_all.head()}")
                for feat in self.config['collect_metric']:
                    lower, upper = np.percentile(df_all[feat], [10, 99.6])
                    df_all.loc[(df_all[feat] > upper) | (df_all[feat] < lower), feat] = np.nan
                    df_all[feat].interpolate(limit_area='inside', limit_direction='both', inplace=True)
                    self.trend_forecaster.fit(feat, df_all[['time', feat]])
                result, _, errno, errmsg = self.serve(self.master_id, None, 0, None)
        except Exception as e:
            self.logger.warning(f"{e}")
            return None, 0, None, e

        self.logger.info(f"result = {result}")
        end_time = timeit.default_timer()
        if errno == 0:
            body["results"] = True

        self.logger.info(f"[training] training_result : {result}")
        self.logger.info(f"[training] WorkingTime: {end_time - start_time:.2f} sec")
        self.logger.info(f"[training] finish training")
        return result, body, errno, errmsg

    def serve(self, master_id, predict_time, data, test_datetime):
        self.logger.info("[serving] start serving")
        try:
            self.logger.info(f"Prophet serving")
            result_df = self.trend_forecaster.predict()
            predict_start_date = result_df.reset_index().iloc[0, 0]
            predict_end_date = result_df.reset_index().iloc[-1, 0]
            predict_date = pd.DataFrame(pd.date_range(predict_start_date, predict_end_date))

            if self.config['algorithm'] == 'prophet':
                for feature_name in self.config['collect_metric']:
                    feat_df = result_df[[f'{feature_name}_pred', f'{feature_name}_lower', f'{feature_name}_upper']]
                    predict_mean = np.clip(feat_df[f'{feature_name}_pred'], 0, None)
                    predict_lower = np.clip(feat_df[f'{feature_name}_lower'], 0, None)
                    predict_upper = feat_df[f'{feature_name}_upper']

                    predict_info = list(zip(np.round(predict_mean, 1), np.round(predict_upper, 1), np.round(predict_lower, 1)))
                    test_dict = dict(zip(predict_date.astype('str').values.ravel(), predict_info))

                    final_result = {
                        "header": ["time", "value", "upper", "lower"],
                        "body": [[date, value[0], value[1], value[2]] for date, value in test_dict.items()]
                    }

                    self.logger.info(f"final_result : {final_result}")
                    self.insert_prediction_result(master_id, feature_name, final_result)
        except Exception as e:
            self.logger.warning(f"{e}")
            return None, 0, None, e
        self.logger.info("[serving] finish serving")
        return self.trend_forecaster.training_result, {"results": True, "master_id" : master_id}, 0, None

    def insert_train_information(self):
        conn = None
        cursor = None

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[insert_train_information] : Connection Get!")

            predict_day = datetime.datetime.strptime(self.config['to_date'], '%Y%m%d') + datetime.timedelta(days=1)
            data_insert_query = (
                "with rows as( "
                "INSERT INTO ai_result_lngtrm_request "
                "(inst_type, target_id, predict_day, from_date, to_date, is_delete, predict_month, algorithm, analysis_data_type) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING result_lngtrm_request_id "
                ") select result_lngtrm_request_id from rows "
            )
            cursor.execute(
                data_insert_query,
                (
                    self.inst_type,
                    self.target_id,
                    predict_day.strftime('%Y%m%d'),
                    self.config['from_date'],
                    self.config['to_date'],
                    False,
                    self.config['predict_month'],
                    self.config['algorithm'],
                    self.config['analysisDataType']
                )
            )
            conn.commit()

            rows = cursor.fetchone()
            master_id = rows[0]
            self.master_id = master_id

            self.logger.info(
                "[insert_train_information] : Complete Values Insertion!"
            )
        except Exception as exception:
            self.logger.error(f"Unexepected exception during inserting train info into master table : {exception}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

            self.logger.info("[insert_train_information] : Connection close")

    def insert_train_data(self, df, master_id):
        self.logger.info("[insert_train_data] start inserting train")
        conn = None
        cursor = None

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[insert_train_data] connection get")

            query = (
                "INSERT INTO ai_result_lngtrm_past (inst_type, metric, value, past_date, target_id, result_lngtrm_request_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            )

            for idx, row in df.iterrows():
                for key in self.config['collect_metric']:
                    cursor.execute(query,
                                   (
                                       self.inst_type,
                                       key,
                                       np.round(row[key],2),
                                       row['time'],
                                       self.target_id,
                                       master_id
                                   )
                                   )

            self.logger.info("[insert_train_data] query executed")
            conn.commit()
            self.logger.info("[insert_train_data] commit completed")

        except Exception as e:
            self.logger.exception("xaiops_module_lngtrm_past 테이블 INSERT 오류.", e)
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        self.logger.info("[insert_train_data] finish inserting train data")

    def insert_prediction_result(self, master_id, stat_name, predict_values):
        self.logger.info(f"[insert_prediction_result] start inserting forecasted values of '{stat_name}' to pg")

        conn = None
        cursor = None

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            predict_values = json.dumps(predict_values)
            self.logger.info("[insert_prediction_result] connection get!")

            query = (
                "INSERT INTO ai_result_lngtrm_predict (result_lngtrm_request_id, metric, predict_values) "
                "VALUES (%s, %s, %s)"
            )

            cursor.execute(
                query,
                (
                    master_id,
                    stat_name,
                    predict_values
                )
            )

            self.logger.info("[insert_prediction_result] query executed!")
            conn.commit()
            self.logger.info("[insert_prediction_result] commit completed!")

        except Exception as e:
            self.logger.exception(
                f"aiops_module_longterm_predict 테이블 INSERT 오류: {str(e)}")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

        self.logger.info("[insert_prediction_result] finish inserting forecasted values to pg")