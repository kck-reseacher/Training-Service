import datetime
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psutil
import psycopg2 as pg2
import psycopg2.extras

from common import constants as bc
from common.error_code import Errors

def sMAPE(y_true, y_pred, inverse=False):
    # symmetric Mean Absolute Percentage Error
    # y_true and y_pred must be inverse_scaled 1-D data
    def inverse_sMAPE(smape_val):
        return 100 - np.clip(smape_val, 0, 100)

    data = np.vstack([y_true.reshape(-1), y_pred.reshape(-1)]).T
    data[(data[:, 0] == 0) & (data[:, 1] == 0)] = 1e-7  # same as keras.epsilon
    smape = 100 * np.mean(np.abs(data[:, 0] - data[:, 1]) / (np.abs(data[:, 0]) + np.abs(data[:, 1])))

    return inverse_sMAPE(smape) if inverse else smape


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(JsonEncoder, self).default(obj)


class Utils:
    @staticmethod
    def get_module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(
            module_name, str(Path(file_path) / f"{module_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def usage():
        print("usage: command [OPTIONS] [MODULE]")
        print("    -m, --module   module name")
        print("    -t, --target   target name")
        print("    -p, --port     port number")
        print("    -s, --sys_id   system id")
        print("--inst_type        instance type [db, os, was] ")

    @staticmethod
    def get_class(module_name, class_name):
        m = __import__(module_name)
        m = getattr(m, class_name)
        return m

    @staticmethod
    def get_module_class(module_name, class_name, path):
        m = Utils.get_module_from_file(module_name, path)
        c = getattr(m, class_name)
        return c

    @staticmethod
    def to_camel_case(snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components[0:])

    @staticmethod
    def print_memory_usage(logger):
        """Prints current memory usage stats.
        See: https://stackoverflow.com/a/15495136

        :return: None
        """
        mega = 1024 * 1024
        svmem = psutil.virtual_memory()
        total, available, used, free = (
            svmem.total / mega,
            svmem.available / mega,
            svmem.used / mega,
            svmem.free / mega,
        )
        proc = psutil.Process(os.getpid()).memory_info()[1] / mega
        logger.warning(
            "process = %sMB total = %sMB available = %sMB used = %sMB free = %sMB percent = %s%%",
            proc,
            total,
            available,
            used,
            free,
            svmem.percent,
        )

    @staticmethod
    def validate_trainig_dataframe(dataframe: pd.core.frame.DataFrame, limit: int, logger):
        """
        학습 데이터를 점검하는 함수. 각 케이스별로 데이터프레임을 점검하며, 모두 통과해야 학습하도록 처리
        """
        result = {"code": 0, "message": "normal"}

        logger.info("[validate_training_dataframe] input dataframe")
        logger.info(f"[validate_training_dataframe] {dataframe}")

        if dataframe is None:
            logger.error(f"[validate_trainig_dataframe] {Errors.E806.desc}")

            result.update(code=Errors.E806.value, message=Errors.E806.desc)

            return result
        elif len(dataframe) == 0:
            logger.error(f"[validate_trainig_dataframe] {Errors.E800.desc}")

            result.update(code=Errors.E800.value, message=Errors.E800.desc)

            return result
        elif len(dataframe) < limit:
            logger.error(f"[validate_trainig_dataframe] {Errors.E801.desc}")
            logger.info(f"[validate_trainig_dataframe] Less training data than {limit}")

            result.update(code=Errors.E801.value, message=Errors.E801.desc)

            return result

        if bool(np.any(pd.isnull(dataframe))):
            logger.error(f"[validate_trainig_dataframe] {Errors.E821.desc}")

            result.update(code=Errors.E821.value, message=Errors.E821.desc)

            return result

        '''
        if not bool(dataframe.index.is_unique):
            logger.error(f"[validate_training_dataframe] {Errors.E822.desc}")
            logger.error(f"[validate_training_dataframe] duplicated index : {dataframe[dataframe.index.duplicated()]}")

            result.update(code=Errors.E822.value, message=Errors.E822.desc)

            return result
        '''

        logger.info("return dataframe since dataframe passed validation process")

        return dataframe

    @staticmethod
    def validate_feautures(features: list, dataframe, logger):
        result = {"code": 0, "message": "normal"}

        if len(features) == 0:
            logger.error(
                "[validate_feautures] failed training feature validation check"
            )

            result.update(code=Errors.E803.value, message=Errors.E803.desc)

            return result
        else:
            for feature in features:
                if feature not in list(dataframe.columns):
                    logger.error(
                        "[validate_feautures] provided feature '{}' is not in the training dataframe".format(
                            feature
                        )
                    )
                    message = (
                        f"provided feature '{feature}' is not in the training dataframe"
                    )
                    result.update(code=Errors.E804.value, message=message)

                    return result

        return None

    @staticmethod
    def min_max_scaler(data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)

        return numerator / (denominator + 1e-7)

    # calculate training mode
    @staticmethod
    def calc_training_mode(train_days):
        if train_days < 2:
            return bc.TRAINING_MODE_ERROR
        elif train_days < 7:
            return bc.TRAINING_MODE_DAILY
        elif train_days < 14:
            return bc.TRAINING_MODE_WORKINGDAY
        else:  # 8 weeks
            return bc.TRAINING_MODE_WEEKDAY

    # calculate training mode for service
    @staticmethod
    def calc_training_mode_for_service(train_days):
        if train_days < 2:
            return bc.TRAINING_MODE_ERROR
        elif train_days < 7:
            return bc.TRAINING_MODE_DAILY
        else:
            return bc.TRAINING_MODE_WORKINGDAY

    @staticmethod
    def drop_except_business_list(except_business_list, train_df):
        """
        1. 학습하지 않을 비즈니스데이를 학습 데이터에서 제외한 후 학습 데이터를 반환
        :param train_df:
        :return:
        """
        if "yymmdd" not in train_df.columns:
            if type(train_df.index[0]) is str:
                train_df.index = pd.to_datetime(train_df.index)
            train_df["yymmdd"] = train_df.index.map(lambda x: x.strftime(bc.INPUT_DATE_FORMAT))
        except_date = []

        for i in range(len(except_business_list)):
            except_date.append(except_business_list[i]["date"])

        except_biz = pd.DataFrame(sum(except_date, []), columns=["date"])
        except_biz["date"] = except_biz["date"].map(
            lambda x: datetime.datetime.strptime(x, bc.INPUT_DATE_YMD).strftime(bc.INPUT_DATE_FORMAT)
        )
        except_biz["except_idx"] = except_biz["date"].map(
            lambda x: train_df[train_df["yymmdd"] == x].index if x in train_df["yymmdd"].values else None
        )

        for idx in except_biz["except_idx"]:
            if idx is not None:
                train_df = train_df.drop(idx)

        return train_df, except_biz["date"].values

    @staticmethod
    def drop_failure_date(except_failure_date_list, train_df):
        """
        1. 장애가 발생한 날짜의 데이터를 학습 데이터에서 제외한 후 학습 데이터를 반환
        :param train_df: 학습 데이터
        :return: train_df(pd.DataFrame)
        """
        if "yymmdd" not in train_df.columns:
            if type(train_df.index[0]) is str:
                train_df.index = pd.to_datetime(train_df.index)
            train_df["yymmdd"] = train_df.index.map(lambda x: x.strftime(bc.INPUT_DATE_FORMAT))
        except_failure = pd.DataFrame(except_failure_date_list, columns=["date"])
        except_failure["date"] = except_failure["date"].map(
            lambda x: datetime.datetime.strptime(x, bc.INPUT_DATE_YMD).strftime(bc.INPUT_DATE_FORMAT)
        )
        except_failure["except_idx"] = except_failure["date"].map(
            lambda x: train_df[train_df["yymmdd"] == x].index if x in train_df["yymmdd"].values else None
        )

        for idx in except_failure["except_idx"]:
            if idx is not None:
                train_df = train_df.drop(idx)

        return train_df, except_failure["date"].values

    @staticmethod
    def change_weekday_of_learning_bizday(
        df_biz, learning_bizday_index, train_df, wday_map
    ):
        for business_index in learning_bizday_index:
            wday_map.append(business_index)
            indexed_df_biz = df_biz.query(f"index == '{business_index}'")
            for date in indexed_df_biz["date"].values:
                train_df["weekday"].loc[
                    date,
                ] = business_index

        return train_df, wday_map

    @staticmethod
    def process_bizday_training_result(business_list, df_biz, df_biz_idx):
        bizday_train = []
        bizday_train_result = {}
        if len(business_list) != 0:
            for idx in df_biz_idx:
                key = f"result_{idx}"
                if bizday_train_result.get(key) == None:
                    bizday_train_result[key] = dict()

                if sum(df_biz[df_biz["index"] == idx]["included_train"].values) == 0:
                    bizday_train_result[key]["index"] = idx
                    bizday_train_result[key]["biztype_name"] = df_biz[
                        df_biz["index"] == idx
                    ]["biztype_name"].values[0]
                    bizday_train_result[key]["result"] = bc.BIZDAY_NOT_IN_DATA
                elif sum(df_biz[df_biz["index"] == idx]["included_train"].values) == 1:
                    bizday_train_result[key]["index"] = idx
                    bizday_train_result[key]["biztype_name"] = df_biz[
                        df_biz["index"] == idx
                    ]["biztype_name"].values[0]
                    bizday_train_result[key]["result"] = bc.BIZDAY_LESS_THAN_TWO_DAYS
                elif sum(df_biz[df_biz["index"] == idx]["included_train"].values) > 1:
                    bizday_train_result[key]["index"] = idx
                    bizday_train_result[key]["biztype_name"] = df_biz[
                        df_biz["index"] == idx
                    ]["biztype_name"].values[0]
                    bizday_train_result[key]["result"] = bc.BIZDAY_TRAINED

                bizday_train.append(bizday_train_result[key])

        return bizday_train

    @staticmethod
    def set_except_msg(result_dict):
        return {"errno":result_dict["error_code"],"errmsg":result_dict["error_msg"]}


    @staticmethod
    def decimal_point_discard(n, point):
        if "float" not in str(type(n)):
            return n
        return np.floor(n * pow(10, point)) / pow(10, point)

class Query:
    def __init__(self, db_conn_str, logger):
        """
        pg에 접속과 접속 종료 (close)를 관리하는 클래스
        프로그램 흐름 : __init__() -> connect() -> make_cursor() -> query()
        생성자에서 접속을 생성하고 커서를 만듬
        소멸자에서 접속을 닫고 커서를 reloase함.

        Examples
        ----------
        Query.CreateQuery() : factory 패턴으로 인스턴스를 생성험.

        Parameters
        ----------
        db_conn_str : ip, port, user, password 정보가 있는 postgresql 접속 정보
        logger : 로거
        """
        self.logger = logger
        self.conn = None
        self.cursor = None
        self.db_conn_str = db_conn_str
        self._init()

    def _init(self):
        self._connect()
        self._make_cursor()

    def _connect(self):
        try:
            self.conn = pg2.connect(self.db_conn_str)
            self.logger.info("[Query] connection get")
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while connect() - {e}"
            )

    def _make_cursor(self):
        try:
            self.cursor = self.conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[Query] cursor get")
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while make_cursor() - {e}"
            )

    def query(self, sql):
        result = None
        try:
            result = psql.read_sql(sql, self.conn)
            self.logger.info("[Query] read_sql()")
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while  psql.read_sql() - {e}"
            )
        finally:
            return result

    def cursor_execute(self, data_insert_query, input_tuple=None):
        try:
            start_time = datetime.datetime.now()
            if input_tuple is None:
                self.cursor.execute(data_insert_query)
            else:
                self.cursor.execute(data_insert_query, input_tuple)
            self.conn.commit()
            self.logger.info("[Query] complete insertion")
            end_time = datetime.datetime.now()
            self.logger.info(
                f"[Query] cursor_execute finished. it took {(end_time - start_time).total_seconds()} seconds"
            )
            return True
        except Exception as e:
            self.logger.info(
                f"[{type(self).__name__}] Unexpected exception occurred while  cursor_execute() - {e}"
            )

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    @staticmethod
    def CreateQuery(config, logger):
        # db
        db_conn_str = config.get("db_conn_str", None)
        if db_conn_str is None:
            pq_config = config["postgres"]
            db_conn_str = (
                f"host={pq_config['host']} "
                f"port={pq_config['port']} "
                f"dbname={pq_config['database']} "
                f"user={pq_config['id']} "
                f"password={pq_config['password']}"
            )
        db_query = Query(db_conn_str, logger)
        return db_query

    @staticmethod
    def update_module_status_by_training_id(db_conn_str, train_history_id, data: dict, train_process, progress, process_start_time=None):
        for k, v in data.items():
            if k == train_process:
                data[k][bc.PROGRESS] = progress
                data[k][bc.DURATION_TIME] = process_start_time if process_start_time is not None else None
                break

        Query.update_training_progress(db_conn_str, train_history_id, data)

    @staticmethod
    def update_training_progress(db_conn_str, train_history_id, data):
        conn = None
        cursor = None

        try:
            conn = pg2.connect(db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)

            statement = (
                "UPDATE ai_history_train "
                "SET module_status = %(data)s "
                "WHERE history_train_id = %(train_history_id)s "
            )

            cursor.execute(statement, {"data": json.dumps(data), "train_history_id": train_history_id})

            conn.commit()

        except Exception as e:
            print(f"[Error] Unexpected exception during serving : {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
