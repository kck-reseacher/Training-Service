import datetime
import json
import os
import pickle
import timeit

import numpy as np
import pandas as pd
import psycopg2 as pg2
import psycopg2.extras
from sklearn.cluster import KMeans

from analyzer import aimodule
from common import aicommon

POSTGRES = "postgres"
HOST = "host"
PORT = "port"
DATABASE = "database"
ID = "id"
PASSWORD = "password"
FROM_DATE = "from_date"
TO_DATE = "to_date"
TRAIN_DIR = "train_dir"
MODEL_DIR = "model_dir"
LOG_DIR = "log_dir"
SYS_ID = "sys_id"
TARGET_ID = "target_id"
INST_TYPE = "inst_type"
TXN_ID = "txn_id"
EXECUTION_COUNT = "execution_count"
ELAPSTED_TIME = "elapsed_time"
CPU_TIME = "cpu_time"
SQL_ELAPSED = "sql_elapsed"
FETCH_TIME = "fetch_time"
DB_CONN_COUNT = "db_conn_count"
OPEN_CONN_COUNT = "open_conn_count"
REMOTE_TIME = "remote_time"
EXCEPTION_COUNT = "exception_count"
CLUSTER_VALUE_01 = "cluster_value01"
CLUSTER_VALUE_02 = "cluster_value02"
CLUSTER_VALUE_03 = "cluster_value03"
CLUSTER_VALUE_04 = "cluster_value04"
CLUSTER_VALUE_05 = "cluster_value05"
AVG_ELAPSED_TIME = "avg_elapsed_time"
AVG = "avg"
INDEX = "index"
TXN_NAME = "txn_name"
CLUSTER_NUMBER = "cluster_number"
LABEL = "label"
TXN_ID_LIST = "txn_id_list"
CLUSTER_ID = "clusterID"
SIZE = "size"
TIME = "time"
TXN_VALUES = "txn_values"


class ExemAiopsWclstWas(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        initialize instance attributes using JSON config information and Confluence documentation. AI Server inputs the JSON config information.
        :param config:
        :param logger:
        """
        self.config = config
        self.logger = logger

        self.db_conn_str = config["db_conn_str"]
        self.from_date = str(config["from_date"])
        self.to_date = str(config["to_date"])

        self.model_dir = config["model_dir"]
        self.log_dir = config["log_dir"]
        self.train_dir = config["train_dir"]

        self.inst_type = config["inst_type"]
        self.sys_id = config["sys_id"]
        self.target_id = config["target_id"]
        self.module = config["module"]

        self.model_id = f"{self.module}_{self.inst_type}_{self.target_id}"
        self.master_id = 0

        self.cluster_core_metrics = {}
        self.metric_filtering_thresholds = 3
        self.txn_filtering_thresholds = 10
        self.number_of_clusters = 5

        self.column_names = [
            TXN_ID,
            EXECUTION_COUNT,
            ELAPSTED_TIME,
            CPU_TIME,
            SQL_ELAPSED,
            FETCH_TIME,
            OPEN_CONN_COUNT,
            REMOTE_TIME,
            EXCEPTION_COUNT,
        ]
        # self.column_names = ['txn_id', 'execution_count', 'elapsed_time', 'cpu_time', 'sql_elapsed', 'fetch_time', 'remote_time']
        self.cluster_names = [
            CLUSTER_VALUE_01,
            CLUSTER_VALUE_02,
            CLUSTER_VALUE_03,
            CLUSTER_VALUE_04,
            CLUSTER_VALUE_05,
        ]
        self.stat_names = [EXCEPTION_COUNT, ELAPSTED_TIME, CPU_TIME]

        self.detail = {}

        # set printoption of float-type data
        np.set_printoptions(precision=2, suppress=True)
        pd.set_option("display.float_format", "{:.2f}".format)

    # 로딩된 CSV파일들을 전처리해서 Training Dataframe을 만드는 과정.
    def make_training_dataframe(self, input_df):
        """
        input : pandas dataframe
                columns : [sql_id, execution_count, row_count, elapseed_time, cpu_time, wait_time, logical_reads, physical_reads, redo_size, table_scan]
                index : 'sql_id', record : the other columns
        output: preprocessed pandas dataframe, queries information dictionary
        detail: transform input dataframe to training dataframe
        """
        self.logger.info("[make_training_dataframe] start making training_dataframe")

        execution_count_column = input_df[EXECUTION_COUNT].map(
            lambda x: 1 if x < 1 else x
        )
        train_df_columns_list = [execution_count_column]

        sliced_input_df = input_df[
            [
                ELAPSTED_TIME,
                CPU_TIME,
                SQL_ELAPSED,
                FETCH_TIME,
                OPEN_CONN_COUNT,
                REMOTE_TIME,
                EXCEPTION_COUNT,
            ]
        ]
        # sliced_input_df = input_df[['elapsed_time', 'cpu_time', 'sql_elapsed', 'fetch_time', 'remote_time']]

        for column in sliced_input_df.columns:
            temp = sliced_input_df[column] / execution_count_column
            train_df_columns_list.append(temp)

        train_df = pd.concat(train_df_columns_list, axis=1)
        train_df[TXN_NAME] = input_df[TXN_NAME]
        train_df.columns = (
                [EXECUTION_COUNT] + list(sliced_input_df.columns) + [TXN_NAME]
        )

        transaction_df = input_df[[AVG_ELAPSED_TIME, TXN_NAME]]
        transaction_df.columns = [AVG, TXN_NAME]
        transaction_df.reset_index(inplace=True)
        transaction_df[INDEX] = list(range(transaction_df.shape[0]))
        transaction_df.set_index([TXN_ID, TXN_NAME], append=True, inplace=True)

        transactions = transaction_df.to_dict(orient=INDEX)

        self.logger.info("[make_training_dataframe] finish making training dataframe")

        return train_df, transactions

    def init_train(self):
        pass

    def train(self, stat_logger):
        """
        training model and serving results

        :param stat_logger:
        :return:
        """
        self.logger.info("<=========== [training] Start Training ===========>")
        body = {"results": False, "master_id": None}
        start_time = timeit.default_timer()

        from_date_string = datetime.datetime.strptime(
            self.from_date, "%Y%m%d"
        ).strftime("%Y-%m-%d")
        to_date_string = datetime.datetime.strptime(self.to_date, "%Y%m%d").strftime(
            "%Y-%m-%d"
        )

        file_path_list = self.get_csvfile_list(
            self.config[TRAIN_DIR], self.target_id, from_date_string, to_date_string
        )
        input_df = self.load_csvfiles(file_path_list)

        result = aicommon.Utils.validate_trainig_dataframe(input_df, self.number_of_clusters, self.logger)
        if type(result) == dict:
            return None, body, result["code"], result["message"]
        else:
            input_df = result

        # input_df time field sort
        input_df['time'] = pd.to_datetime(input_df['time'])
        input_df = input_df.sort_values('time')
        input_df['time'] = input_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        input_df = input_df.set_index(TXN_ID)

        self.logger.info(f"[training] clustering workload of inst_id={self.target_id}")
        self.logger.info("[training] prepare input data")
        self.logger.info(f"{input_df.head(10)}")

        self.logger.info("[training] prepare training data")

        train_df, transactions = self.make_training_dataframe(input_df)
        train_data = train_df[
            [column for column in train_df.columns if column != TXN_NAME]
        ].values
        train_data = train_data.astype(np.float32)

        self.logger.info("[training] Scale Data")

        scaled_data = aicommon.Utils.min_max_scaler(train_data)

        self.logger.debug(f"{scaled_data}")

        # Clustering with Input data
        kmeans_with_random_init = KMeans(n_clusters=self.number_of_clusters)
        kmeans_with_random_init = kmeans_with_random_init.fit(scaled_data)
        cluster_centers_with_random_init = kmeans_with_random_init.cluster_centers_
        temp_cluster_centers = pd.DataFrame(
            cluster_centers_with_random_init,
            columns=self.column_names[1:],
            index=np.arange(cluster_centers_with_random_init.shape[0]),
        )
        temp_cluster_centers = temp_cluster_centers.applymap(
            lambda x: 0 if x < 0 else x
        )
        temp_cluster_centers.index.name = CLUSTER_NUMBER
        cluster_centers_with_random_init = temp_cluster_centers.values
        label = kmeans_with_random_init.predict(scaled_data)

        self.logger.info(
            "[training] fitting to scaled input Data with random initial point"
        )
        self.logger.info(f"{temp_cluster_centers}")

        # Argsort
        eucledian_distance_square = np.sum(
            np.square(cluster_centers_with_random_init), axis=1
        )
        sort_index = eucledian_distance_square.argsort()[::-1]
        sorted_centers = np.take(cluster_centers_with_random_init, sort_index, axis=0)

        self.logger.info(
            "[training] sort clusters depending on eucledian distance from the origin"
        )

        # Clustering with Init Centers
        kmeans_with_sorted_init = KMeans(
            n_clusters=self.number_of_clusters, init=sorted_centers, n_init=1
        )
        kmeans_with_sorted_init = kmeans_with_sorted_init.fit(scaled_data)
        cluster_centers_with_sorted_init = kmeans_with_sorted_init.cluster_centers_

        self.logger.info("[training] truncate negative cluster centers to zero")

        centers = pd.DataFrame(
            cluster_centers_with_sorted_init,
            columns=self.column_names[1:],
            index=list(range(1, 1 + cluster_centers_with_sorted_init.shape[0])),
        )
        centers = centers.applymap(lambda x: 0 if x < 0 else x)
        centers.index.name = CLUSTER_NUMBER
        self.cluster_ids = list(centers.index)
        centers_for_serving = centers.copy()

        # Predicting labels
        labels = kmeans_with_sorted_init.predict(scaled_data)

        self.logger.info(
            "[training] fitting to scaled input Data with sorted initial point"
        )
        self.logger.info(f"{centers}")

        for row_index in self.cluster_ids:
            row_series = centers.loc[row_index]
            sorted_row = row_series.sort_values(ascending=False)
            core_metric_values = sorted_row[: self.metric_filtering_thresholds]
            core_metric_names = sorted_row[: self.metric_filtering_thresholds].index

            if self.cluster_core_metrics.get(row_index) == None:
                self.cluster_core_metrics[row_index] = dict(
                    zip(core_metric_names, core_metric_values)
                )

        self.logger.info("[training] find core metrics of each cluster")
        self.logger.info(f"{self.cluster_core_metrics}")

        input_df[LABEL] = labels + 1

        self.logger.info("[training] add cluster label of sql on input_df")

        serving_df = input_df.reset_index().copy()

        self.logger.info("[training] create serving_df from input_df")

        for key, transaction in transactions.items():
            label = labels[transaction[INDEX]]  # label: int type
            transaction[LABEL] = label

        for cluster_id in pd.unique(self.cluster_ids):
            cluster_df = serving_df.query(f"{LABEL} == '{cluster_id}'")
            cluster_metrics = list(self.cluster_core_metrics[cluster_id].keys())

            sorted_cluster_df = cluster_df.sort_values(
                by=cluster_metrics, ascending=False
            )

            if sorted_cluster_df.shape[0] < self.txn_filtering_thresholds:
                sliced_cluster_df = sorted_cluster_df.iloc[: sorted_cluster_df.shape[0]]
            else:
                sliced_cluster_df = sorted_cluster_df.iloc[
                                    : self.txn_filtering_thresholds
                                    ]

            for row_number in list(range(sliced_cluster_df.shape[0])):
                record = sliced_cluster_df.iloc[row_number]

                # 클러스터 상세 정보 저장
                if self.detail.get(cluster_id):
                    self.detail[cluster_id][TIME].append(record[TIME])
                    self.detail[cluster_id][TXN_ID_LIST].append(
                        record[TXN_ID]
                    )  # TODO  key: tuple type (index, txn_id)에 txn_name 추가
                    self.detail[cluster_id][TXN_NAME].append(record[TXN_NAME])
                    self.detail[cluster_id][TXN_VALUES]["body"].append(
                        list(record[cluster_metrics].values)
                    )
                else:
                    self.detail[cluster_id] = {}
                    self.detail[cluster_id][CLUSTER_ID] = cluster_id  # cluster_id는 1 부터
                    # self.detail[label]['total'] = len(transactions)
                    self.detail[cluster_id][TIME] = [record[TIME]]
                    self.detail[cluster_id][TXN_ID_LIST] = [record[TXN_ID]]
                    self.detail[cluster_id][TXN_NAME] = [record[TXN_NAME]]
                    self.detail[cluster_id][TXN_VALUES] = {
                        "header": cluster_metrics,
                        "body": [
                            list(record[cluster_metrics].values)
                        ]
                    }  # TODO txn_name 추가, txn_name 중복 있을 수 있는지 확인

        # 클러스터 상세 정보 저장
        for key, item in self.detail.items():
            # item_df = pd.DataFrame(item)
            # item[TXN_ID_LIST] = list(item_df[TXN_ID_LIST].values)
            self.detail[key][SIZE] = len(self.detail[key][TXN_ID_LIST])

        # Save Fitted Model
        with open(os.path.join(self.model_dir, f"{self.model_id}.pkl"), "wb") as pickle_file:
            pickle.dump(from_date_string, pickle_file)
            pickle.dump(to_date_string, pickle_file)
            pickle.dump(kmeans_with_sorted_init, pickle_file)
            pickle.dump(centers, pickle_file)
            pickle.dump(transactions, pickle_file)

        centers.reset_index(inplace=True)
        # master_id = self.select_master_id()
        overall_result, serving_result, errcode, errmsg = self.serve(serving_df, transactions, centers_for_serving)
        if errcode != 0:
            return None, body, errcode, errmsg

        report = {
            "train_m": input_df.shape[0],
            "from_date": from_date_string,
            "to_date": to_date_string,
            "master_id": self.master_id
        }

        result = {"result": True, "report": report}
        end_time = timeit.default_timer()
        body = {"results" : True, "master_id" : self.master_id}

        self.logger.info(
            f"[training] training result of inst_id={self.target_id} is {result['result']} "
        )
        self.logger.info(f"[training] {report}")
        self.logger.info(
            f"[training] Training Total Working Time is {end_time - start_time:.2f} sec"
        )
        self.logger.info("<=========== End of Training ===========>")

        return result, body, 0, None

    def end_train(self):
        pass

    def serve(self, serving_df, transactions, centers_for_serving):
        """
        Do serving. Serving consists of two steps. First is creating data for line chart and radar chart, second is insert the data into database
        :param serving_df
            - DataFrame to be Served
        :param transactions
            - Dictionary including id, average and standard deviation of elapsed time
        :param centers_for_serving:
            - Numpy ndarray. Clustering center information to be served
        :return:
            - result : Boolean
            - serving_result_by_stat : DataFrame including serving result group by stat_name.
        """

        self.logger.info("<========= Start Serving =========>")

        start_time = timeit.default_timer()

        from_date_string = datetime.datetime.strptime(
            self.from_date, "%Y%m%d"
        ).strftime("%Y-%m-%d")
        to_date_string = datetime.datetime.strptime(self.to_date, "%Y%m%d").strftime(
            "%Y-%m-%d"
        )
        serving_df.set_index([TIME, TXN_ID, TXN_NAME], inplace=True)
        serving_df[INDEX] = list(range(len(serving_df)))

        try:
            self.insert_train_info_into_master_table()
            # Line Chart를 그릴 stat_name별로 다음 과정을 반복
            for stat_name in self.stat_names:
                self.logger.info(f"<===== start serving of '{stat_name}' =====>")

                if stat_name not in serving_df.columns:
                    return None, None, -1, f"Invalid stat name '{stat_name}'"

                # 결과를 저장할 target_df와 target_row를 만든다.
                # target_df는 [cluster_value01, cluster_value02, cluster_value03, cluster_value04, cluster_value05] 형태로 구성
                target_df_list = []
                target_row = [0 for _ in range(len(self.cluster_names))]
                prev_time = "0000-00-00 00:00:00"

                # serving_df를 한 줄씩 읽어들인다.
                # 컬럼은 [time, txn_id, execution_count, elapsed_time, avg_elapsed_time, cpu_time, sql_elapsed, ...] 식으로 구성
                # 이 때, index는 [time, txn_id]이고 나머지 컬럼들이 record를 구성한다.
                for index, record in serving_df.iterrows():
                    time = index[0]
                    txn_id = index[1]
                    txn_name = index[2]

                    if prev_time != time:
                        if prev_time != "0000-00-00 00:00:00":
                            target_row.insert(0, prev_time)
                            target_df_list.append(target_row)
                        target_row = [0 for _ in range(len(self.cluster_names))]

                    # compose target_row with query information
                    # transactions은 {('index', 'txn_id') : {'index' : ':index', 'avg' : ':avg', 'label' : ':label'}}로 구성
                    transaction = transactions[(record[INDEX], txn_id, txn_name)]
                    label = transaction[LABEL]

                    # serving_df도 time, txn_id, elapsed_time, execution_count, cpu_time, 그 외의 컬럼으로 구성
                    # txn_id가 해당하는 군집 번호에 해당하는 컬럼에 해당 txn에서 현재 계산되는 stat_name의 stat_value를 더한다.
                    target_row[label] = target_row[label] + record[stat_name]
                    prev_time = time

                if prev_time != "0000-00-00 00:00:00":
                    target_row.insert(0, prev_time)
                    target_df_list.append(target_row)

                target_df = pd.DataFrame(
                    target_df_list, columns=[TIME] + self.cluster_names
                )

                serving_results_by_stat = target_df.to_dict(orient="records")

                report = {
                    "from_date": from_date_string,
                    "to_date": to_date_string,
                    "stat_name": stat_name,
                }

                result = {"result": True, "report": report}

                self.logger.info(
                    f"[serving] serving result of inst_id={self.target_id} is {result['result']} "
                )
                self.logger.info(f"[serving] report - {report}")
                self.insert_serving_result_for_line(self.master_id, stat_name, target_df)

            # Radar_chart Serving
            column_names_1 = self.column_names[1:]
            centers_values = centers_for_serving[column_names_1].values

            self.insert_serving_result_for_radar(self.master_id, centers_values)

            end_time = timeit.default_timer()

            self.logger.info(
                f"[serving] Serving working time is {end_time - start_time:.2f} sec"
            )
            self.logger.info("<======== [serving] end serving ========>")

            return result, serving_results_by_stat, 0, None
        except Exception as e:
            return None, None, -1, "Unknown exception occured while was workload clustering serving"

    def insert_train_info_into_master_table(self):
        conn = None
        cursor = None

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[insert_train_info_into_master_table] : Connection Get!")

            data_insert_query = (
                "with rows as( "
                "INSERT INTO aiops_module_wclst_master "
                "(sys_id, inst_type, target_id, from_date, to_date)"
                "VALUES (%s, %s, %s, %s, %s) RETURNING master_id"
                ") select master_id from rows"
            )
            cursor.execute(
                data_insert_query,
                (
                    self.sys_id,
                    self.inst_type,
                    self.target_id,
                    self.from_date,
                    self.to_date
                ),
            )
            conn.commit()

            # get master_id
            rows = cursor.fetchone()
            master_id = rows[0]
            self.master_id = master_id

            self.logger.info(
                "[insert_train_info_into_master_table] : Complete Values Insertion!"
            )
        except Exception as exception:
            self.logger.error(
                f"Unexepected exception during inserting train info into master table : {exception}"
            )
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

            self.logger.info("[insert_train_info_into_master_table] : Connection close")
            self.logger.info(
                "<======== End Inserting Train Info into Master Table ========>"
            )

    # def select_master_id(self):
    #     self.logger.info(
    #         "<======== [select_master_id] : start select_master_id() ========>"
    #     )
    #
    #     conn = None
    #     cursor = None
    #
    #     try:
    #         conn = pg2.connect(self.db_conn_str)
    #         cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
    #
    #         self.logger.info("[select_master_id] : Connection get!")
    #
    #         cursor.execute("SELECT nextval('aiops_module_wclst_master_seq')")
    #         rows = cursor.fetchone()
    #         master_id = rows[0]
    #
    #         self.logger.info(
    #             f"[select_master_id] : select_master_id() >>> {str(master_id)} "
    #         )
    #     except Exception as e:
    #         self.logger.error(
    #             "[select_master_id] Unexpected exception during getting master_id from aiops_module_wclst_master_seq - ",
    #             e,
    #         )
    #     else:
    #         if cursor:
    #             cursor.close()
    #         if conn:
    #             conn.close()
    #         self.logger.info("[select_master_id] : Connection Close!")
    #
    #         return master_id

    def insert_serving_result_for_line(self, master_id, stat_name, target_df):
        """
        Connect PostgreSQL DB and insert result into the database
        :param stat_name: one of 'execution_count, elapsed_time, cpu_time'
        :param serving_results_by_stat: result from serving function.
        :return:
        """
        self.logger.info("<==== Start insert_serving_result_for_line ====>")
        self.logger.info(f"stat_name is '{stat_name}'")

        start_time = timeit.default_timer()
        conn = None
        cursor = None
        original_columns_index = target_df.columns
        new_columns_index = [
            "master_id",
            "stat_name"
        ]
        target_df[new_columns_index[0]] = master_id
        target_df[new_columns_index[1]] = stat_name
        target_df = target_df.reindex(
            columns=new_columns_index + list(original_columns_index)
        )

        # DB는 6개, WAS는 5개의 클러스터로 데이터를 군집화했다.
        # DB 부하패턴 클러스터링과 동일한 테이블에 결과를 넣기 위해 6번 클러스터의 값을 Null로 채워준다.
        target_df["cluster_names06"] = None
        parameter_array = list(map(lambda x: tuple(x), target_df.values.tolist()))

        try:
            self.logger.info("[insert_serving_result_for_line] db_conn_str")
            self.logger.info(f"{self.db_conn_str}")

            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[insert_serving_result_for_line] Connection Get!")

            data_insert_query = (
                "INSERT INTO aiops_module_wclst_line "
                "(master_id, "
                'stat_name, "time" '
                ", cluster_value01, cluster_value02, cluster_value03, cluster_value04, cluster_value05, cluster_value06) "
                "VALUES %s"
            )

            pg2.extras.execute_values(cursor, data_insert_query, parameter_array)

            conn.commit()
            self.logger.info(
                "[insert_serving_result_for_line] Complete Values Insertion!"
            )
        except Exception as e:
            self.logger.error(
                "[insert_serving_result_for_line] Unexpected exception occured while inserting serving result into aiops_module_wclst_line ",
                e,
            )
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            self.logger.info("[insert_serving_result_for_line] Connection close")

            end_time = timeit.default_timer()

            self.logger.info(
                f"[insert_serving_result_for_line] WorkingTime is {end_time - start_time:.2f}"
            )
            self.logger.info("<==== End Inserting Serving Result For Line Chart ====>")

    def insert_serving_result_for_radar(self, master_id, centers_values):
        """
                connect postgreSQL Database and insert radar chart data into the database
        :param centers_values: result from serving functions.
        :return:
        """
        self.logger.info("<==== Start Inserting Serving Result For Radar Chart ====>")

        conn = None
        cursor = None

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[insert_serving_result_for_radar] Connection Get!")

            cluster_id = 1

            # TODO detail 컬럼 저장
            data_insert_query = (
                "INSERT INTO aiops_module_wclst_radar "
                "(master_id, "
                "cluster_id, stat_value01, stat_value02, stat_value03, stat_value04, stat_value05, stat_value06, stat_value07, stat_value08, detail) "
                "VALUES (%s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )

            for centers_value in centers_values:
                centers_value = centers_value.astype(float)
                cursor.execute(
                    data_insert_query,
                    (
                        master_id,
                        cluster_id,
                        centers_value[0],
                        centers_value[1],
                        centers_value[2],
                        centers_value[3],
                        centers_value[4],
                        centers_value[5],
                        centers_value[6],
                        centers_value[7],
                        json.dumps(self.detail[cluster_id], cls=aicommon.JsonEncoder),
                    ),
                )
                conn.commit()
                cluster_id += 1
            self.logger.info(
                "[insert_serving_result_for_radar] Complete Values Insertion!"
            )
        except Exception as e:
            self.logger.error(
                "Unexpected exception occured while inserting result into aiops_module_wclst_radar - ",
                e,
            )
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            self.logger.info("[insert_serving_result_for_radar] Connection Close!")

            self.logger.info("<==== End Inserting Serving Result For Radar Chart ====>")

