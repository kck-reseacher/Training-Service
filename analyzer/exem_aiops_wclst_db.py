import datetime
import json
import os
import pickle
import timeit

import numpy as np
import pandas as pd
import psycopg2 as pg2  # for window
import psycopg2.extras
from sklearn.cluster import KMeans

from analyzer import aimodule
from common import aicommon
from common.error_code import Errors

SYS_ID = "sys_id"
TARGET_ID = "target_id"
INST_TYPE = "inst_tye"
SQL_ID = "sql_id"
EXECUTION_COUNT = "execution_count"
ELAPSED_TIME = "elapsed_time"
CPU_TIME = "cpu_time"
CLUSTER_VALUE_O1 = "cluster_value01"
CLUSTER_VALUE_O2 = "cluster_value02"
CLUSTER_VALUE_O3 = "cluster_value03"
CLUSTER_VALUE_O4 = "cluster_value04"
CLUSTER_VALUE_O5 = "cluster_value05"
CLUSTER_VALUE_O6 = "cluster_value06"
WAIT_TIME = "wait_time"
LOGICAL_READS = "logical_reads"
PHYSICAL_READS = "physical_reads"
REDO_SIZE = "redo_size"
TABLE_SCAN = "table_scan"
CLUSTER_NAMES_06 = "cluster_names06"
CLUSTER_NUMBER = "cluster_number"
AVG = "avg"
STDDEV = "stddev"
AVG_ELAPSED_TIME = "avg_elapsed_time"
STDDEV_ELAPSED_TIME = "stddev_elapsed_time"
SQL_UID = "sql_uid"
INDEX = "index"
LABEL = "label"
SQL_ID_LIST = "sqlIdList"
SQL_UID_LIST = "sqlUidList"
CLUSTER_ID = "clusterID"
TOTAL = "total"
SIZE = "size"
TIME = "time"
MASTER_ID = "master_id"
INST_ID = "inst_id"
STAT_NAME = "stat_name"
SQL_VALUES = "sql_values"


class ExemAiopsWclstDb(aimodule.AIModule):
    def __init__(self, config, logger):
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

        self.detail = {}
        self.cluster_core_metrics = {}

        self.column_names = [
            SQL_ID,
            EXECUTION_COUNT,
            ELAPSED_TIME,
            CPU_TIME,
            WAIT_TIME,
            LOGICAL_READS,
            PHYSICAL_READS,
            REDO_SIZE,
            TABLE_SCAN,
        ]
        self.stat_names = [EXECUTION_COUNT, ELAPSED_TIME, CPU_TIME]
        self.cluster_names = [
            CLUSTER_VALUE_O1,
            CLUSTER_VALUE_O2,
            CLUSTER_VALUE_O3,
            CLUSTER_VALUE_O4,
            CLUSTER_VALUE_O5,
            CLUSTER_VALUE_O6,
        ]

        # 분석 파라미터
        self.number_of_clusters = 6
        self.metric_filtering_threshold = 3
        self.sql_filtering_threshold = 10

        # set printoption of float-type data
        np.set_printoptions(precision=2, suppress=True)
        pd.set_option("display.float_format", "{:.2f}".format)

    # 로딩된 CSV파일들을 전처리해서 Training Dataframe을 만드는 과정.
    def make_training_dataframe(self, input_df):
        """
        input : pandas dataframe
                columns : [sql_id, execution_count, row_count, elapseed_time, cpu_time, wait_time, logical_reads, physical_reads, redo_size, table_scan, sql_text]
                index : 'sql_id', record : the other columns
        output: preprocessed pandas dataframe, queries information dictionary
        detail: transform input dataframe to training dataframe
        """
        self.logger.info("[make_training_dataframe] start making training_dataframe ")

        execution_count_column = input_df[EXECUTION_COUNT].map(
            lambda x: 1 if x < 1 else x
        )
        train_df_columns_list = [execution_count_column]

        sliced_input_df = input_df[
            [
                ELAPSED_TIME,
                CPU_TIME,
                WAIT_TIME,
                LOGICAL_READS,
                PHYSICAL_READS,
                REDO_SIZE,
                TABLE_SCAN,
            ]
        ]

        for column in sliced_input_df.columns:
            temp = sliced_input_df[column] / execution_count_column
            train_df_columns_list.append(temp)

        train_df = pd.concat(train_df_columns_list, axis=1)
        train_df[SQL_UID] = input_df[SQL_UID]
        train_df.columns = (
            [EXECUTION_COUNT] + list(sliced_input_df.columns) + [SQL_UID]
        )

        query_df = input_df[[AVG_ELAPSED_TIME, STDDEV_ELAPSED_TIME, SQL_UID]]
        query_df.columns = [AVG, STDDEV, SQL_UID]
        query_df.reset_index(inplace=True)
        query_df[INDEX] = list(range(query_df.shape[0]))
        query_df.set_index([SQL_ID, SQL_UID], append=True, inplace=True)

        self.logger.debug("[make_training_dataframe] query_df")
        self.logger.debug(f"{query_df}")

        queries = query_df.to_dict(orient="index")

        self.logger.debug("[make_training_dataframe] queries")
        self.logger.debug(f"{queries}")

        self.logger.info("[make_training_dataframe] finish making training dataframe")

        return train_df, queries

    def init_train(self):
        pass

    # Training
    def train(self, stat_logger):
        """
        Input : from_date, to_date, train_dir
                from_date : 데이터 분석 시작 날짜. 사용자가 지정한 날짜를 서버로부터 전달받음
                to_date : 데이터 분석 마지막 날짜. 사용자가 지정한 날짜를 서버로부터 전달받음
                train_dir : 학습 데이터(csv파일) 위치
        Output : None
        Detail : Model Training and Serving. 이 모듈은 다른 모듈들과 다르게 Training과 Serving이 별개로 진행되는 게 아니라 동시에 진행된다.
        """
        self.logger.info("<======== [training] : start training ========>")

        start_time = timeit.default_timer()

        from_date_string = datetime.datetime.strptime(
            self.from_date, "%Y%m%d"
        ).strftime("%Y-%m-%d")
        to_date_string = datetime.datetime.strptime(self.to_date, "%Y%m%d").strftime(
            "%Y-%m-%d"
        )

        # csv에서 학습할 data를 가져온다
        filepath_list = self.get_csvfile_list(
            self.train_dir, self.target_id, from_date_string, to_date_string
        )
        input_df = self.load_csvfiles(filepath_list)
        body = {"results" : False, "master_id" : None}

        result = aicommon.Utils.validate_trainig_dataframe(input_df, self.number_of_clusters, self.logger)
        if type(result) == dict:
            return None, body, result["code"], result["message"]
        else:
            input_df = result

        # input_df time field sort
        input_df['time'] = pd.to_datetime(input_df['time'])
        input_df = input_df.sort_values('time')
        input_df['time'] = input_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        input_df = input_df.set_index(SQL_ID)

        self.logger.info("[training] prepare input data")
        self.logger.info(input_df.head(20))
        self.logger.info(
            f"[training] prepare clustering workload of inst_id={self.target_id}"
        )

        # training dataframe을 만든다
        self.logger.info("[training] prepare train data")

        train_df, queries = self.make_training_dataframe(input_df)
        train_data = train_df[
            [column for column in train_df.columns if column != SQL_UID]
        ].values

        self.logger.info(train_df.head(20))

        self.logger.info("[training] Data Scaling")

        scaled_data = aicommon.Utils.min_max_scaler(train_data)

        # Fitting with inputs
        kmeans_with_random_init = KMeans(n_clusters=int(self.number_of_clusters))
        kmeans_with_random_init = kmeans_with_random_init.fit(scaled_data)

        # get the cluster centers
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

        self.logger.info(
            "[training] fitting to scaled input Data with random initial point"
        )
        self.logger.info(
            "[training] debug labels of scaled data with random initial point"
        )
        self.logger.info(kmeans_with_random_init.predict(scaled_data))

        # 원점에서 클러스터 중심점까지의 Eucledian Distance 계산.
        distance_square = np.sum(np.square(cluster_centers_with_random_init), axis=1)

        # 내림차순으로 정렬. 즉, 원점에서 거리가 가장 먼 클러스터 순으로 정렬한다.
        sort_index = distance_square.argsort()[::-1]

        self.logger.debug(f"sort_index : {sort_index}")

        # 가장 거리가 먼 클러스터부터 Array의 첫 번째 로우에 위치시킨다.
        sorted_centers = np.take(cluster_centers_with_random_init, sort_index, axis=0)

        self.logger.info(
            "[training] sort clusters depending on eucledian distance from the origin"
        )
        self.logger.debug(sorted_centers)

        # Fitting with initial centers
        kmeans_with_sorted_init = KMeans(
            n_clusters=int(self.number_of_clusters), init=sorted_centers, n_init=1
        )
        kmeans_with_sorted_init = kmeans_with_sorted_init.fit(scaled_data)
        cluster_centers_with_sorted_init = kmeans_with_sorted_init.cluster_centers_

        self.logger.info("[training] truncate negative cluster centers to zero")

        centers = pd.DataFrame(
            cluster_centers_with_sorted_init,
            index=list(range(1, 1 + self.number_of_clusters)),
            columns=self.column_names[1:],
        )
        centers = centers.applymap(lambda x: 0 if x < 0 else x)
        centers.index.name = CLUSTER_NUMBER
        self.cluster_ids = list(centers.index)
        centers_for_serving = centers.copy()
        labels = kmeans_with_sorted_init.predict(scaled_data)

        self.logger.info(
            "[training] fitting to scaled input Data with sorted initial point"
        )
        self.logger.info(centers)
        self.logger.info(
            "[training] debug labels of scaled data with sorted initial point"
        )
        self.logger.info(f"{labels}")

        # 클러스터별로 Top-3 지표 추출
        for row_index in list(self.cluster_ids):
            row_series = centers.loc[row_index]
            sorted_row = row_series.sort_values(ascending=False)
            core_metric_values = sorted_row[: self.metric_filtering_threshold]
            core_metric_names = sorted_row[: self.metric_filtering_threshold].index
            if self.cluster_core_metrics.get(row_index) == None:
                self.cluster_core_metrics[row_index] = dict(
                    zip(core_metric_names, core_metric_values)
                )

        self.logger.info("[training] find core metrics of each cluster")

        input_df[LABEL] = labels + 1

        self.logger.info("[training] add cluster label of sql on input_df")

        serving_df = input_df.reset_index().copy()

        self.logger.info("[training] create serving_df from input_df")

        # queries는 {(:index, :sql_id, :sql_query) : {'avg' : :avg, 'stddev' : :stddev, 'index' : :index, 'label' : :label}}로 구성
        # 데이터가 속하는 군집 번호 저장
        for key, query in queries.items():
            label = labels[query[INDEX]]  # label: int type
            query[LABEL] = label

        for cluster_id in pd.unique(serving_df[LABEL]):
            # 특정 클러스터에 해당하는 데이터를 전체 데이터프레임에서 슬라이싱
            cluster_df = serving_df.query(f"{LABEL} == '{cluster_id}'")
            cluster_metrics = list(self.cluster_core_metrics[cluster_id].keys())

            self.logger.info(
                f"[training] start getting detail information of cluster no.{cluster_id}"
            )

            sorted_cluster_df = cluster_df.sort_values(
                by=cluster_metrics, ascending=False
            )

            self.logger.info(
                f"[training] sort values of cluster no.{cluster_id} by cluster metric {cluster_metrics}"
            )

            if sorted_cluster_df.shape[0] < self.sql_filtering_threshold:
                sliced_cluster_df = sorted_cluster_df.iloc[: sorted_cluster_df.shape[0]]
            else:
                sliced_cluster_df = sorted_cluster_df.iloc[
                    : self.sql_filtering_threshold
                ]

            # 슬라이싱된 특정 클러스터의 데이터로부터 클러스터 상세정보 저장
            for row_number in list(range(sliced_cluster_df.shape[0])):
                record = sliced_cluster_df.iloc[row_number]
                if self.detail.get(cluster_id):
                    self.detail[cluster_id][TIME].append(record[TIME])
                    self.detail[cluster_id][SQL_ID].append(record[SQL_ID])
                    self.detail[cluster_id][SQL_UID].append(record[SQL_UID])
                    self.detail[cluster_id][SQL_VALUES]["body"].append(
                        list(record[cluster_metrics].values)
                    )
                else:
                    self.detail[cluster_id] = {}
                    self.detail[cluster_id][CLUSTER_ID] = cluster_id  # cluster_id는 1 부터
                    # self.detail[label][TOTAL] = len(queries)
                    self.detail[cluster_id][TIME] = [record[TIME]]
                    self.detail[cluster_id][SQL_ID] = [record[SQL_ID]]
                    self.detail[cluster_id][SQL_UID] = [record[SQL_UID]]
                    self.detail[cluster_id][SQL_VALUES] = {
                        "header": cluster_metrics,
                        "body": [
                            list(record[cluster_metrics].values)
                        ]
                    }

            self.logger.info(
                f"[training] finish getting detail information of cluster no.{cluster_id}"
            )

        for key, item in self.detail.items():
            # item_df = pd.DataFrame(item)
            # item_df.drop_duplicates([SQL_ID_LIST, SQL_TEXT], inplace=True)
            # item[SQL_ID] = list(item_df[SQL_ID].values)
            # item[SQL_TEXT] = list(item_df[SQL_TEXT].values)
            self.detail[key][SIZE] = len(self.detail[key][SQL_ID])

        self.logger.info("[training] complete deleting duplicated data")

        # Save Fitted Model
        with open(os.path.join(self.model_dir, f"{self.model_id}.pkl"), "wb") as f:
            pickle.dump(from_date_string, f)
            pickle.dump(to_date_string, f)
            pickle.dump(kmeans_with_sorted_init, f)
            pickle.dump(centers, f)  # 클러스터 중심점
            pickle.dump(queries, f)  # 데이터가 속한 군집 번호

        centers.reset_index(inplace=True)

        # master_id = self.select_master_id()
        overall_result, serving_result, errcode, errmsg = self.serve(serving_df, queries, centers_for_serving)
        report = {
            "train_m": input_df.shape[0],
            "from_date": from_date_string,
            "to_date": to_date_string,
            "master_id" : self.master_id
        }

        if errcode != 0:
            return None, body, errcode, errmsg

        result = {"result": True, "report": report}
        end_time = timeit.default_timer()
        body = {"results" : True, "master_id" : self.master_id}

        self.logger.info(
            f"[training] Total Training WorkingTime : {end_time - start_time:.2f} sec"
        )
        self.logger.info("<======== [training] : end training========>")

        return result, body, 0, None

    def serve(self, serving_df, queries, centers_for_serving):
        """

        :param serving_df:
        :param queries:
        :param centers_for_serving:
        :return:
        """
        self.logger.info("<======== [serving] : start serving ========>")
        start_time = timeit.default_timer()

        from_date = datetime.datetime.strptime(self.from_date, "%Y%m%d").strftime(
            "%Y-%m-%d"
        )
        to_date = datetime.datetime.strptime(self.to_date, "%Y%m%d").strftime(
            "%Y-%m-%d"
        )
        serving_df.set_index([TIME, SQL_ID, SQL_UID], inplace=True)
        serving_df[INDEX] = list(range(len(serving_df)))

        # line_chart Serving
        # Line Chart를 그릴 세 개의 지표에 대해서 결과 계산 및 DB 삽입
        # 리팩토링 필요
        try:
            self.insert_train_info_into_master_table()

            for stat_name in self.stat_names:
                self.logger.info(
                    f"<==== [serving] start serving of stat_name '{stat_name}'====>"
                )
                self.logger.info(f"[serving] : target_id={self.target_id}")

                if stat_name not in serving_df.columns:
                    return None, None, -1, f"Invalid stat_name '{stat_name}'"

                target_df_list = []
                prev_time = "0000-00-00 00:00:00"

                # serving_df의 row를 하나씩 읽어들인다
                for index, record in serving_df.iterrows():
                    # index는 ['time', 'sql_id']으로 구성, 나머지 컬럼들이 record를 구성
                    time = index[0]
                    sql_id = index[1]
                    sql_query = index[2]

                    # 10분 단위로 Serving을 하므로, 시간대가 바뀌면 이전까지 계산된 정보를 저장하고 새롭게 로우를 만들어줘야 한다.
                    if prev_time != time:
                        # 이전까지 계산된 target_row를 target_df에 저장하는 곳
                        if prev_time != "0000-00-00 00:00:00":
                            target_row.insert(0, prev_time)
                            target_df_list.append(target_row)

                        # 새롭게 만들어진 로우의 각 컬럼 값은 전부 0으로 초기화
                        target_row = [0 for _ in range(len(self.cluster_names))]

                    # sql_id별로 정보가 저장되어 있는 queries dictionary에서 해당 sql_id가 해당하는 군집 번호를 추출
                    # queries는 {(:index, :sql_id, :sql_query) : {'avg' : :avg, 'stddev' : :stddev, 'index' : :index, 'label' : :label}}로 구성
                    query = queries[(record[INDEX], sql_id, sql_query)]
                    label = query[LABEL]

                    # sql_id가 해당하는 군집 번호에 해당하는 컬럼(cluster_01, ..., cluster_06)에 해당 sql에서 현재 계산되는 stat_name의 stat_value를 더한다.
                    target_row[label] = target_row[label] + record[stat_name]

                    prev_time = time

                if prev_time != "0000-00-00 00:00:00":
                    target_row.insert(0, prev_time)
                    target_df_list.append(target_row)

                target_df = pd.DataFrame(
                    target_df_list, columns=[TIME] + self.cluster_names
                )

                self.logger.debug("target dataframe")
                self.logger.debug(f"{target_df}")

                serving_results_by_stat = target_df.to_dict(orient="records")

                self.logger.debug("serving_results_by_stat")
                self.logger.debug(f"{serving_results_by_stat}")

                report = {
                    "from_date": from_date,
                    "to_date": to_date,
                    "stat_name": stat_name
                }

                result = {"result": True, "report": report}

                self.logger.info(
                    f"[serving] : serving result({result['result']}) - ={self.target_id}"
                )
                self.logger.info(f"[serving] report : {report}")

                self.insert_serving_result_for_line(self.master_id, stat_name, target_df)

            # Radar_chart Serving
            column_names_1 = self.column_names[1:]
            centers_values = centers_for_serving[column_names_1]

            self.insert_serving_result_for_radar(self.master_id, centers_values)

            end_time = timeit.default_timer()
            self.logger.info(f"[serving] WorkingTime: {end_time - start_time:.2f} sec")
            self.logger.info("<======== [serving] : end serving ========>")

            return result, serving_results_by_stat, 0, None
        except Exception as e:
            return None, None, -1, "Unknown exception occured while db workload clustering serving"

    # def select_master_id(self):
    #     self.logger.info(
    #         "<======== [select_master_id] : start select_radar_id() ========>"
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

    def insert_serving_result_for_line(self, master_id, stat_name, target_df):
        """

        :param master_id:
\       :param serving_results_by_stat:
        :return:
        """
        self.logger.info(
            f"<========= Start Inserting Serving Result For Line Chart of stat name '{stat_name}'=========>"
        )
        start_time = timeit.default_timer()
        original_columns_index = target_df.columns
        new_columns_index = [MASTER_ID, STAT_NAME]
        target_df[new_columns_index[0]] = master_id
        target_df[new_columns_index[1]] = stat_name

        target_df = target_df.reindex(
            columns=new_columns_index + list(original_columns_index)
        )
        parameter_array = list(map(lambda x: tuple(x), target_df.values.tolist()))

        self.logger.debug("target_dataframe")
        self.logger.debug(f"{target_df.head(10)}")

        conn = None
        cursor = None

        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            self.logger.info("[insert_serving_result_for_line] : Connection Get!")

            data_insert_query = (
                "INSERT INTO aiops_module_wclst_line "
                "(master_id "
                ', stat_name, "time" '
                ", cluster_value01, cluster_value02, cluster_value03, cluster_value04, cluster_value05, cluster_value06) "
                "VALUES %s "
            )
            pg2.extras.execute_values(cursor, data_insert_query, parameter_array)

            conn.commit()

            self.logger.info(
                "[insert_serving_result_for_line] : Complete Values Insertion!"
            )
        except Exception as exception:
            self.logger.error(
                f"Unexepected exception during inserting serving result into aiops_module_wclst_line : {exception}"
            )
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

            self.logger.info("[insert_serving_result_for_line] : Connection close")

            end_time = timeit.default_timer()

            self.logger.info(
                f"Line Chart Serving WorkingTime : {end_time - start_time:.2f} sec"
            )
            self.logger.info(
                "<======== End Inserting Serving Result For Line Chart ========>"
            )

    def insert_serving_result_for_radar(self, master_id, centers_values):
        """
        Input : centers_values
                centers_values : 클러스터별 중심값. numpy.ndarray
        Output: None
        Detail: PostgreSQL DB Connection and insert result into the database
        """
        self.logger.info(
            "<======= Start Inserting Serving Result For Radar Chart =======>"
        )
        start_time = timeit.default_timer()

        conn = None
        cursor = None

        # TODO 클러스터 상세 정보 넣기
        try:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.DictCursor)
            cluster_id = 1

            self.logger.info("[insert_serving_result_for_radar] : Connection Get!")

            data_insert_query = (
                "INSERT INTO aiops_module_wclst_radar "
                "(master_id, "
                "cluster_id, stat_value01, stat_value02, stat_value03, stat_value04, stat_value05, stat_value06, stat_value07, stat_value08, detail) "
                "VALUES (%s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            )

            for centers_value in centers_values.values:
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
            self.logger.error(f"Unexpected exception during serving.. {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

            self.logger.info("[insert_serving_result_for_radar] Connection Close!")

            end_time = timeit.default_timer()

            self.logger.info(
                f"Radar Chart Serving WorkingTime : {end_time - start_time:.2f} sec"
            )
            self.logger.info(
                "<======= End Inserting Serving Result For Radar Chart =======>"
            )

    def end_train(self):
        pass
