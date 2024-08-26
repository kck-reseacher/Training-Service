import json
import os
import traceback
import pandas as pd
import psycopg2 as pg2
from pathlib import Path
from datetime import datetime
from common.system_util import SystemUtil
from common.base64_util import Base64Util
from common import constants
from common import aicommon
from common.constants import SystemConstants as sc
from common.clickhouse_client import execute_query
from resources.config_manager import Config


class TSAUtils:
    def __init__(self):
        self.os_env = SystemUtil.get_environment_variable()
        self.db_conn_str = self.get_db_conn_str()

    @staticmethod
    def get_server_run_configuration():
        os_env = SystemUtil.get_environment_variable()
        py_config = Config(os_env[sc.MLOPS_TRAINING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()

        return os_env[sc.MLOPS_TRAINING_PATH], py_config, os_env[sc.MLOPS_LOG_PATH]

    def get_db_conn_str(self):
        py_path, py_config, log_path = TSAUtils.get_server_run_configuration()

        try:
            pg_decode_config = Base64Util.get_config_decode_value(py_config[sc.POSTGRES])
        except Exception:
            tb = traceback.format_exc()
            print(f"get_db_conn_str Exception: {tb}")
            print('base64 decode error, config: ' + str(py_config[sc.POSTGRES]))
            pg_decode_config = py_config[sc.POSTGRES]

        db_conn_str = (
            f"host={pg_decode_config['host']} "
            f"port={pg_decode_config['port']} "
            f"dbname={pg_decode_config['database']} "
            f"user={pg_decode_config['id']} "
            f"password={pg_decode_config['password']}"
        )

        return db_conn_str

    def get_train_meta(self, train_history_id: int):
        """
        PG에서 train meta 데이터를 조회
        Parameters
        ----------
        train_history_id: 학습 관리 아이디

        Returns
        -------
        json_data: train meta
        status: 학습 상태
        regenerate_train_data: 학습 데이터 재생성 여부, ex) True or False
        """
        conn, cursor = None, None
        conn = pg2.connect(self.db_conn_str)
        cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

        query = f"""
                    SELECT 
                        train_meta, status, regenerate_train_data
                    FROM 
                        ai_history_train
                    WHERE 
                        history_train_id = {train_history_id};
                    """
        cursor.execute(query)
        query_result = cursor.fetchone()
        json_data = json.loads(query_result['train_meta'])
        status = query_result['status']
        regenerate_train_data = query_result['regenerate_train_data']

        if cursor:
            cursor.close()
        if conn:
            conn.close()

        return json_data, status, regenerate_train_data

    def get_product_type(self, target_id):
        """
        PG에서 DB product type 정보를 조회
        Parameters
        ----------
        sys_id: XAIOps system id
        target_id: XAIOps target id

        Returns
        -------
        DB product type ex) 'ORACLE'
        """
        conn, cursor = None, None
        conn = pg2.connect(self.db_conn_str)
        cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

        query = f"""
                    SELECT 
                        inst_product_type
                    FROM 
                        xaiops_config_instance
                    WHERE 
                        target_id = '{target_id}';
                    """
        cursor.execute(query)
        query_result = cursor.fetchone()
        product_type = query_result['inst_product_type']

        if cursor:
            cursor.close()
        if conn:
            conn.close()

        return product_type

    def get_dbsln_url(self) -> str:
        """
        PG에서 dbsln 학습 완료 후 reload API host url을 조회
        Returns
        -------
        host_url : ex) http://ip:port
        """
        conn, cursor = None, None
        conn = pg2.connect(self.db_conn_str)
        cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

        query = f"""
                SELECT
                    host_url
                FROM
                    xaiops_meta_connection_backend_server
                WHERE
                    server_type = 'dbsln';
                """
        cursor.execute(query)
        query_result = cursor.fetchone()
        host_url = query_result['host_url']

        if cursor:
            cursor.close()
        if conn:
            conn.close()

        return host_url

    def update_train_history(self, train_history_id: int, train_result: dict):
        """
        ai_history_train 테이블에 학습 결과를 업데이트
        ex) 학습 시작, 종료 시간, 학습 성공한 타겟 리스트 등
        Parameters
        ----------
        train_history_id: 학습 관리 아이디
        train_result: 학습 결과 meta

        Returns
        -------

        """
        if self.os_env[sc.AIMODULE_SERVER_ENV] == 'local':
            pass
        else:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

            if train_result['module'] in [constants.EXEM_AIOPS_ANLS_INST, constants.EXEM_AIOPS_ANLS_LOG, constants.EXEM_AIOPS_FCST_TSMIXER]:
                success_target_list = [train_result['target_id']]
            elif train_result['module'] == constants.EXEM_AIOPS_ANLS_SERVICE:
                success_target_list = train_result['results']['dbsln']['train_target_list']
            elif train_result['module'] == constants.EXEM_AIOPS_LOAD_FCST:
                success_target_list = train_result['results']['seq2seq']['target_list']
            elif train_result['module'] == constants.EXEM_AIOPS_EVENT_FCST:
                success_target_list = train_result['results']['target_list']

            query = f"""
                        UPDATE 
                            ai_history_train
                        SET
                            results = '{json.dumps(train_result, cls=aicommon.JsonEncoder, ensure_ascii=False)}',
                            success_target_list = '{json.dumps(success_target_list, cls=aicommon.JsonEncoder)}',
                            module_start_time = to_timestamp('{train_result['train_start_time'].strftime("%Y-%m-%d %H:%M:%S.%f")}', 'YYYY-MM-DD HH24:MI:SS.US'),
                            module_end_time = to_timestamp('{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}', 'YYYY-MM-DD HH24:MI:SS.US')
                        WHERE 
                            history_train_id = {train_history_id};
                        """
            cursor.execute(query)
            conn.commit()

            if cursor:
                cursor.close()
            if conn:
                conn.close()

            return True

    def update_train_start_time(self, train_history_id: int):
        """
        학습/서비스 > 학습 > 학습 이력 화면에 학습 요청 시간을 표출하기 위해 테이블을 업데이트
        Parameters
        ----------
        train_history_id: 학습 관리 아이디

        Returns
        -------
        """
        if self.os_env[sc.AIMODULE_SERVER_ENV] == 'local':
            pass
        else:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

            select_query = f"""
                        SELECT 
                            train_start_time
                        FROM
                            ai_history_train
                        WHERE 
                            history_train_id = {train_history_id};
                        """

            cursor.execute(select_query)
            existing_start_time = cursor.fetchone()

            if existing_start_time['train_start_time'] is None:
                update_query = f"""
                            UPDATE 
                                ai_history_train
                            SET
                                train_start_time = to_timestamp('{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}', 'YYYY-MM-DD HH24:MI:SS.US')
                            WHERE 
                                history_train_id = {train_history_id};
                            """
                cursor.execute(update_query)
                conn.commit()

            if cursor:
                cursor.close()
            if conn:
                conn.close()

            return True

    def update_config_service(self, train_result: dict):
        """

        Parameters
        ----------
        train_result

        Returns
        -------
        objective : 설정 > 서빙 프로세스 화면에서 학습된 타겟 목록이나 학습된 지표 등을 표출하기 위해 사용

        """
        if self.os_env[sc.AIMODULE_SERVER_ENV] == 'local':
            pass
        else:
            service_name = f"{train_result['module']}_{train_result['inst_type']}_{train_result['target_id']}"
            business_types = None
            target_list = None
            train_hyper_params = None

            if train_result["module"] == constants.EXEM_AIOPS_ANLS_INST:
                metric_ids_json = {'gdn': list(train_result['results']['gdn']['train_metrics'].keys())}
                target_list = [train_result['target_id']]

            elif train_result["module"] == constants.EXEM_AIOPS_ANLS_SERVICE:
                metric_ids_json = {'dbsln': list(train_result['results']['dbsln']['train_metrics'].keys())}
                business_types = train_result['results']['dbsln']['business_list']
                target_list = train_result['results']['dbsln']['train_target_list']

            elif train_result["module"] == constants.EXEM_AIOPS_LOAD_FCST:
                metric_ids_json = {'seq2seq': list(train_result['results']['seq2seq']['train_metrics'].keys())}
                target_list = train_result['results']['seq2seq']['target_list']

            elif train_result["module"] == constants.EXEM_AIOPS_ANLS_LOG:
                metric_ids_json = {'digcn': list(train_result['results']['digcn']['train_metrics'].keys())}
                target_list = [train_result['target_id']]

            elif train_result["module"] == constants.EXEM_AIOPS_EVENT_FCST:
                metric_ids_json = train_result['results']['target_metric']
                target_list = train_result['results']['target_list']

            elif train_result["module"] == constants.EXEM_AIOPS_FCST_TSMIXER:
                metric_ids_json = {'tsmixer': list(train_result['results']['tsmixer']['train_metrics'].keys())}
                target_list = [train_result['target_id']]

            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

            if business_types is not None:
                business_types = json.dumps(business_types, cls=aicommon.JsonEncoder, ensure_ascii=False)

            if target_list is not None:
                target_list = json.dumps(target_list, cls=aicommon.JsonEncoder)

            if train_hyper_params is not None:
                train_hyper_params = json.dumps(train_hyper_params, cls=aicommon.JsonEncoder)

            select_query = f"""
                                    SELECT 
                                        serving_name
                                    FROM
                                        ai_config_serving
                                    WHERE 
                                        serving_name = '{service_name}';
                                    """

            cursor.execute(select_query)
            existing_service_name = cursor.fetchone()

            insert_query = f"""
                                INSERT
                                INTO
                                    ai_config_serving
                                (
                                    serving_name,
                                    inst_type,
                                    target_id,
                                    module,
                                    service_on_boot,
                                    update_pkl_dt,
                                    business_types,
                                    metric_ids_json,
                                    train_hyper_params,
                                    target_list
                                )
                                VALUES 
                                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """

            update_query = f"""
                                UPDATE 
                                    ai_config_serving
                                SET
                                    metric_ids_json = %s,
                                    business_types = %s,
                                    train_hyper_params = %s,
                                    target_list = %s
                                WHERE 
                                    serving_name = %s;
                                """

            if existing_service_name is None:
                data_to_insert = (
                    service_name,
                    train_result['inst_type'],
                    train_result['target_id'],
                    train_result['module'],
                    False,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    business_types,
                    json.dumps(metric_ids_json, cls=aicommon.JsonEncoder),
                    train_hyper_params,
                    target_list
                )
                cursor.execute(insert_query, data_to_insert)
                conn.commit()
            else:
                data_to_update = (
                    json.dumps(metric_ids_json, cls=aicommon.JsonEncoder),
                    business_types,
                    train_hyper_params,
                    target_list,
                    service_name
                )
                cursor.execute(update_query, data_to_update)
                conn.commit()

            if cursor:
                cursor.close()
            if conn:
                conn.close()

            return True

    def insert_log_parameter(self, train_result: dict):
        """
        로그 이상탐지 학습 후 파라미터 값을 업데이트 ex) anomaly_threshold
        Parameters
        ----------
        train_result: 학습 결과 meta

        Returns
        -------
        로그 이상탐지 기능만 사용
        설정 > 서빙 프로세스 화면에서 학습 결과 저장 용도
            로그 이상탐지의 경우 임계치 설정 값 저장
        """

        if self.os_env[sc.AIMODULE_SERVER_ENV] == 'local':
            pass
        else:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor()

            log_digcn_delete_query = f"""
                            DELETE
                            FROM
                                ai_config_serving_log_digcn
                            WHERE
                                module = %s
                                AND inst_type = %s 
                                AND target_id = %s
                                AND hyper_type = %s
                                AND metric_id = %s
                            """

            log_digcn_insert_query = f"""
                            INSERT
                            INTO
                                ai_config_serving_log_digcn
                                (
                                    module,
                                    inst_type,
                                    target_id,
                                    hyper_type,
                                    metric_id,
                                    params,
                                    recommend_param,
                                    created_dt
                                )
                            VALUES
                            (
                                %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            """

            log_sparse_insert_query = f"""
                                    INSERT
                                    INTO
                                        ai_config_serving_log_sparse
                                        (
                                        log_id,
                                        alert_threshold,
                                        rare_rate,
                                        is_service_on
                                        )
                                    VALUES
                                    (
                                        %s, %s, %s, %s
                                    )
                                    """

            for algo in train_result['results'].keys():
                if algo != 'log_type':
                    for metric in train_result['results'][algo]['train_metrics']:
                        data_to_delete = (
                            train_result['module'],
                            train_result['inst_type'],
                            train_result['target_id'],
                            'range',
                            metric
                        )

                        anomaly_threshold = train_result['results'][algo]['anomaly_threshold']
                        data_to_insert = (
                            train_result['module'],
                            train_result['inst_type'],
                            train_result['target_id'],
                            'range',
                            metric,
                            json.dumps({"anomaly_threshold": 1}),
                            json.dumps({"anomaly_threshold": anomaly_threshold}),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        )
                        cursor.execute(log_digcn_delete_query, data_to_delete)
                        cursor.execute(log_digcn_insert_query, data_to_insert)

            select_log_id_query = f"""
                                SELECT 
                                    log_id
                                FROM
                                    xaiops_config_log
                                WHERE 
                                    target_id = %s
                                """
            cursor.execute(select_log_id_query, (train_result['target_id'],))
            log_id = cursor.fetchone()[0]

            check_log_id_query = f"""
                                SELECT EXISTS (
                                    SELECT 1
                                    FROM
                                        ai_config_serving_log_sparse
                                    WHERE
                                        log_id = %s
                                )
                                """
            cursor.execute(check_log_id_query, (log_id,))
            sparse_config_exists = cursor.fetchone()[0]

            if not sparse_config_exists:
                sparse_default_data = (
                    log_id,
                    1,
                    0.0001,
                    False
                )
                cursor.execute(log_sparse_insert_query, sparse_default_data)

            conn.commit()

            if cursor:
                cursor.close()
            if conn:
                conn.close()

            return True

    def update_train_status(self, train_history_id: int, train_status: str, error_message=None):
        """
        ai_history_train 테이블에 아래 학습 상태를 업데이트
        '3':학습데이터 생성(컨슈밍완료 후), '4':학습 진행중(모듈 호출 후), '5':학습 완료(모델 리로드 전), '9':학습 실패(모듈오류)
        Parameters
        ----------
        train_history_id: 학습 히스토리 아이디 (TID)
        train_status: 학습 상태
        error_message: 학습 에러 메시지 string

        Returns
        -------
        동작 Flow
        1) train_status 값이 '5':학습 완료(모델 리로드 전) or '9':학습 실패(모듈오류)일 경우에 train_end_time을 함께 업데이트
        1-1) train_status 값이 '9':학습 실패(모듈오류)일 경우 error_message 를 함께 업데이트
        2) train_status 값이 '3':학습데이터 생성(컨슈밍완료 후), '4':학습 진행중(모듈 호출 후) 인 경우 status 만 업데이트
        """
        if self.os_env[sc.AIMODULE_SERVER_ENV] == 'local':
            pass
        else:
            conn = pg2.connect(self.db_conn_str)
            cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)

            if error_message is None:
                error_message = None

            if train_status in ['5', '9']:
                train_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                train_end_time = None

            train_end_time_param = 'NULL' if train_end_time is None else f"to_timestamp('{train_end_time}', 'YYYY-MM-DD HH24:MI:SS.US')"
            query = f"""
                    UPDATE
                        ai_history_train
                    SET
                        status = %s,
                        train_end_time = {train_end_time_param},
                        warning_comment = %s
                    WHERE
                        history_train_id = %s;
                    """

            cursor.execute(query, (train_status, error_message, train_history_id))
            conn.commit()

            if cursor:
                cursor.close()
            if conn:
                conn.close()

            return True

    def select_traindata_from_dm(self, inst_type: str, target_id: str, from_date: str, to_date: str, product_type=None):
        """
        dm(clickhouse db)에서 학습 데이터를 조회하는 함수
        Parameters
        ----------
        inst_type: XAIOps 인스턴스 타입, ex) 'was', 'db'
        target_id: XAIOps 타겟 아이디, ex) '1201', '1301'
        from_date: 학습 데이터 기간, ex) '2024-06-17'
        to_date: 학습 데이터 기간, ex) '2024-06-17'
        product_type: DB product type, ex) 'ORACLE'

        Returns
        -------
        df: 학습 데이터(pandas DataFrame)
        """
        if product_type:
            inst_type = f"{inst_type}_{product_type.lower()}"
        query = f"""
                SELECT 
                    *
                FROM 
                    dm_{inst_type}
                WHERE 
                    target_id = '{target_id}'
                    AND time BETWEEN '{from_date}' AND '{to_date} 23:59:00'
                """
        df = execute_query(query)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = df['time'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.sort_values(by='time')

        return df

    def get_df_list(self, inst_type: str, target_info: list, from_date: str, to_date: str) -> list:
        """
        DM(clickhouse)에서 target_id 수 만큼 학습 데이터를 여러번 조회
        Parameters
        ----------
        inst_type: XAIOps 인스턴스 타입, ex) 'was', 'db'
        target_info: XAIOps 타겟 아이디 리스트, ex) ['1201', '1301']
        from_date: 학습 데이터 시작 날짜, ex) 2024-06-26
        to_date: 학습 데이터 종료 날짜, ex) 2024-06-26

        Returns
        -------
        df_list: 학습 데이터(pandas DataFrame)가 있는 리스트
        """
        df_list = []
        for target_id in target_info:
            target_df = self.select_traindata_from_dm(inst_type, target_id, from_date, to_date)
            if not target_df.empty:
                df_list.append(target_df)
        return df_list

    def get_df_from_service(self, config: dict, inst_type: str, target_id: str, from_date: str, to_date: str):
        """
        service 타입 전용으로 DM(clickhouse)에서 학습 데이터를 조회하는 함수
        Parameters
        ----------
        config: XAIOps train meta
        inst_type: XAIOps 인스턴스 타입, ex) 'was', 'db'
        target_id: XAIOps 타겟 아이디, ex) '1201', '1301'
        from_date: 학습 데이터 시작 날짜, ex) 2024-06-26
        to_date: 학습 데이터 종료 날짜, ex) 2024-06-26

        Returns
        -------
        부하예측(tsmixer)의 경우 : 학습 데이터(pandas DataFrame), 서비스 이상탐지, 부하예측(RMC)의 경우 학습 데이터 리스트
        """

        if config['module'] in [constants.EXEM_AIOPS_ANLS_SERVICE]:
            df_list = self.get_df_list(inst_type, config['service_list'], from_date, to_date)
            return pd.concat(df_list) if df_list else pd.DataFrame()
        elif config['module'] in [constants.EXEM_AIOPS_FCST_TSMIXER, constants.EXEM_AIOPS_LOAD_FCST]:
            return self.select_traindata_from_dm(inst_type, target_id, from_date, to_date)

    def get_train_data(self, config: dict, inst_type: str, target_id: str) -> pd.DataFrame:
        """
        DM(clickhouse)에서 학습 데이터를 조회하는 함수
        Parameters
        ----------
        config: XAIOps train meta
        inst_type: XAIOps 인스턴스 타입, ex) 'was', 'db'
        target_id: XAIOps 타겟 아이디, ex) '1201', '1301'

        Returns
        -------
        df: 학습 데이터(pandas DataFrame)
        """
        from_date = datetime.strptime(config['date'][0], '%Y%m%d').strftime('%Y-%m-%d')
        to_date = datetime.strptime(config['date'][-1], '%Y%m%d').strftime('%Y-%m-%d')

        if inst_type == 'service':
            return self.get_df_from_service(config, inst_type, target_id, from_date, to_date)
        else:
            product_type = self.get_product_type(target_id) if inst_type == 'db' else None
            return self.select_traindata_from_dm(inst_type, target_id, from_date, to_date, product_type)

    def select_datelist_traindata_from_dm(self, inst_type: str, target_id: str, datelist: list, product_type=None) -> pd.DataFrame:
        """
        dm(clickhouse db)에서 비즈니스 데이 학습 데이터를 조회하는 함수
        Parameters
        ----------
        inst_type: XAIOps 인스턴스 타입, ex) 'was', 'db'
        target_id: XAIOps 타겟 아이디, ex) '1201', '1301'
        datelist: 학습 데이터 리스트, ex) ['2024-06-17', '2024-06-18', ...]
        product_type: DB product type, ex) 'ORACLE'

        Returns
        -------
        df: 학습 데이터(pandas DataFrame)
        """
        if product_type:
            inst_type = f"{inst_type}_{product_type.lower()}"
        query_conditions = []
        for date in datelist:
            query_conditions.append(f"(time BETWEEN '{date}' AND '{date} 23:59:00')")
        sql_condition = " OR ".join(query_conditions)

        query = f"""
                SELECT
                    *
                FROM
                    dm_{inst_type}
                WHERE
                    target_id = '{target_id}'
                    AND ({sql_condition})
                """
        df = execute_query(query)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = df['time'].dt.tz_convert('Asia/Seoul').dt.strftime('%Y-%m-%d %H:%M:%S')
            df = df.sort_values(by='time')

        return df

    def get_train_datelist_data(self, config: dict, datelist: list, inst_type: str, target_id: str) -> pd.DataFrame:
        """
        DM(clickhouse)에서 학습 데이터를 조회하는 함수
        Parameters
        ----------
        config: XAIOps train meta
        datelist: 학습 데이터 리스트, ex) ['2024-06-17', '2024-06-18', ...]
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        학습 데이터 ex) pandas DataFrame
        """
        df = pd.DataFrame()
        if inst_type == 'service':
            df_list = []
            target_info = config['service_list']
            for target_id in target_info:
                target_df = self.select_datelist_traindata_from_dm(
                    inst_type,
                    target_id,
                    datelist)
                df_list.append(target_df)
            df_list = [df for df in df_list if not df.empty]
            if len(df_list) > 0:
                df = pd.concat(df_list)
        else:
            product_type = self.get_product_type(target_id) if inst_type == 'db' else None
            df = self.select_datelist_traindata_from_dm(
                inst_type,
                target_id,
                datelist,
                product_type)

        return df

    def dataframe_to_parquet(self, train_dir: str, df: pd.DataFrame, target_id=None):
        """
        학습 데이터(dataframe)를 parquet 포맷의 파일로 저장
        Parameters
        ----------
        train_dir: 학습 데이터 경로
        df: 학습 데이터 ex) pandas DataFrame
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        """
        Path(train_dir).mkdir(exist_ok=True, parents=True)
        if df.empty:
            pass
        else:
            df.to_parquet(f"{train_dir}/{target_id}.parquet")

    def check_missing_date(self, config: dict, df: pd.DataFrame, inst_type: str, target_id: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        config: XAIOps train meta
        df: 학습 데이터, ex) pandas dataframe
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        boolean flag ex) True or False

        check_missing_date 동작 Flow
        1) 학습 요청 데이터(train meta)의 기간(date)이 학습 데이터(dataframe)에 없는 경우 DM(CH)에서 데이터 조회 (return True)
        2) 기간이 모두 있는 경우 그대로 사용 (return False)
        """
        missing_dates = []
        date_range = pd.date_range(start=config["date"][0], end=config["date"][-1])
        df.loc[:, 'datetime'] = pd.to_datetime(df['time'])
        existing_dates = df['datetime'].dt.date.unique()
        df = df.drop(columns=['datetime'])
        for date in date_range:
            if date not in existing_dates:
                missing_dates.append(date.strftime('%Y-%m-%d'))

        if missing_dates:
            return True
        else:
            return False

    def check_missing_date_biz(self, config: dict, df: pd.DataFrame, biz_index: str, inst_type: str, target_id: str) -> pd.DataFrame:
        """
        비즈니스 데이터 전용 missing data 체크
        Parameters
        ----------
        config: XAIOps train meta
        df: 학습 데이터, ex) pandas dataframe
        biz_index: XAIOps business index, ex) '8', '10'
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        학습 데이터, ex) pandas DataFrame
        """
        missing_dates = []
        df.loc[:, 'datetime'] = pd.to_datetime(df['time'])
        existing_dates = df['datetime'].dt.date.unique()
        df = df.drop(columns=['datetime'])
        for business_dict in config['business_list']:
            if business_dict['index'] == biz_index:
                for date in business_dict['date']:
                    if datetime.strptime(date, "%Y%m%d").date() > existing_dates[-1]:
                        missing_dates.append(datetime.strptime(date, "%Y%m%d").date().strftime('%Y-%m-%d'))

        if missing_dates:
            return True
        else:
            return False

    def read_parquet(self, config: dict, inst_type: str, target_id: str, features: list, logger: object) -> pd.DataFrame:
        """
        Parameters
        ----------
        config: XAIOps train meta
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'
        features: 지표 리스트, ex) ['tps', 'cpu_usage']
        logger: python logger

        Returns
        -------
        학습 데이터, ex) pandas DataFrame

        read_parquet 동작 Flow
        1) 재생성 옵션이 False이면서 로컬 경로에 학습 데이터 파일(parquet)이 존재하는 경우 파일을 pandas dataframe 형태로 read
        1-1) parquet 파일을 read하면서 에러(exception)가 발생한 경우 재생성 옵션을 True로 변경
        2) 학습 데이터 재생성 (regenerate_train_data)
        2-1) DM(CH)으로부터 학습 데이터를 조회하여 pandas dataframe 형태로 사용
        2-2) 학습 데이터(dataframe)를 parquet 포맷의 파일로 최신화
        3) missing data 체크
        3-1) 학습 요청 데이터(train meta)의 기간(date)이 학습 데이터(dataframe)에 없는 경우 추가로 DM(CH)으로 부터 데이터 조회
        3-2) 학습 요청 데이터(train meta)의 기간(date)이 학습 데이터(dataframe)에 모두 존재하는 경우 그대로 사용
        3-3) 학습 데이터(dataframe)를 parquet 포맷의 파일로 최신화
        """
        train_dir = f"{self.os_env[sc.AIMODULE_HOME]}/train_data/{config['sys_id']}/{inst_type}/{target_id}"
        train_data_path = f"{train_dir}/{target_id}.parquet"
        regenerate_train_data = False
        if os.path.exists(train_data_path) and config["regenerate_train_data"] == False:
            try:
                df = pd.read_parquet(
                    train_data_path,
                    filters=[
                        ('time', '>=', datetime.strptime(config["date"][0], '%Y%m%d').strftime('%Y-%m-%d')),
                        ('time', '<=', datetime.strptime(config["date"][-1], '%Y%m%d').strftime('%Y-%m-%d 23:59:00'))
                    ],
                    columns=["time", "target_id"] + features
                )
                missing_date_flag = self.check_missing_date(config, df, inst_type, target_id)
                if missing_date_flag:
                    regenerate_train_data = True
            except Exception:
                regenerate_train_data = True
                tb = traceback.format_exc()
                logger.info(f"read_parquet Exception: {tb}")
        else:
            regenerate_train_data = True

        if regenerate_train_data:
            df = self.get_train_data(config, inst_type, target_id)
            # df의 컬럼이 meta feature에 모두 있어야만 features 인덱싱
            # df 컬럼과 meta feature가 일치하지 않으면 features 인덱싱하지 않고 빈 DataFrame 리턴
            if all(col in df.columns for col in features):
                self.dataframe_to_parquet(train_dir, df, target_id)
                df = df[["time", "target_id"] + features]
            else:
                # 빈 DataFrame 생성 및 열 설정
                logger.info(f"no train data feature {set(features) - set(df.columns)}")
                df = pd.DataFrame(columns=["time", "target_id"])

        return df

    def read_business_parquet(self, config: dict, inst_type: str, target_id: str) -> dict:
        """
        business 학습 데이터의 경우 dbsln 이상탐지(system) 기능에서만 사용
        Parameters
        ----------
        config: XAIOps train meta
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        학습 데이터, ex) {'business_index': pandas DataFrame}
        """
        business_data_dict = {}
        for business_dict in config['business_list']:
            biz_index = business_dict['index']
            train_dir = f"{self.os_env[sc.AIMODULE_HOME]}/train_data/{config['sys_id']}/{inst_type}/{target_id}/business_{biz_index}"
            train_data_path = f"{train_dir}/{target_id}.parquet"
            regenerate_train_data = False
            if os.path.exists(train_data_path) and config["regenerate_train_data"] == False:
                try:
                    df = pd.read_parquet(
                        train_data_path,
                        filters=[
                            ('time', '>=', datetime.strptime(business_dict["date"][0], '%Y%m%d').strftime('%Y-%m-%d')),
                            ('time', '<=',
                             datetime.strptime(business_dict["date"][-1], '%Y%m%d').strftime('%Y-%m-%d 23:59:00'))
                        ]
                    )
                    missing_date_flag = self.check_missing_date_biz(config, df, biz_index, inst_type, target_id)
                    if missing_date_flag:
                        regenerate_train_data = True
                except Exception:
                    regenerate_train_data = True
                    tb = traceback.format_exc()
                    print(f"read_business_parquet Exception: {tb}")
            else:
                regenerate_train_data = True

            if regenerate_train_data:
                datelist = []
                for date in business_dict["date"]:
                    datelist.append(datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'))
                df = self.get_train_datelist_data(config, datelist, inst_type, target_id)
                self.dataframe_to_parquet(train_dir, df, target_id)

            if not df.empty:
                business_data_dict[biz_index] = df

        return business_data_dict

    def business_data_loading(self, config: dict, inst_type=None, target_id=None) -> dict:
        """
        business 학습 데이터를 로딩하는 함수
        business 학습 데이터의 경우 dbsln 이상탐지(system) 기능에서만 사용
        Parameters
        ----------
        config: XAIOps train meta
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        학습 데이터, ex) {'business_index': pandas DataFrame}
        """
        if inst_type is None:
            inst_type = config['inst_type']

        if target_id is None:
            target_id = config['target_id']

        business_data_dict = self.read_business_parquet(config, inst_type, target_id)

        return business_data_dict

    def data_loading(self, config: dict, logger: object, inst_type=None, target_id=None) -> pd.DataFrame:
        """
        Parameters
        ----------
        config: XAIOps train meta
        logger: python logger
        inst_type: XAIOps instance type, ex) 'was', 'db'
        target_id: XAIOps target id, ex) '1201'

        Returns
        -------
        학습 데이터, ex) pandas DataFrame

        data_loading 동작 Flow
        1) pg 테이블의 학습 상태를 3으로 업데이트 (3: 학습데이터 생성)
        2) 기능 별로 parquet 포맷의 학습 데이터를 read
        3) pg 테이블의 학습 상태를 4로 업데이트 (4: 학습 진행중)
        """
        # 학습 상태 업데이트 '3':학습데이터 생성
        self.update_train_status(config['train_history_id'], '3')
        if inst_type is None:
            inst_type = config['inst_type']

        if target_id is None:
            target_id = config['target_id']

        df = pd.DataFrame()
        if config['module'] in [constants.EXEM_AIOPS_ANLS_INST, constants.EXEM_AIOPS_ANLS_SERVICE,
                                constants.EXEM_AIOPS_FCST_TSMIXER]:
            if config['module'] == constants.EXEM_AIOPS_FCST_TSMIXER:
                features = config['train']['tsmixer']['features']
            elif config['module'] == constants.EXEM_AIOPS_ANLS_INST:
                features = config['train']['gdn']['features']
            else:
                features = config['train']['dbsln']['features']
            df = self.read_parquet(config, inst_type, target_id, features, logger)

        if config['module'] == constants.EXEM_AIOPS_LOAD_FCST:
            features = config['train']['seq2seq']['features']
            df_list = []
            for target_id in config["target_list"]:
                target_df = self.read_parquet(config, inst_type, target_id, features, logger)
                df_list.append(target_df)
            df_list = [df for df in df_list if not df.empty]
            if len(df_list) > 0:
                df = pd.concat(df_list, ignore_index=True)

        if config['module'] == constants.EXEM_AIOPS_EVENT_FCST:
            if type(inst_type) != dict:
                features = config['train']['eventfcst']['features'][str(inst_type)]
            else:
                features = config['train']['eventfcst']['features']['db'][inst_type[str(target_id)]]
                inst_type = "db"

            df = self.read_parquet(config, inst_type, target_id, features, logger)
            df.drop(columns=["target_id"], inplace=True)

        # 학습 상태 업데이트 '4':학습 진행중
        self.update_train_status(config['train_history_id'], '4')

        return df