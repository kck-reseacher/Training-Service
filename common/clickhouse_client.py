import pandas as pd
import clickhouse_connect
from resources.config_manager import Config
from common.system_util import SystemUtil
from common.base64_util import Base64Util
from common.constants import SystemConstants as sc


os_env = SystemUtil.get_environment_variable()
py_config = Config(os_env[sc.MLOPS_TRAINING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()

def decode_config(key, nginx=None):
    if nginx:
        return Base64Util.base64decoding(py_config[sc.CLICKHOUSE]['nginx'][key])
    else:
        return Base64Util.base64decoding(py_config[sc.CLICKHOUSE][key])

nginx_host, nginx_port = decode_config("host", nginx=True), decode_config("port", nginx=True)
user, password, database = map(decode_config, ["id", "password", "database"])


class ClickhouseClient:
    def __init__(self):
        """
        ClickhouseClient 클래스의 인스턴스가 단 하나만 생성되도록 싱글톤 패턴으로 관리
        """
        self.client = None

    def get_client(self):
        """
        파이썬 프로세스에서 clickhouse db와 연결
        Returns
        -------
        client: clickhouse db 커넥션 객체
        """
        if self.client is None:
            self.client = clickhouse_connect.get_client(
                host=nginx_host,
                port=nginx_port,
                username=user,
                password=password,
                database=database
            )

        return self.client

    def close_client(self):
        """
        파이썬 프로세스에서 clickhouse db 연결을 해제
        Returns
        -------
        """
        if self.client is not None:
            self.client.close()
            self.client = None


clickhouse_client_instance = ClickhouseClient()

def get_client():
    return clickhouse_client_instance.get_client()

def close_client():
    return clickhouse_client_instance.close_client()

def execute_query(query):
    try:
        client = get_client()
        result = client.query(query)
        df = pd.DataFrame(result.result_rows, columns=result.column_names)
    finally:
        close_client()

    return df
