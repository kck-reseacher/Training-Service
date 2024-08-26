import psycopg2 as pg2

from common.base64_util import Base64Util
from common.constants import SystemConstants as sc
from api.tsa.tsa_utils import TSAUtils


def get_db_conn_str():
    py_path, py_config, log_path = TSAUtils.get_server_run_configuration()

    try:
        pg_decode_config = Base64Util.get_config_decode_value(py_config[sc.POSTGRES])
    except Exception as e:
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

def get_db_conn_obj():
    return pg2.connect(get_db_conn_str())