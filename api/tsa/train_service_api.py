import json
from pathlib import Path
import subprocess
from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI, Depends, Body
from fastapi.responses import JSONResponse
import psycopg2 as pg2
import psycopg2.extras

from resources.logger_manager import Logger
from resources.config_manager import Config
from api.tsa.tsa_utils import TSAUtils
from database.connection_object import get_db_conn_obj
from common.system_util import SystemUtil
from common.constants import SystemConstants as sc


##########################################################
#                    Uvicorn Fast API                    #
##########################################################

app = FastAPI(title="XAIOps Training-Service-Api")
os_env = SystemUtil.get_environment_variable()
py_config = Config(os_env[sc.MLOPS_TRAINING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()


@app.put("/train/cancel")
def train_cancel(trainHistoryIdList: List[int] = Body(...), conn=Depends(get_db_conn_obj)):
    train_history_id_list = trainHistoryIdList

    for train_history_id in train_history_id_list:
        cursor = conn.cursor(cursor_factory=pg2.extras.RealDictCursor)
        query = f"""
                        UPDATE
                            ai_history_train
                        SET
                            status = '6',
                            warning_comment = '"stop training by user."',
                            train_end_time = to_timestamp('{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}', 'YYYY-MM-DD HH24:MI:SS.US')
                        WHERE
                            history_train_id = {train_history_id}
                    """

        try:
            cursor.execute(query)
            conn.commit()
        except Exception as e:
            logger.error(f"{e}")
            conn.rollback()
            return JSONResponse(status_code=500,
                                content={
                                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                                    "status": 500,
                                    "error": f"train_history_id[{train_history_id}] is not in Database",
                                    "path": '/train/cancel'}
                                )
        finally:
            cursor.close()

    for train_history_id in train_history_id_list:
        try:
            cmd = f"ps -ef | grep \'train.py -t {train_history_id} -g\' | grep -v grep | awk \'{{print $2}}\'"
            logger.info(cmd)
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, \
                                     stderr=subprocess.PIPE, text=True, check=True)
            pid = res.stdout
            if not pid:
                continue

        except subprocess.CalledProcessError as e:
            logger.error(f"There is no train_history_id[{train_history_id}] process")
            continue

        try:
            cmd = f"kill -9 {pid}"
            logger.info(cmd)
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, \
                                     stderr=subprocess.PIPE, text=True, check=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"There is no train_history_id[{train_history_id}] process")
            continue

    logger.info(f"train_history_id_list[{train_history_id_list}] is canceled")
    return JSONResponse(status_code=200, content=True)


@app.delete("/train/data/{instType}")
def delete_data(instType: str):
    # api에서 sys_id제거 되서 dummy sys_id 추가
    sysId = py_config["sys_id"]
    try:
        cmd = f"rm -rf {os_env[sc.AIMODULE_HOME]}/train_data/{sysId}/{instType}"
        logger.info(cmd)
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, \
                                 stderr=subprocess.PIPE, text=True, check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to remove directory: /{os_env[sc.AIMODULE_HOME]}/train_data/{sysId}/{instType}")
        return JSONResponse(status_code=500,
                            content={
                                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
                                "status": 500,
                                "error": f"Internal Error",
                                "path": f"/train/data/{sysId}/{instType}"}
                            )

    return JSONResponse(status_code=200, content=True)


@app.on_event("startup")
async def startup_event():
    logger.info(f"TRAINING SERVICE API start !!")
    print("TRAINING SERVICE API START....")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"TRAINING SERVICE API shutdown !!")
    print(f"TRAINING SERVICE API SHUTDOWN....")


if __name__ == "__main__":
    py_path, py_config, log_path = TSAUtils.get_server_run_configuration()
    log_conf = json.load(open(Path(py_path + sc.LOGGER_FILE_PATH) / sc.UVICORN_LOGGER_CONFIG_FILE))

    error_log_dict = dict()
    error_log_dict["log_dir"] = str(Path(log_path) / "tsa" / "Error")
    error_log_dict["file_name"] = "train_service_api"
    logger = Logger().get_default_logger(logdir=log_path + "/tsa", service_name="train_service_api", error_log_dict=error_log_dict)

    uvicorn.run(
        app,
        host=py_config['train_api_server']['host'],
        port=py_config['train_api_server']['port'],
        access_log=True,
        reload=False,
        log_level="info",
        log_config=log_conf,
        workers=1,
    )
