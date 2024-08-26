import requests
import argparse
import json
import os
import sys
import traceback
import subprocess
import re
from datetime import datetime

from pathlib import Path

from common import aicommon, constants
from common.error_code import Errors
from common.module_exception import ModuleException
from common.base64_util import Base64Util
from common.constants import SystemConstants as sc
from resources.config_manager import Config
from resources.logger_manager import Logger
from common.system_util import SystemUtil
from common.clickhouse_client import close_client
from api.tsa.tsa_utils import TSAUtils


class MLOpsTrainer():
    def __init__(self, train_history_id, gpu_number):
        self.train_history_id = train_history_id
        self.gpu_number = gpu_number
        self.os_env = SystemUtil.get_environment_variable()
        self.py_config = Config(self.os_env[sc.MLOPS_TRAINING_PATH], self.os_env[sc.AIMODULE_SERVER_ENV]).get_config()

    def load_train_meta(self, train_meta, regenerate_train_data):
        self.train_meta = train_meta
        self.train_meta["regenerate_train_data"] = regenerate_train_data
        self.sys_id = self.py_config["sys_id"]
        self.module_name = self.train_meta["module"]
        self.inst_type = self.train_meta["inst_type"] if "inst_type" in self.train_meta.keys() else self.train_meta[
            "module"]
        if self.module_name == 'exem_aiops_anls_log':
            self.inst_type = 'log'
        self.target_id = self.train_meta["target_id"]

    def get_gpu_uuid(self):
        """
        GPU UUID 정보를 가져오는 함수
        OS 환경변수 - GPU_MIG 가 True일 때만 실행
        nvidia-smi의 MIG Device가 Setting 안되어 있으면 raise Exception으로 학습 종료
        """
        cmd = "nvidia-smi -L"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        gpu_uuids = []
        lines = output.strip().split("\n")
        pattern = r"UUID:\s*(MIG-\S+[0-9a-zA-Z])"
        for line in lines:
            match = re.search(pattern, line)
            if match:
                gpu_uuids.append(match.group(1))
        if len(gpu_uuids) == 0:
            raise Exception("Please Setting MIG Mode !!")

        return gpu_uuids

    def get_train_parameter(self, train_data_path):

        param = self.train_meta.copy()
        param["sys_id"] = self.sys_id
        service_name = f"{self.module_name}_{self.inst_type}_{self.sys_id}_{self.target_id}"

        # train_dir
        if self.module_name in constants.TRAINING_MODULE_LIST:
            # 학습모듈 전용, 변경된 학습데이터 구조 적용
            param[
                "train_dir"] = f"{self.os_env[sc.AIMODULE_HOME]}/train_data/{self.sys_id}/{self.inst_type}/{self.target_id}"
        elif self.module_name in constants.MULTI_TARGET_TRAINING_MODULE_LIST:
            param["train_dir"] = f"{self.os_env[sc.AIMODULE_HOME]}/train_data/{self.sys_id}/{self.inst_type}"
        else: # 분석 모듈
            param["train_dir"] = train_data_path

        # home_dir
        param["home"] = self.os_env[sc.AIMODULE_HOME]

        # model_dir
        model_dir = Path(
            f"{self.os_env[sc.AIMODULE_HOME]}/model/{self.sys_id}/{self.module_name}/{self.inst_type}/{self.target_id}")
        model_dir.mkdir(exist_ok=True, parents=True)
        param["model_dir"] = str(model_dir)

        # log_dir
        log_dir = Path(
            f"{self.os_env[sc.MLOPS_LOG_PATH]}/train/{self.sys_id}/{self.module_name}/{self.inst_type}/{self.target_id}")
        param["log_dir"] = str(log_dir)

        # 데이터 베이스 접속 설정
        try:
            pg_decode_config = Base64Util.get_config_decode_value(self.py_config[sc.POSTGRES])
        except Exception:
            tb = traceback.format_exc()
            print(f"pg_decode_config exception : {tb}")
            # config 값 base64 decoding 시 encoding이 안된값이 있을 경우 except catch 하여 기본값 으로 세팅
            print('base64 decode error, config : ' + self.py_config[sc.POSTGRES])
            pg_decode_config = self.py_config[sc.POSTGRES]

        param["db_conn_str"] = (
            f"host={pg_decode_config['host']} "
            f"port={pg_decode_config['port']} "
            f"dbname={pg_decode_config['database']} "
            f"user={pg_decode_config['id']} "
            f"password={pg_decode_config['password']}"
        )

        # API 서버 접속 정보
        param["api_server"] = self.py_config[sc.API_SERVER]

        param["parameter"] = {"train": param["train"] if param.get("train") is not None else None,
                              "service": None,
                              "data_set": param["data_set"] if param.get("data_set") is not None else None}

        if param.get("train") is None:
            print(f"{service_name} train parameter does not exist in train_meta.json")

        return param

    def get_logger(self, param):
        service_name = f"{self.module_name}_{self.inst_type}_{self.sys_id}_{self.target_id}"

        # module_error_dir
        error_log_dict = dict()
        error_log_dict["log_dir"] = f"{self.os_env[sc.MLOPS_LOG_PATH]}/train/{self.sys_id}/{sc.ERROR_LOG_DEFAULT_PATH}"
        error_log_dict["file_name"] = service_name

        # logger
        logger = Logger().get_default_logger(logdir=param["log_dir"], service_name=service_name,
                                             error_log_dict=error_log_dict, train_flag=True)
        stat_logger = Logger().get_stat_logger(logdir=param["log_dir"], service_name=service_name)

        return logger, stat_logger

    def training(self, train_data_path):

        if self.os_env[sc.GPU_MIG]:
            gpu_uuids = self.get_gpu_uuid()
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuids[int(self.gpu_number)]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_number
        print(f"usable cuda devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

        param = self.get_train_parameter(train_data_path)
        logger, stat_logger = self.get_logger(param)
        class_name = aicommon.Utils.to_camel_case(self.module_name)
        target_class = aicommon.Utils.get_module_class(
            self.module_name, class_name, f"{self.os_env[sc.MLOPS_TRAINING_PATH]}/analyzer"
        )
        instance = target_class(param, logger)

        header = None
        body = None
        errno = -1
        errmsg = None

        logger.info("============== Start Training ==============")
        logger.info("Train meta info (%s)", param)

        analysis_train_result = dict()
        analysis_train_result["results"] = False
        analysis_train_result["body"] = {
            "master_id": None
        }

        model_train_result = dict()
        exception_result = dict()

        # make default train_result
        default_error_result = {
            "results": False,
            "body": {"master_id": None},
            "train_time": datetime.now(),
            "errno": Errors['DEFAULT_ERROR'].value,
            "errmsg": Errors['DEFAULT_ERROR'].desc
        }
        if train_data_path is None:
            train_data_path = f"{self.os_env[sc.AIMODULE_HOME]}/train_request/{self.sys_id}/{self.module_name}/{str(self.train_history_id)}"
            Path(train_data_path).mkdir(exist_ok=True, parents=True)
            (Path(train_data_path) / "train_result.json").write_text(
                json.dumps({**param, **default_error_result}, cls=aicommon.JsonEncoder, ensure_ascii=False)
            )

        try:
            model_train_result["train_start_time"] = datetime.now()
            instance.init_train()
            header, body, errno, errmsg = instance.train(stat_logger)
            if self.module_name in constants.ANALYZER_MODULE_LIST:
                analysis_train_result["results"] = body[
                    "results"] if body is not None and "results" in body.keys() else False
                analysis_train_result["body"] = {
                    "master_id": body["master_id"] if body is not None and "master_id" in body.keys() else None}
                analysis_train_result["errno"] = errno
                analysis_train_result["errmsg"] = errmsg
            else:
                model_train_result["train_time"] = datetime.now()
                model_train_result["errno"] = errno

                if errmsg is not None:
                    model_train_result["errmsg"] = errmsg

                if header is not None:
                    if self.module_name == constants.EXEM_AIOPS_ANLS_SERVICE:
                        model_train_result["empty_data_target_list"] = instance.empty_data_target_dict[
                            "empty_data_target_list"]

                    model_train_result["results"] = header

                if body is not None:
                    model_train_result["body"] = body

        except MemoryError as error:
            logger.exception("트레이닝 과정에 MemoryError 오류가 발생하였습니다.", error)
            aicommon.Utils.print_memory_usage(logger)
            exception_result["train_time"] = datetime.now()
            exception_result["errno"] = Errors['E889'].value
            exception_result["errmsg"] = Errors['E889'].desc

        except ModuleException as me:
            logger.exception(f"트레이닝 과정에 오류가 발생하였습니다. : {me}")
            exception_result["train_time"] = datetime.now()
            exception_result["errno"] = me.error_code
            exception_result["errmsg"] = me.error_msg

        except Exception:
            tb = traceback.format_exc()
            logger.info(f"트레이닝 과정에 오류가 발생하였습니다 : {tb}")
            exception_result["train_time"] = datetime.now()
            exception_result["errno"] = Errors['E888'].value
            exception_result["errmsg"] = Errors['E888'].desc

        finally:
            # result file save
            merged_model_train_result = {**param, **model_train_result, **exception_result}
            merged_analysis_train_result = {**analysis_train_result, **exception_result}

            # 정상 학습 종료
            if errno == 0:
                (Path(param['model_dir']) / "model_config.json").write_text(
                    json.dumps(merged_model_train_result, cls=aicommon.JsonEncoder, ensure_ascii=False)
                )

            # result file save
            if self.module_name in constants.ANALYZER_MODULE_LIST:
                (Path(train_data_path) / "train_result.json").write_text(
                    json.dumps(merged_analysis_train_result, cls=aicommon.JsonEncoder, ensure_ascii=False)
                )
            else:
                (Path(train_data_path) / "train_result.json").write_text(
                    json.dumps(merged_model_train_result, cls=aicommon.JsonEncoder, ensure_ascii=False)
                )

            # dummy로 넘겨주던 sys_id 제거
            del merged_model_train_result["sys_id"]
            instance.end_train()
            logger.info("============== End Training ==============")

        return merged_model_train_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training argument")
    parser.add_argument('-d', nargs='?', const=None, type=str, help="train data path")
    parser.add_argument('-t', nargs='?', const=None, type=int, help="train history id")
    parser.add_argument('-g', type=str, help="gpu number")
    args = parser.parse_args()

    train_data_path = args.d
    train_history_id = args.t
    gpu_number = args.g

    trainer = MLOpsTrainer(train_history_id, gpu_number)
    tsa = TSAUtils()

    if train_history_id is None:
        if SystemUtil.is_windows_os():
            train_meta = json.loads((Path(train_data_path) / "train_meta.json").read_text("UTF-8"))
        else:
            train_meta = json.loads((Path(train_data_path) / "train_meta.json").read_text())
        regenerate_train_data = False
    else:
        train_meta, status, regenerate_train_data = tsa.get_train_meta(train_history_id)
        tsa.update_train_start_time(train_history_id)

    trainer.load_train_meta(train_meta, regenerate_train_data)

    try:
        train_result = trainer.training(train_data_path)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"training Exception: {tb}")
        if train_history_id is None:
            pass
        else:
            exception_msg = f"{type(e).__name__}: {e}"
            tsa.update_train_status(train_history_id, '9', exception_msg)
            sys.exit()
    finally:
        close_client()

    if train_data_path is None:
        if train_result["errno"] == 0:
            try:
                # train_result 를 PG에 Update
                tsa.update_train_history(train_history_id, train_result)
                tsa.update_config_service(train_result)
                if train_result['module'] in [constants.EXEM_AIOPS_ANLS_LOG]:
                    tsa.insert_log_parameter(train_result)

                # 학습 상태 업데이트 '5': 학습 완료
                tsa.update_train_status(train_history_id, '5')

                # 학습 완료된 모델 파일을 Redis-server에 Store
                py_path, py_config, log_path = tsa.get_server_run_configuration()
                # MLOps-TSA를 container로 전환 시 Host IP, Port number 변경 필요
                mlc_url = f"http://{py_config['mlc']['host']}:{py_config['mlc']['port']}"
                model_update_url = f"mlc/model/{py_config['sys_id']}/{train_meta['module']}/{train_meta['inst_type']}/{train_meta['target_id']}"
                update_api_res = requests.patch(f"{mlc_url}/{model_update_url}")
                if update_api_res.status_code == 200:
                    print(f"[MLC] model update res: {update_api_res}")
                else:
                    print("[MLC] model update failure")

                # DBSLN 학습 시 /reload/cache API 요청 (TSA -> be)
                if train_meta['module'] == 'exem_aiops_anls_service':
                    dbsln_url = tsa.get_dbsln_url()
                    res = requests.get(f"{dbsln_url}/reload/cache")
                    print(f"dbsln reload res: {res.status_code}")

            except Exception as e:
                tb = traceback.format_exc()
                print(f"train result update Exception: {tb}")
                exception_msg = f"{type(e).__name__}: {e}"
                tsa.update_train_status(train_history_id, '9', exception_msg)
        else:
            # 학습 상태 업데이트 '9': 학습 실패
            tsa.update_train_status(train_history_id, '9', train_result['errmsg'])
    else:
        pass
