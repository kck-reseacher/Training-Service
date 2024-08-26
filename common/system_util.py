import os
import sys
import argparse
import platform
import json

from common.constants import SystemConstants as sc
from common import aicommon
from pathlib import Path

class SystemUtil:

    @staticmethod
    def get_run_function_name() -> str:
        name = ""
        try:
            name = sys._getframe().f_code.co_name
        except SystemError:
            return name
        return name

    @staticmethod
    def get_class_name(cls) -> str:
        return type(cls).__name__

    @staticmethod
    def get_environment_variable():
        os_env = dict()
        # AIMODULE_HOME
        home = os.environ.get(sc.AIMODULE_HOME)
        if home is None:
            print("plz export AIMODULE_HOME")
            home = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.AIMODULE_HOME] = home

        # MLOPS_LOG_PATH
        log_path = os.environ.get(sc.MLOPS_LOG_PATH)
        if log_path is None:
            print("plz export MLOPS_LOG_PATH")
            log_path = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.MLOPS_LOG_PATH] = log_path

        # MLOPS_TRAINING_PATH
        py_path = os.environ.get(sc.MLOPS_TRAINING_PATH)
        if py_path is None:
            print("plz export MLOPS_TRAINING_PATH")
            py_path = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.MLOPS_TRAINING_PATH] = py_path

        # AIMODULE_SERVER_ENV
        server_env = os.environ.get(sc.AIMODULE_SERVER_ENV)
        if server_env is None:
            print("plz export AIMODULE_SERVER_ENV")
            py_path = os.path.dirname(os.path.abspath(__file__))
        else:
            os_env[sc.AIMODULE_SERVER_ENV] = server_env

        # MLOPS_SERVER_ENV
        mlops_server_env = os.environ.get(sc.MLOPS_SERVER_ENV)
        if mlops_server_env is None:
            print("plz export MLOPS_SERVER_ENV")
        else:
            os_env[sc.MLOPS_SERVER_ENV] = mlops_server_env.lower()

        # USE_SLAVE_SERVER
        use_slave_server = os.environ.get(sc.USE_SLAVE_SERVER)
        if use_slave_server is None:
            print("plz export USE_SLAVE_SERVER")
        else:
            os_env[sc.USE_SLAVE_SERVER] = use_slave_server.lower() == "true" if use_slave_server else False

        # GPU_MIG
        gpu_mig = os.environ.get(sc.GPU_MIG)
        if gpu_mig is None:
            print("plz export GPU_MIG")
        else:
            os_env[sc.GPU_MIG] = gpu_mig.lower() == "true" if gpu_mig else False

        return os_env

    @staticmethod
    def get_server_start_param(module_name):
        # 입력 인자 설정
        if len(sys.argv) < 4:
            aicommon.Utils.usage()
            sys.exit()

        target_flag = False if "multi" in module_name else True

        parser = argparse.ArgumentParser(prog=module_name, description=module_name, add_help=True)
        parser.add_argument("-m", "--module", help="module name.", required=True)
        parser.add_argument("-t", "--target", help="target name", required=target_flag)
        parser.add_argument("-p", "--port", help="port number", required=True)
        parser.add_argument("-s", "--sys_id", help="system id", required=target_flag)
        parser.add_argument(
            "-i", "--inst_type", help="instance type [db, os, was].", required=target_flag
        )

        args = parser.parse_args()
        module_name = args.module
        target_id = args.target
        target_port = args.port
        sys_id = args.sys_id
        inst_type = args.inst_type

        return module_name, target_id, target_port, sys_id, inst_type

    @staticmethod
    def set_model_config(model_config_path, param, logger, rsync_model_to_db=None):
        if rsync_model_to_db is not None:
            if rsync_model_to_db.access_model_config(model_config_path):
                model_config = rsync_model_to_db.update_model_config(model_config_path, param)
                if model_config is False:
                    logger.exception("can't load model config file")
                    logger.error("exit serving process with code no.1")
                    sys.exit(1)
        else:
            model_config = json.loads(Path(model_config_path).read_text(encoding='utf-8'))

        # 학습 지표 셋 설정
        if model_config.get("data_set"):
            param["data_set"] = model_config["data_set"]
        if model_config.get("weight"):
            param["weight"] = model_config["weight"]
        if model_config.get("business_list"):
            param["business_list"] = model_config["business_list"]
        if model_config.get("agg_type"):
            param["agg_type"] = model_config["agg_type"]
        if model_config.get("tier_map"):
            param["tier_map"] = model_config["tier_map"]
        if model_config.get("service_list"):  # 서비스 모니터링 분석
            param["service_list"] = model_config["service_list"]
        if model_config.get("empty_data_target_dict"):  # 서비스 모니터링 분석
            param["empty_data_target_dict"] = model_config["empty_data_target_dict"]
        if model_config.get("results"):  # 서비스 모니터링 분석
            param["train_results"] = model_config["results"]
        if model_config.get("parameter"):
            param["parameter"] = model_config["parameter"]

        return param

    @staticmethod
    def is_windows_os():
        return True if "windows" in platform.platform().lower() else False
