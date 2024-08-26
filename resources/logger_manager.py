import copy
import os
import logging.config
import json
from pathlib import Path
from datetime import datetime

from common.constants import SystemConstants as sc

class Logger:

    def __init__(self):
        py_path = os.environ.get(sc.MLOPS_TRAINING_PATH)
        self.py_logger_path = py_path + sc.LOGGER_FILE_PATH

        # os 환경 변수 AIMODULE_SERVER_ENV, 환경 변수 없으면 dev
        self.logger_env = os.environ.get(sc.AIMODULE_LOGGER_ENV)
        if self.logger_env is None:
            self.logger_env = "dev"

        logger_file_name = sc.LOGGER_FILE_PREFIX + str(self.logger_env) + sc.LOGGER_FILE_SUFFIX

        self.logger_config = json.load(open(Path(self.py_logger_path) / logger_file_name))
        self.stat_logger_config = copy.deepcopy(self.logger_config)

    def get_default_logger(self, logdir, service_name, error_log_dict=None, train_flag=False):

        Path(logdir).mkdir(exist_ok=True, parents=True)

        if train_flag:
            filename = str(Path(logdir) / "train.log")
        else:
            filename = str(Path(logdir) / f"{service_name}.log")

        self.logger_config["handlers"]["file"]["filename"] = filename

        handlers = list()
        handlers.append("file")

        # error_log_dict 없으면 생성 안하게 설정
        if error_log_dict is not None:
            Path(error_log_dict["log_dir"]).mkdir(exist_ok=True, parents=True)
            error_filename = str(Path(error_log_dict["log_dir"]) / f"{error_log_dict['file_name']}.log")
            self.logger_config["handlers"]["error"]["filename"] = error_filename
            handlers.append("error")

        self.logger_config["loggers"][f"{service_name}"] = dict()
        self.logger_config["loggers"][f"{service_name}"]["handlers"] = handlers

        logging.config.dictConfig(self.logger_config)
        logger = logging.getLogger(service_name)

        return logger

    # train status logger 생성.. 완전히 걷어내게(file > db) 되면 삭제 해도 무방
    def get_stat_logger(self, logdir, service_name):
        stat_name = service_name + "_stat"
        filename = str(Path(logdir) / "status.log")

        self.stat_logger_config["handlers"]["file"]["filename"] = filename
        self.stat_logger_config["handlers"]["file"]["formatter"] = "stat"

        self.stat_logger_config["handlers"].pop("error")

        handlers = list()
        handlers.append("file")

        self.stat_logger_config["loggers"][stat_name] = dict()
        self.stat_logger_config["loggers"][stat_name]["handlers"] = handlers

        logging.config.dictConfig(self.stat_logger_config)
        logger = logging.getLogger(stat_name)

        return logger

    def get_serving_status_logger(self, logdir, target_info):
        Path(logdir).mkdir(exist_ok=True, parents=True)
        status_log_filename = str(Path(logdir) / f"{datetime.today().strftime('%Y-%m-%d_%H-%M')}_serving_status.json")

        if target_info is None:
            target_info = {"get_serving_status_logger": {"fail": "get target_info response is None"}}
        with open(status_log_filename, 'w') as f:
            json.dump(target_info, f, indent=4)

        return True

    # serving.py
    # logger = logging.getLogger(service_name)
    # log_file = str(Path(param["log_dir"]) / f"{service_name}.log")
    # handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1)
    # handler.suffix = "%Y%m%d"
    #
    # fomatter = logging.Formatter(
    #     "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] - %(message)s",
    #     "%Y-%m-%d %H:%M:%S",
    # )
    # handler.setFormatter(fomatter)
    #
    # logger.setLevel(logging.INFO)
    # logger.addHandler(handler)

    # multi_serving.py
    # logger = logging.getLogger("serving")
    # log_file = str(Path(param["log_dir"]) / "serving.log")
    # handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1)
    # handler.suffix = "%Y%m%d"
    #
    # fomatter = logging.Formatter(
    #     "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] - %(message)s",
    #     "%Y-%m-%d %H:%M:%S",
    # )
    # handler.setFormatter(fomatter)
    #
    # logger.setLevel(logging.INFO)
    # logger.addHandler(handler)

    # train.py
    # logger
    # logger = logging.getLogger(service_name)
    # log_file = str(Path(param["log_dir"]) / "train.log")
    # handler = TimedRotatingFileHandler(
    #     log_file, when="midnight", interval=1, encoding="utf-8"
    # )
    # handler.suffix = "%Y%m%d"
    #
    # fomatter = logging.Formatter(
    #     "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] - %(message)s",
    #     "%Y-%m-%d %H:%M:%S",
    # )
    # handler.setFormatter(fomatter)
    #
    # logger.setLevel(logging.INFO)
    # logger.addHandler(handler)

    # stat_logger
    # stat_name = service_name + "_stat"
    # stat_logger = logging.getLogger(stat_name)
    # stat_log_file = str(Path(param["log_dir"]) / "status.log")
    # stat_handler = TimedRotatingFileHandler(stat_log_file, when="midnight", interval=1)
    # stat_handler.suffix = "%Y%m%d"
    #
    # stat_fomatter = logging.Formatter("%(asctime)s > %(message)s")
    # stat_handler.setFormatter(stat_fomatter)
    #
    # stat_logger.setLevel(logging.INFO)
    # stat_logger.addHandler(stat_handler)

    # ..multi.py
    # logger_name = f"{self.module_id}_{self.inst_type}_{self.sys_id}_{target_id}"
    # logger = logging.getLogger(logger_name)
    # log_file = str(Path(self.log_dir) / f"{logger_name}.log")
    # handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1)
    # handler.suffix = "%Y%m%d"
    # fomatter = logging.Formatter(
    #     "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] - %(message)s",
    #     "%Y-%m-%d %H:%M:%S",
    # )
    # handler.setFormatter(fomatter)
    # logger.setLevel(logging.INFO)
    # logger.addHandler(handler)