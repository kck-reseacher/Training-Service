import os
import json
import sys

from pathlib import Path
from common.constants import SystemConstants as sc

from common.base64_util import Base64Util

class Config:

    def __init__(self, py_path, server_env):
        self.py_config_path = py_path + sc.CONFIG_FILE_PATH
        # os 환경 변수 AIMODULE_SERVER_ENV
        self.env = server_env

    # flask 환경 별 config.json 을 로드한다.
    def get_config(self):
        if self.env is not None:
            # config.json 파일 이름 ex) 개발계 config-dev.json
            config_file_name = sc.CONFIG_FILE_PREFIX + str(self.env) + sc.CONFIG_FILE_SUFFIX
        else:
            print("plz export AIMODULE_SERVER_ENV")
            sys.exit()

        py_config = json.loads((Path(self.py_config_path) / config_file_name).read_text())

        return py_config

    """
    환경 별 config.json에 실제 사용되는 값을 넣어주고 
    인코딩이 필요한 key를 설정하고 main을 실행해서 인코딩된 config를 생성 해준다.     

    """
    # base 64 encoding 적용할 config value generate를 위한 config.json 내 1depth key list
    base64_applied_configkey = list()
    base64_applied_configkey.append(sc.POSTGRES)

    def generate_config(self):
        config_name = sc.CONFIG_FILE_PREFIX + str(self.env) + sc.CONFIG_FILE_SUFFIX

        with open(self.py_config_path + config_name, 'r') as f:
            config = json.load(f)

        for configkey in Config.base64_applied_configkey:
            for key in config[configkey].keys():
                config[configkey][key] = Base64Util.base64encoding(config[configkey][key])

        with open(self.py_config_path + sc.CONFIG_FILE_PREFIX + str(self.env) + sc.CONFIG_FILE_SUFFIX, 'w', encoding='utf-8') as make_file:
            json.dump(config, make_file, indent="\t")

    # 인코딩된 config 확인용
    def check_config_value(self):
        config_name = sc.CONFIG_FILE_PREFIX + str(self.env) + sc.CONFIG_FILE_SUFFIX

        with open(self.py_config_path + config_name, 'r') as f:
            config = json.load(f)

        for configkey in Config.base64_applied_configkey:
            for key in config[configkey].keys():
                config[configkey][key] = Base64Util.base64decoding(config[configkey][key])

        print(json.dumps(config, indent="\t"))

if __name__ == "__main__":
    py_path = os.environ.get(sc.MLOPS_TRAINING_PATH)
    server_env = os.environ.get(sc.AIMODULE_SERVER_ENV)

    if py_path is None:
        py_path = "C:/Users/shshin/Documents/AI1/aiops-module"

    config = Config(py_path, server_env)
    config.generate_config()

    # base64 인코딩된 pg의 디코딩 정보를 알고싶을때 generate_config 주석 후 아래 함수 실행행
    # config.check_config_value()
    print("config json data base64 trans success")


