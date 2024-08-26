import base64
import json
from pathlib import Path

class Base64Util:

    def base64encoding(input: str) -> str:
        return str(base64.b64encode(input.encode()), encoding='utf-8')

    def base64decoding(input: str) -> str:
        return base64.b64decode(input).decode(encoding='utf-8')

    @staticmethod
    def get_config_decode_value(config):
        if type(config) is dict:
            for key in config.keys():
                config[key] = Base64Util.base64decoding(config[key])

        return config

if __name__ == "__main__":

    py_path = ""
    py_config = json.loads((Path(py_path) / "config.json").read_text())

    # print(py_config['postgres'])
    #
    # for key in py_config['postgres'].keys():
    #     py_config['postgres'][key] = Base64Util.base64encoding(py_config['postgres'][key])
    # print(py_config)

    decode_config = Base64Util.get_config_decode_value(py_config['postgres'])
    print(decode_config)
