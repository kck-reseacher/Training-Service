import json
import os
import pickle
import threading
from io import BytesIO

import joblib
import ml2rt
from redisai import Client
import rejson

from common.system_util import SystemUtil
from common.constants import SystemConstants as sc
from resources.config_manager import Config

os_env = SystemUtil.get_environment_variable()
py_config = Config(os_env[sc.MLOPS_TRAINING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()

redisai_clients, redisjson_clients = {}, {}
'''
이중화 서버를 사용하고 현 서버가 slave 서버가 아닌 경우
redis client는 local과 slave 정보를 둘다 가지고 있어야함.
그 외의 경우에는 local 정보만 가지고 있으면 됨.
'''

if os_env[sc.USE_SLAVE_SERVER]:
    server_keys = [sc.MASTER, sc.SLAVE] if os_env[sc.MLOPS_SERVER_ENV] == sc.MASTER else [sc.SLAVE]
else:
    server_keys=[sc.MASTER]

for keys, info in py_config["redis_server"].items():
    redisai_clients[keys] = Client(host=py_config["redis_server"][keys]["host"], port=int(py_config["redis_server"][keys]["port"]))
    redisjson_clients[keys] = rejson.Client(host=py_config["redis_server"][keys]["host"],
                                            port=int(py_config["redis_server"][keys]["port"]), decode_responses=True)

if not os_env[sc.USE_SLAVE_SERVER]:
    del redisai_clients[sc.SLAVE]
    del redisjson_clients[sc.SLAVE]


print(f"REDIS CONNECTION {redisai_clients}")

class REDISAI:
    @staticmethod
    def exist_key(key):
        if redisai_clients[server_keys[0]].exists(key) == 0:
            return False
        return True

    @staticmethod
    def _set(server, key, data):
        redisai_clients[server].set(key, data)

    @staticmethod
    def _modelstore(server, key, backend, device, data, timestamp):
        redisai_clients[server].modelstore(key, backend, device, data, tag=timestamp)

    ##############
    # model save #
    ##############
    @staticmethod
    def check_model_key(model_key):
        res = redisjson_clients[server_keys[0]].exists(model_key)
        return res

    @staticmethod
    def set_rejson(key, dbsln_model):
        for feat in dbsln_model.keys():
            redisjson_clients[server_keys[0]].jsonset(f"{key}_{feat}", rejson.Path.rootPath(),
                                                      json.loads(dbsln_model[feat].to_json(orient='index')))

    @staticmethod
    def save_onnx_to_redis(onnx_model_path, reload=False):
        """
            onnx file redis write
        """
        model_key = REDISAI.make_redis_model_key(onnx_model_path, ".onnx")
        if model_needs_update(onnx_model_path, model_key, reload):
            model_data = ml2rt.load_model(onnx_model_path)
            model_timestamp = os.path.getmtime(onnx_model_path)

            # threads = []
            for server_key in server_keys:
                thread = threading.Thread(target=REDISAI._modelstore, args=(server_key, model_key, 'ONNX', 'CPU', model_data, model_timestamp))
                thread.start()
                # threads.append(thread)
            # for thread in threads:
            #     thread.join()

            return model_key
        else:
            return f"[exist] {model_key}"

    @staticmethod
    def save_pickle_to_redis(pickle_model_path):
        """
            pickle file redis write
        """
        with open(pickle_model_path, "rb") as p:
            model_data = p.read() # bytes 로 저장.

        model_key = REDISAI.make_redis_model_key(pickle_model_path, ".pkl")

        # threads = []
        for server_key in server_keys:
            thread = threading.Thread(target=REDISAI._set, args=(server_key, model_key, model_data))
            thread.start()
           # threads.append(thread)
        # for thread in threads:
        #     thread.join()

        return model_key # service_1_target

    @staticmethod
    def save_service_to_redis(logger, root_path, pickle_file_list, reload=False):
        for file in pickle_file_list:
            pickle_model_path = os.path.join(root_path, file)

            with open(pickle_model_path, "rb") as f:
                model_dict = pickle.load(f)

            model_key = REDISAI.make_redis_model_key(pickle_model_path, ".pkl")
            if reload:
                REDISAI.set_rejson(model_key, model_dict)
                logger.info(f"RedisJSON set model_key: {model_key}")
            else:
                res = REDISAI.check_model_key(model_key)
                if res == 1:
                    logger.info(f"[exist]: {model_key}")
                else:
                    REDISAI.set_rejson(model_key, model_dict)
                    logger.info(f"RedisJSON set model_key: {model_key}")

    @staticmethod
    def save_json_to_redis(json_file_path):
        """
            json file (model_config) 를 redis write
        """
        with open(json_file_path, "rb") as f:
            json_data = f.read()

        json_key = REDISAI.make_redis_model_key(json_file_path, ".json")

        # threads = []
        for server_key in server_keys:
            thread = threading.Thread(target=REDISAI._set, args=(server_key, json_key, json_data))
            thread.start()
            # threads.append(thread)
        # for thread in threads:
        #     thread.join()

        return json_key

    @staticmethod
    def save_joblib_to_redis(joblib_file_path):
        """
            joblib 로 압축된 pickle file redis write
        """
        with open(joblib_file_path, "rb") as f:
            joblib_data = f.read()

        joblib_key = REDISAI.make_redis_model_key(joblib_file_path, ".pkl")

        # threads = []
        for server_key in server_keys:
            thread = threading.Thread(target=REDISAI._set, args=(server_key, joblib_key, joblib_data))
            thread.start()
        #     threads.append(thread)
        # for thread in threads:
        #     thread.join()

        return joblib_key

    @staticmethod
    def save_3_dbsln_to_redis(dbsln_file_path):
        """
            이상탐지/부하예측 dbsln 3회 dump 된 pickle file redis write
        """
        with open(dbsln_file_path, 'rb') as f:
            training_mode = pickle.load(f)
            biz_status = pickle.load(f)
            biz_status = pickle.dumps(biz_status)
            dbsln_model = pickle.load(f)
            dbsln_model = pickle.dumps(dbsln_model)

        training_mode_key = REDISAI.make_redis_model_key(dbsln_file_path,".pkl")+"_training_mode"
        biz_status_key = REDISAI.make_redis_model_key(dbsln_file_path,".pkl")+"_biz_status"
        dbsln_model_key = REDISAI.make_redis_model_key(dbsln_file_path,".pkl")

        # threads = []
        for server_key in server_keys:
            thread1 = threading.Thread(target=REDISAI._set, args=(server_key, training_mode_key, training_mode))
            thread2 = threading.Thread(target=REDISAI._set, args=(server_key, biz_status_key, biz_status))
            thread3 = threading.Thread(target=REDISAI._set, args=(server_key, dbsln_model_key, dbsln_model))

            thread1.start()
            thread2.start()
            thread3.start()
        #     threads.extend([thread1,thread2,thread3])
        # for thread in threads:
        #     thread.join()

        return training_mode_key, biz_status_key, dbsln_model_key

    @staticmethod
    def save_mem_to_redis(json_key, json_data):
        """
            메모리에 로드된 json 를 redis write
        """
        for key, redisai_client in redisai_clients.items():
            redisai_client.set(json_key, json_data)

        return json_key

    @staticmethod
    def save_body_to_redis(body_key, data):
        """
            event_fcst serving data redis write
        """
        serialized_df = pickle.dumps(data)
        for key, redisai_client in redisai_clients.items():
            res = redisai_client.set(body_key, serialized_df)
        return body_key

    ##############
    ###  추 론  ###
    ##############
    @staticmethod
    def inference(model_key, input_data, data_type='float'):
        """
            input_data: serving data

            input_name : input_data의 unique id
            ex) {algo}_input_{feat}
            SeqAttn_input_tps
            SeqAttn_input_response_time
            ...
            S2S_Attn_input_tps
            S@S_Attn_input_response_time

            output_name : 모델 추론 값의 unique id
            ex) {algo}_output_{feat}
            SeqAttn_output_tps
            SeqAttn_output_response_time
            ...
            S2S_Attn_output_tps
            S@S_Attn_output_response_time
        """
        input_name = f"{model_key}/in"
        output_name = f"{model_key}/out"
        redisai_clients[server_keys[0]].tensorset(input_name, input_data, dtype=data_type)
        redisai_clients[server_keys[0]].modelrun(model_key, inputs=[input_name], outputs=[output_name])
        preds = []
        pred = redisai_clients[server_keys[0]].tensorget(output_name)
        preds.append(pred)

        return preds

    @staticmethod
    def inference_gdn(model_key, input_data, data_type='float'):
        input_name = f"{model_key}/in"
        output_pred = f"{model_key}/out1"
        output_attn = f"{model_key}/out2"
        output_edge = f"{model_key}/out3"

        redisai_clients[server_keys[0]].tensorset(input_name, input_data, dtype="float")
        redisai_clients[server_keys[0]].modelrun(model_key,
                                                 inputs=[input_name],
                                                 outputs=[output_pred,output_attn,output_edge])
        predicate = redisai_clients[server_keys[0]].tensorget(output_pred)
        attention_weight = redisai_clients[server_keys[0]].tensorget(output_attn)
        edge_index = redisai_clients[server_keys[0]].tensorget(output_edge)

        return predicate, attention_weight, edge_index

    @staticmethod
    def event_cpd_inference(model_key, input_data, x_mark_data, y_mark_data):
        input_name = f"{model_key}/in"
        x_mark_name = f"{model_key}/x_mark"
        y_mark_name = f"{model_key}/y_mark"
        output_name = f"{model_key}/out"

        redisai_clients[server_keys[0]].tensorset(input_name, input_data, dtype="float")
        redisai_clients[server_keys[0]].tensorset(x_mark_name, x_mark_data, dtype="float")
        redisai_clients[server_keys[0]].tensorset(y_mark_name, y_mark_data, dtype="float")

        redisai_clients[server_keys[0]].modelrun(model_key,
                                          inputs=[input_name, x_mark_name, y_mark_name],
                                          outputs=[output_name])
        pred = redisai_clients[server_keys[0]].tensorget(output_name)

        return pred

    @staticmethod
    def event_clf_inference(model_key, input_data):
        input_name = f"{model_key}/in"
        output_name = f"{model_key}/out"

        redisai_clients[server_keys[0]].tensorset(input_name, input_data, dtype="float")
        redisai_clients[server_keys[0]].modelrun(model_key, inputs=input_name, outputs=[f'{output_name}1', f'{output_name}2'])
        pred = redisai_clients[server_keys[0]].tensorget(f'{output_name}1')
        recon = redisai_clients[server_keys[0]].tensorget(f'{output_name}2')

        return pred, recon

    @staticmethod
    def inference_pickle(model_key):
        pickled_data = redisai_clients[server_keys[0]].get(model_key)
        model_object = pickle.loads(pickled_data)
        return model_object

    @staticmethod
    def inference_json(model_key):
        json_data = redisai_clients[server_keys[0]].get(model_key)
        model_object = json.loads(json_data)
        return model_object

    @staticmethod
    def inference_joblib(model_key):
        json_data = redisai_clients[server_keys[0]].get(model_key)
        buffer = BytesIO(json_data)
        model_object = joblib.load(buffer)
        return model_object

    @staticmethod
    def inference_log_model(model_key):
        pickled_data = redisai_clients[server_keys[0]].get(model_key)
        model_object = pickle.loads(pickled_data)
        return model_object

    @staticmethod
    def get(model_key):
        try:
            object = redisai_clients[server_keys[0]].get(model_key)
            if type(object) is bytes:
                object = object.decode()
            return object
        except:
            raise print("no model key in redis")

    @staticmethod
    def set(key, value):
        if type(value) is dict or list:
            value = json.dumps(value)
        elif type(value) is bool:
            value = int(value)

        for _, redisai_client in redisai_clients.items():
            redisai_client.set(key, value)

    @staticmethod
    def make_redis_model_key(path, suffix=''):
        return path.split("/model/")[1].replace(suffix,"")


def model_needs_update(path, model_key, reload=False):
    if reload:
        return True
    if any(redisai_clients[server_key].exists(model_key) == 0 for server_key in server_keys):
        return True
    else:
        redisai_model_info = redisai_clients[server_keys[0]].modelget(model_key)
        directory_model_timestamp = os.path.getmtime(path)
        redisai_model_timestamp = redisai_model_info['tag']
        if str(directory_model_timestamp) != redisai_model_timestamp:
            return True
    return False
