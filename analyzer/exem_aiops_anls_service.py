from copy import deepcopy
import time
import numpy as np
import pathos
import psutil

from analyzer import aimodule
from analyzer.exem_aiops_anls_service_service import ExemAiopsAnlsServiceService
from analyzer.exem_aiops_anls_service_inst import ExemAiopsAnlsServiceInst
from common.error_code import Errors
from common import aicommon, constants


class ExemAiopsAnlsService(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        initialize instance attributes using JSON config information and Confluence documentation. AI Server inputs the JSON config information.
        :param config:
        :param logger:
        """
        self.config = config
        self.logger = logger

        # BE - empty_data_target_list 정보 필요 시 작업 예정
        self.empty_data_target_dict = dict()
        self.empty_data_target_dict["empty_data_target_list"] = list()

        if self.config["inst_type"] == 'service':
            self.dbsln_analyzer = ExemAiopsAnlsServiceService(self.config, logger)
        else:  # DBSLN for instance like WAS, DB, OS ......
            # config 형태 맞추기
            self.dbsln_analyzer = []
            for s in self.config['service_list']:
                target_config = deepcopy(self.config)
                target_config['target_id'] = s
                target_config['train_dir'] = self.config["home"] + "/train_data/" + str(self.config['sys_id']) + "/" + self.config["inst_type"] + "/" + s
                self.dbsln_analyzer.append(ExemAiopsAnlsServiceInst(target_config, self.logger))

            # INST DBSLN 병렬 처리 관련 param
            self.number_of_child_processes = int(psutil.cpu_count(logical=True) * 0.2)
            if self.number_of_child_processes <= 0:
                self.number_of_child_processes = 1

            chunk_size, remainder = divmod(
                len(self.dbsln_analyzer), self.number_of_child_processes
            )
            if remainder != 0:
                chunk_size += 1
            self.chunk_size = chunk_size

    def _train_proc(self, train_prog, cnt, proc_type, process_start_time, done):
        if done:
            proc = 100
        else:
            proc = int(((cnt + 1) / self.chunk_size * 100))

        aicommon.Query.update_module_status_by_training_id(
            self.config['db_conn_str'],
            self.config['train_history_id'],
            train_prog,
            train_process=proc_type,
            progress=proc,
            process_start_time=round(time.time() - process_start_time, 2)
        )

    def train(self, stat_logger):
        if self.config["inst_type"] == 'service':
            header, body, errno, errmsg = self.dbsln_analyzer.train(stat_logger)
            return header, body, errno, errmsg
        else:  # DBSLN for instance like WAS, DB, OS ......
            process_start_time = time.time()
            train_prog = {
                #'process train data': self.default_module_status(),
                'Dynamic Baseline': self.default_module_status()
            }
            result_list = []
            for i in range(self.chunk_size):
                pool = pathos.multiprocessing.Pool(processes=self.number_of_child_processes)
                n = i*self.number_of_child_processes
                if i+1 == self.chunk_size:
                    result = pool.map(lambda x: x.train(), self.dbsln_analyzer[n:])
                else:
                    result = pool.map(lambda x: x.train(), self.dbsln_analyzer[n:n+self.number_of_child_processes])
                result_list.extend(result)
                self._train_proc(train_prog, i, constants.MODEL_F_DBSLN, process_start_time=process_start_time,
                                 done=False)

            self._train_proc(train_prog, -1, constants.MODEL_F_DBSLN, process_start_time=process_start_time, done=True)

            # Return 포맷 생성
            train_target_list = []
            duration_time = 0
            train_mode_dict = {}
            train_business_status_dict = {}
            dummy = 0
            for i in range(len(result_list)):
                if result_list[i][2] == 0:  # 학습이 정상 종료된 target
                    target_id = self.config['service_list'][i]
                    train_target_list.append(target_id)
                    train_business_status_dict[target_id] = result_list[i][0]["dbsln"]["train_business_status"]
                    train_mode_dict[target_id] = result_list[i][0]["dbsln"]["train_mode"]
                    # 정상 학습 종료된 result dummy 형태 이용
                    dummy = i
                    # total duration 생성
                    duration_time += result_list[i][0]["dbsln"]["results"]["duration_time"]
            if len(train_target_list) == 0:  # 학습 성공한 target 1개도 없다.
                self.logger.info(f"All {self.config['inst_type']} target train fail")
                return None, None, result_list[i][2], result_list[i][3]

            untrained_target_list = list(set(self.config.get("service_list", [])) - set(train_target_list))
            result = deepcopy(result_list[dummy][0])
            result["dbsln"]["train_business_status"] = train_business_status_dict  # biz status
            result["dbsln"]["train_mode"] = train_mode_dict            # train mode
            result["dbsln"]["duration_time"] = round(duration_time, 2)
            result["dbsln"]["train_target_list"] = np.unique(train_target_list)
            result["dbsln"]["untrained_target_list"] = untrained_target_list
            return result, None, 0, None