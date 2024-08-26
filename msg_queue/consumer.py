from confluent_kafka import Consumer, KafkaError
import json
import subprocess
import threading
import traceback
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from common.system_util import SystemUtil
from common.constants import SystemConstants as sc
from resources.logger_manager import Logger
from api.tsa.tsa_runner import TSARunner
from api.tsa.tsa_utils import TSAUtils
from msg_queue.config import Config


class KafkaCPUConsumer(threading.Thread):
    def __init__(self, topic, consumer_conf, logger):
        super(KafkaCPUConsumer, self).__init__()
        # Create a Kafka consumer instance
        self.consumer = Consumer(consumer_conf)
        self.topic = topic
        self.logger = logger

    def cpu_train_consume(self):
        self.consumer.subscribe([self.topic])

        with ThreadPoolExecutor(
                max_workers=Config.cpu_training_max_process) as executor:  # max_workers : CPU 학습 최대 프로세스 수
            future_dict = {}
            futures = set()

            while True:
                completed_futures, _ = wait(futures, timeout=0, return_when=FIRST_COMPLETED)

                for completed_future in list(completed_futures):
                    train_info = future_dict.pop(completed_future)
                    process = completed_future.result()
                    if process.returncode == 0:
                        self.logger.info(
                            f"[CPUConsumer] training success TID : {train_info['train_history_id']}")
                    elif process.returncode in [1, 2]:
                        self.logger.info(
                            f"[CPUConsumer] training failed TID: {train_info['train_history_id']}")
                    elif process.returncode == -9:
                        self.logger.info(
                            f"[CPUConsumer] killed process TID: {train_info['train_history_id']}")
                    futures.remove(completed_future)

                msg = self.consumer.poll(1.0)  # 1 second timeout for message polling

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() in [KafkaError._PARTITION_EOF, KafkaError._MAX_POLL_EXCEEDED]:
                        # Reached the end of the partition
                        continue
                    else:
                        self.logger.info(f"[CPUConsumer] message polling error: {msg.error()}")
                        continue

                self.logger.info(f"[CPUConsumer] {self.topic} msg : {msg}")
                # Message processing logic
                value = msg.value().decode('utf-8')
                msg_dict = json.loads(value)
                try:
                    train_meta, status, _ = TSAUtils().get_train_meta(msg_dict['trainHistoryId'])

                    if status != "6":
                        self.logger.info(f"[CPUConsumer] train_history_id: {msg_dict['trainHistoryId']} training start !")
                        mls = train_meta['module']
                        future = executor.submit(TSARunner.training, msg_dict['trainHistoryId'], mls, -1)
                        future_dict[future] = {'train_history_id': msg_dict['trainHistoryId']}
                        futures.add(future)

                except Exception:
                    TSAUtils().update_train_status(msg_dict['trainHistoryId'], '9', 'Error in CPUConsumer')
                    tb = traceback.format_exc()
                    self.logger.info(
                        f"[CPUConsumer] train_history_id: {msg_dict['trainHistoryId']} 학습실패 발생, {tb}"
                    )

    def run(self):
        try:
            self.cpu_train_consume()
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[CPUConsumer] cpu_train_consume Error : {tb}")
            self.consumer.close()


class KafkaGPUConsumer(threading.Thread):
    def __init__(self, topic, consumer_conf, logger):
        super(KafkaGPUConsumer, self).__init__()
        # Create a Kafka consumer instance
        self.consumer = Consumer(consumer_conf)
        self.topic = topic
        self.allocated_gpu_numbers = []
        self.logger = logger

    def check_gpu_count(self):
        os_env = SystemUtil.get_environment_variable()
        GPU_MIG = os_env[sc.GPU_MIG]
        gpu_count = None
        if GPU_MIG:
            cmd = "nvidia-smi -L | grep Device | wc -l"
            gpu_count = int(subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")[0])
        else:
            cmd = "nvidia-smi -L | grep UUID | wc -l"
            gpu_count = int(subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split("\n")[0])
        self.logger.info(f"[GPUConsumer] gpu_count : {gpu_count}")
        return gpu_count

    def gpu_train_consume(self):
        self.consumer.subscribe([self.topic])
        gpu_count = self.check_gpu_count()

        with ThreadPoolExecutor(max_workers=gpu_count) as executor:  # max_workers : GPU counts
            future_dict = {}
            futures = set()

            while True:
                # Check for completed futures
                completed_future, _ = wait(futures, timeout=0, return_when=FIRST_COMPLETED)

                for completed_future in list(completed_future):
                    train_info = future_dict.pop(completed_future)
                    process = completed_future.result()
                    if process.returncode == 0:
                        self.logger.info(
                            f"[GPUConsumer] unallocate gpu {train_info['gpu_number']} due to training success TID : {train_info['train_history_id']}")
                        self.allocated_gpu_numbers.remove(train_info['gpu_number'])
                    elif process.returncode in [1, 2]:
                        self.logger.info(
                            f"[GPUConsumer] unallocate gpu {train_info['gpu_number']} due to training failed TID: {train_info['train_history_id']}")
                        self.allocated_gpu_numbers.remove(train_info['gpu_number'])
                    elif process.returncode == -9:
                        self.logger.info(
                            f"[GPUConsumer] unallocate gpu {train_info['gpu_number']} due to a killed process TID: {train_info['train_history_id']}")
                        self.allocated_gpu_numbers.remove(train_info['gpu_number'])
                    futures.remove(completed_future)

                usable_gpu_numbers = [gpu_number for gpu_number in range(gpu_count) if
                                      gpu_number not in self.allocated_gpu_numbers]

                if len(usable_gpu_numbers) > 0:
                    msg = self.consumer.poll(1.0)  # 1 second timeout for message polling

                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() in [KafkaError._PARTITION_EOF, KafkaError._MAX_POLL_EXCEEDED]:
                            # Reached the end of the partition
                            continue
                        else:
                            self.logger.info(f"[GPUConsumer] message polling error: {msg.error()}")
                            continue

                    self.logger.info(f"[GPUConsumer] {self.topic} msg : {msg}")
                    # Message processing logic
                    # key = msg.key().decode('utf-8')
                    value = msg.value().decode('utf-8')
                    msg_dict = json.loads(value)

                    try:
                        train_meta, status, _ = TSAUtils().get_train_meta(msg_dict['trainHistoryId'])

                        if usable_gpu_numbers and status != "6":
                            mls = train_meta['module']
                            gpu_number = usable_gpu_numbers[0]
                            self.allocated_gpu_numbers.append(gpu_number)
                            self.logger.info(
                                f"[GPUConsumer] gpu_number: {gpu_number}, train_history_id: {msg_dict['trainHistoryId']} training start !")
                            future = executor.submit(TSARunner.training, msg_dict['trainHistoryId'], mls, gpu_number)
                            future_dict[future] = {'train_history_id': msg_dict['trainHistoryId'], 'gpu_number': gpu_number}
                            futures.add(future)

                    except Exception:
                        TSAUtils().update_train_status(msg_dict['trainHistoryId'], '9', 'Error in GPUConsumer')
                        tb = traceback.format_exc()
                        self.logger.info(
                            f"[GPUConsumer] train_history_id: {msg_dict['trainHistoryId']} 학습실패 발생, {tb}"
                        )

    def run(self):
        try:
            self.gpu_train_consume()
        except Exception:
            tb = traceback.format_exc()
            self.logger.info(f"[GPUConsumer] gpu_train_consume Error : {tb}")
            self.consumer.close()


def main():
    py_path, py_config, log_path = TSAUtils().get_server_run_configuration()
    error_log_dict = dict()
    error_log_dict["log_dir"] = str(Path(log_path) / "tsa" / "Error")
    error_log_dict["file_name"] = "consumer"
    logger = Logger().get_default_logger(logdir=log_path + "/tsa/consumer", service_name="consumer",
                                         error_log_dict=error_log_dict)
    gpu_consumer = KafkaGPUConsumer(Config.gpu_train_topic, Config.gpu_consumer_conf, logger)
    cpu_consumer = KafkaCPUConsumer(Config.cpu_train_topic, Config.cpu_consumer_conf, logger)

    gpu_consumer.start()
    cpu_consumer.start()

    gpu_consumer.join()
    cpu_consumer.join()


if __name__ == "__main__":
    main()