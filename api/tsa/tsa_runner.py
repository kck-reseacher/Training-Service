import subprocess
from msg_queue.producer import KafkaMessageProducer
from api.tsa.tsa_utils import TSAUtils

# Kafka broker address
bootstrap_servers = '10.10.34.31:19092'
# Kafka consumer configuration
producer_conf = {'bootstrap.servers': bootstrap_servers}


class TSARunner:

    @staticmethod
    def manual_training(train_history_id):
        producer = KafkaMessageProducer(producer_conf)
        producer.manual_train_produce(train_history_id)
        pass

    @staticmethod
    def training(train_history_id, mls, gpu_number):
        """
        train_history_id: 학습 프로세스 관리 ID
        mls: Machine Learning Service (module)
        gpu_number: GPU Device ID
        sh training.sh : shell 파일 구동 방식
        -> nohup background 실행 시 학습 완료 시점 추적 불가
        """
        cmd = f"sh training.sh -m {mls} -t {train_history_id} -g {gpu_number}"
        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return process