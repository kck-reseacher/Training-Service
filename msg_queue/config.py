from api.tsa.tsa_utils import TSAUtils


class Config:
    py_path, py_config, log_path = TSAUtils().get_server_run_configuration()

    # Kafka broker address
    bootstrap_servers = f"{py_config['kafka']['host']}:{py_config['kafka']['port']}"

    gpu_consumer_conf = {'bootstrap.servers': bootstrap_servers,
                     'group.id': 'ai-train-dev2_true',
                     'auto.offset.reset': 'latest',
                     'max.poll.interval.ms': 86400000,
                     'enable.auto.commit': True}

    cpu_consumer_conf = {'bootstrap.servers': bootstrap_servers,
                         'group.id': 'ai-train-dev2_false',
                         'auto.offset.reset': 'latest',
                         'max.poll.interval.ms': 86400000,
                         'enable.auto.commit': True}

    cpu_train_topic = "train_central"
    gpu_train_topic = "train_graphics"
    cpu_training_max_process = 6
