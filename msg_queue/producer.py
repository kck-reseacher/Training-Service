from confluent_kafka import Producer, KafkaError
import json


## Producer Example

# Kafka broker address
bootstrap_servers = '10.10.34.31:19092'
# Kafka consumer configuration
producer_conf = {'bootstrap.servers': bootstrap_servers}

# Create a Kafka producer instance
# producer = Producer(producer_conf)

# Kafka topic
topic_name = "auto_training_queue"

class KafkaMessageProducer:
    def __init__(self, producer_conf):
        # Create a Kafka consumer instance
        self.producer = Producer(producer_conf)

    def manual_train_produce(self, train_history_id):
        msg_dict = {"train_history_id": train_history_id}
        value_bytes = json.dumps(msg_dict).encode("utf-8")
        # Produce a message to the Kafka topic
        self.producer.produce(topic=topic_name, key=str(msg_dict['train_history_id']), value=value_bytes)
        print(f"topic_name : {topic_name}, value: {value_bytes} produce")
        # Flush the producer to ensure all messages are delivered
        self.producer.flush()

    def auto_train_produce(self, target_values):

        sample_tid = {"1201": 30663, "1202": 30664, "1203": 30665, "1204": 30666, "1205": 30667}
        for target in target_values:
            # Key and value for the message
            msg_dict = {"train_history_id": sample_tid[target]}
            value_bytes = json.dumps(msg_dict).encode("utf-8")
            # Produce a message to the Kafka topic
            self.producer.produce(topic=topic_name, key=str(msg_dict['train_history_id']), value=value_bytes)
            print(f"topic_name : {topic_name}, value: {value_bytes} produce")
            # Flush the producer to ensure all messages are delivered
            self.producer.flush()