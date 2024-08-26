import time
import schedule
import numpy as np
from api.tsa.tsa_runner import TSARunner
from msg_queue.producer import KafkaMessageProducer


# Kafka broker address
bootstrap_servers = '10.10.34.31:19092'
# Kafka consumer configuration
producer_conf = {'bootstrap.servers': bootstrap_servers}


class AutoTrainScheduler:

    def __init__(self):
        # Initialize schedule
        schedule.every()
        self.producer = KafkaMessageProducer(producer_conf)
        self.db_conn = TSARunner.get_db_connection()

    def schedule_every_week(self, day, hour, minute, target_values):
        if day == 0:
            schedule.every().monday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)
        elif day == 1:
            schedule.every().tuesday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)
        elif day == 2:
            schedule.every().wednesday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)
        elif day == 3:
            schedule.every().thursday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)
        elif day == 4:
            schedule.every().friday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)
        elif day == 5:
            schedule.every().saturday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)
        elif day == 6:
            schedule.every().sunday.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce, target_values)

    def train_scheduling(self):
        ## 최초 스케줄링 등록 시 DB에 저장된 모든 mls에 대한 스케줄을 등록
        ## 단발성 업데이트 시 업데이트된 mls에 대해서만 스케줄을 재등록
        # meta info
        auto_train_df = TSARunner.get_train_schedule(self.db_conn)
        for inst_type in np.unique(auto_train_df['inst_type'].values):
            type_df = auto_train_df.loc[auto_train_df['inst_type'] == inst_type]
            schedule_type = type_df['schedule_type'].values[0]
            hour = type_df['schedule_hour'].values[0]
            minute = type_df['schedule_minute'].values[0]
            if schedule_type == "EVERY_DATE":
                # every day
                schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self.producer.auto_train_produce,
                                                                       type_df['target_id'].values)
            elif schedule_type == "EVERY_WEEK":
                # every week
                day = type_df['schedule_day'].values[0]
                self.schedule_every_week(day, hour, minute, type_df['target_id'].values)
            elif schedule_type == "EVERY_MONTH":
                # every month
                date = type_df['schedule_date'].values[0]
                schedule.every().month.at(f"{hour:02d}:{minute:02d}").day.at(f"{date}").do(self.producer.auto_train_produce,
                                                                                           type_df['target_id'].values)

        for job in schedule.get_jobs():
            print(f"train schedule : {job}, next run : {job.next_run}")

        while True:
            schedule.run_pending()
            time.sleep(1)


def main():
    auto_train_scheduler = AutoTrainScheduler()
    auto_train_scheduler.train_scheduling()


if __name__ == "__main__":
    main()
