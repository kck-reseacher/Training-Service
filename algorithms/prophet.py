import time
import pandas as pd
from prophet import Prophet

class akprophet:
    def __init__(self, config, logger):
        """장기부하예측 Prophet 모델

        Parameters
        Args:
            config (_type_): train.py의 param
            logger (_type_): logger
        """

        self.config = config
        self.logger = logger

        self.target_id = config['target_id']
        self.predict_month = config['predict_month']
        self.predict_day_range = 60
        self.models = {}
        self.res = {}
        self.training_result = {
            'predict': {
                'from_date': self.config['from_date'],
                'to_date': self.config['to_date'],
                'except_failure_date_list': self.config['except_failure_date_list'],
                'except_business_list': self.config['except_business_list'],
                'business_list': self.config['business_list'],
                'train_metrics': {},
            }
        }

    def fit(self, feat, feat_df):
        feature_training_start = time.time()

        feat_df = feat_df.rename({'time' : 'ds', feat : 'y'}, axis='columns')
        self.calculate_predict_day_range(feat_df)

        feat_df['cap'] = feat_df['y'].max()
        feat_df['floor'] = feat_df['y'].min()

        if feat_df['y'].var() <= 0.05:
            feat_df['cap'] += 0.00001
            feat_df['floor'] -= 0.00001

        self.res[feat] = feat_df
        self.models[feat] = Prophet(changepoint_range=0.95, interval_width=0.95,
                                    daily_seasonality='auto', weekly_seasonality='auto', yearly_seasonality='auto', growth='logistic')

        self.logger.info(f"{feat}")
        # fit
        self.models[feat].fit(feat_df)
        feature_elapse = time.time() - feature_training_start
        self.training_result['predict']['train_metrics'][feat] = {
            'duration_time': int(1000 * feature_elapse),  # ex. 709ms
        }

    def predict(self):
        result = []
        self.logger.info(f"feats = {self.models.keys()}")

        for feat in self.models.keys():
            future = self.models[feat].make_future_dataframe(periods=self.predict_day_range, freq='D')
            future['cap'] = self.res[feat]['cap'][0]
            future['floor'] = self.res[feat]['floor'][0]
            future = future[future['ds'] > pd.to_datetime(self.training_result['predict']['to_date'])]

            forecast = self.models[feat].predict(future)
            feat_result = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]

            self.logger.info(f"\nbefore renaming :\n {feat_result}\n")
            feat_result = feat_result.rename(columns={'yhat': f"{feat}_pred", 'yhat_lower': f"{feat}_lower", 'yhat_upper': f"{feat}_upper"})
            self.logger.info(f"\nafter renaming :\n {feat_result}\n")

            result.append(feat_result)
            self.logger.info(f"\nserving result :\n {pd.concat(result, axis=1)}\n")

        return pd.concat(result, axis=1)

    def calculate_predict_day_range(self, data):
        """
        데이터 양에 따라서 예측 기간을 정하는 함수
        만약 사용자가 예측 기간을 수동으로 정하면 해당 기간에 맞춰 예측을 하고, 그렇지 않으면 데이터 양에 따라서 예측 기간이 달라진다.
        1 ~ 3개월 -> 1개월, 3 ~ 6개월 -> 3개월, 6 ~ 12개월 -> 6개월, 12개월 이상이면 12개월 예측하게 개발
        :param data:
        :return:
        """
        self.logger.info("[calculate_predict_day_range] start calculating day range")
        self.logger.info(f"[calculate_predict_day_range] self.predict_month : {self.predict_month}")

        if self.predict_month in (1, 3, 6, 12):
            self.logger.info(f"[calculate_predict_day_range] user requested {self.predict_month} month prediction")
            self.predict_day_range = (30 * self.predict_month) + (5 if self.predict_month == 12 else 0)
        elif self.predict_month == 0:
            number_of_rows = len(data)
            self.logger.info("[calculate_predict_day_range] automatically default prediction range")
            self.logger.info(f"[calculate_predict_day_range] length of dataframe is {number_of_rows}")

            if number_of_rows < 30:
                raise ValueError("Not Enough Training Data!")
            self.predict_day_range = 90 if number_of_rows < 90 else 180 if number_of_rows < 180 else 365
        self.logger.info(f"[calculate_predict_day_range] predict day range is {self.predict_day_range} days")
        self.logger.info("[calculate_predict_day_range] finish calculating day range")
