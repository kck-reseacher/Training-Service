from analyzer import aimodule
import pandas as pd
from analyzer.exem_aiops_event_fcst_clf import ExemAiopsEventFcstClf
from common.module_exception import ModuleException


class ExemAiopsEventFcst(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        initialize instance attributes using JSON config information and Confluence documentation. AI Server inputs the JSON config information.
        :param config:
        :param logger:
        """
        self.config = config
        self.logger = logger
        self.saved_df = None
        self.event_fcst_clf = ExemAiopsEventFcstClf(self.config, self.logger)

    def init_train(self):
        try:
            self.event_fcst_clf.init_train()
        except KeyError as ke:
            raise ModuleException("E830")

    def train(self, stat_logger):
        train_prog = {}
        header, body, errno, errmsg = self.event_fcst_clf.train(stat_logger, train_prog)
        
        return header, body, errno, errmsg

    def end_train(self):
        pass

    def init_serve(self, reload=False):
        pass

    def serve(self, header, body):
        pass

    def preprocessing(self, header, body):
        last_time = pd.to_datetime(body[0]['time'], format='%Y-%m-%d %H:%M:%S')

        input_df = pd.DataFrame(body)
        input_df['time'] = pd.to_datetime(input_df['time'], format='%Y-%m-%d %H:%M:%S')

        if self.saved_df is not None:
            input_df = pd.concat([input_df, self.saved_df])

        time_filter2 = self._time_filter(input_df['time'], last_time, timedelta=5)
        self.saved_df = input_df.loc[time_filter2].drop_duplicates().sort_values(by='time', ascending=False, ignore_index=True)

        time_filter = self._time_filter(self.saved_df['time'], last_time)
        return self.saved_df.loc[time_filter].sort_values(by='time', ascending=False, ignore_index=True)

    @staticmethod
    def _time_filter(input_data, last_time, timedelta=0):
        '''
        Parameters
        ----------
        input_data: Timestamp type의 데이터
        last_time: header의 predict_time이나 standard_time
        timedelta: buffer 길이(min)

        Returns
        -------
        '''
        first_time = last_time - pd.Timedelta(minutes=(60 + timedelta))
        last_time = last_time + pd.Timedelta(minutes=timedelta)
        time_filter = (input_data > first_time) & (input_data <= last_time)
        return time_filter
