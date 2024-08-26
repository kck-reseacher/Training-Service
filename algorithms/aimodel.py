class AIModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def load(self, dir):
        pass

    def save(self, dir):
        pass

    def estimate(self, input):
        pass

    def pre_model_preprocessing(self, x_test_data, y_test_data=None, feat=None):
        return x_test_data, y_test_data


class TrainProgress:
    def __init__(self, progress_names, logger):
        self.logger = logger
        self.progress_entrys = {}
        for name in progress_names:
            self.progress_entrys[name] = 0

    def log(self, progress_name, percent=0):

        if progress_name in self.progress_entrys:
            self.progress_entrys[progress_name] = int(percent)

        if self.logger != None:
            self.logger.critical(self.progress_entrys)
