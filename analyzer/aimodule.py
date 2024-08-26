import json
import logging
import os
import time

import pandas as pd

from pathlib import Path
from common import aicommon, constants
from common.constants import SystemConstants as sc
from common.system_util import SystemUtil
from resources.logger_manager import Logger
from resources.config_manager import Config


class AIModule:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.name = __name__

    # API for training
    def init_train(self):
        pass

    def train(self, stat_logger):
        pass

    def test_train(self):
        pass

    def end_train(self):
        pass

    # API for serving
    def init_serve(self, reload):
        pass

    def serve(self, input_df):
        pass

    def end_serve(self):
        pass

    def estimate(self, serving_date: list = None, input_df: pd.DataFrame = None, sbiz_df: pd.DataFrame = None):
        pass

    # API for dev and debug
    def get_debug_info(self):
        pass

    """
    def split_file_by_day(self, source_file_path, source_file_name, target_file_path, inst_id):
        df = pd.read_csv(os.path.join(source_file_path, source_file_name))
        df['time_ymd'] = pd.to_datetime(df['time']).dt.strftime('%Y%m%d')

        for time in df['time_ymd'].unique():
            df.loc[df['time_ymd'] == time].drop(columns=['time_ymd']).to_csv(os.path.join(target_file_path, inst_id + "_" + time + '.csv'), index=False)
    """

    def get_csvfile_list(self, root_dir, prefix, from_date, to_date):
        date_list = pd.date_range(from_date, to_date, freq="D")
        file_list = list(
            map(
                lambda x: os.path.join(
                    root_dir, f"{prefix}_{x.strftime('%Y%m%d')}.csv"
                ),
                date_list,
            )
        )
        return file_list

    def get_business_csvfile_list(self, root_dir, prefix, date_list):
        file_list = list(
            map(
                lambda x: os.path.join(
                    root_dir, f"{prefix}_{x}.csv"
                ),
                date_list,
            )
        )
        return file_list

    '''
    학습 데이터 구조 변경용 함수
    '''
    def get_csvfile_list_all(self, root_dir, from_date, to_date):
        date_list = pd.date_range(from_date, to_date, freq="D")
        file_list = list(
            map(
                lambda x: os.path.join(
                    root_dir, f"{x.strftime('%Y%m%d')}.csv"
                ),
                date_list,
            )
        )
        return file_list

    def get_business_csvfile_list_all(self, root_dir, date_list):
        file_list = list(
            map(
                lambda x: os.path.join(
                    root_dir, f"{x}.csv"
                ),
                date_list,
            )
        )
        return file_list

    def load_csvfiles(self, fpath_list, log_data=False):
        dfs = []
        for fpath in fpath_list:
            try:
                if log_data:
                    df = pd.read_csv(fpath, quotechar='"')
                else:
                    df = pd.read_csv(fpath)
                if not df.empty:
                    dfs.append(df)
                else:
                    self.logger.debug(f"warning : no data in file {fpath}")
            except Exception as e:
                self.logger.warn(f"fail to load file {fpath} caused by {e}")
            else:
                self.logger.debug(f"load file {fpath}")
            finally:
                pass

        if len(dfs) == 1:
            return dfs[0]
        elif len(dfs) > 1:
            return pd.concat(dfs)
        else:
            # len(dfs)==0
            self.logger.info("No data found. Please check the data and range.")

        return dfs

    @staticmethod
    def check_multi_logger(target_id, modelMap):
        return target_id not in modelMap or modelMap[target_id].get("logger", None) is None

    @staticmethod
    def create_multi_logger(logger_dir, logger_name, sys_id, module_name):
        os_env = SystemUtil.get_environment_variable()
        py_config = Config(os_env[sc.MLOPS_TRAINING_PATH], os_env[sc.AIMODULE_SERVER_ENV]).get_config()

        # module_error_dir
        error_log_dict = dict()
        error_log_dict["log_dir"] = str(
            Path(os_env[sc.MLOPS_LOG_PATH]) / "train" / str(sys_id) / sc.ERROR_LOG_DEFAULT_PATH / sc.SERVING
        )
        error_log_dict["file_name"] = module_name

        logger = Logger().get_default_logger(
            logdir=logger_dir, service_name=logger_name, error_log_dict=error_log_dict,
        )

        if py_config["use_integration_log"]:
            module = module_name.rsplit("_", 1)[0]
            inst_type = module_name.rsplit("_", 1)[1]
            integration_log_dir = str(Path(os_env[sc.MLOPS_LOG_PATH]) / "integration" / str(sys_id) / module / inst_type)
            integration_error_log_dict = {
                "log_dir": str(Path(os_env[sc.MLOPS_LOG_PATH]) / "integration" / str(sys_id) / sc.ERROR_LOG_DEFAULT_PATH / sc.SERVING),
                "file_name": module_name,
            }
            Logger().get_default_logger(
                logdir=integration_log_dir, service_name=f"integration_{logger_name}",
                error_log_dict=integration_error_log_dict,
            )

        return logger

    @staticmethod
    def default_module_status():
        return {constants.PROGRESS: 0, constants.DURATION_TIME: None}


class CommandLineUtil:
    def __init__(self, module_class, json_config, log_level):

        self.module_class = module_class

        self.config = json.loads(json_config)

        if self.config.get("temp_dir", None) is None:
            self.config["temp_dir"] = "./temp"

        log_dir = self.config.get("log_dir", None)
        if log_dir is None:
            log_dir = os.path.join(".", "logs", module_class.__name__)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d%H%M%S')}.log")
        fileHandler = logging.FileHandler(log_file)
        streamHandler = logging.StreamHandler()

        formatter = logging.Formatter(
            "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s"
        )
        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self.logger = logging.getLogger("module_logger")
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(streamHandler)
        self.logger.setLevel(log_level)

    def runCommand(self, mode, run_info):
        self.module = self.module_class(self.config, self.logger)

        if mode == "train":
            self.logger.critical(
                f"[cmdutil] start train {self.module.module_id}: {self.config['from_date']}-{self.config['to_date']}, \nconfig: {self.config}"
            )
            elapse = time.time()
            self.do_train()
            self.logger.critical(
                f"[cmdutil] end train {self.module.module_id} (elapse {time.time() - elapse}s)"
            )
        elif mode == "serve":
            self.logger.critical(
                f"[cmdutil] start serve {self.module.module_id}: {run_info['from_date']}-{run_info['to_date']}, \nconfig: {self.config}"
            )
            elapse = time.time()
            self.do_serve(run_info)
            self.logger.critical(
                f"[cmdutil] end serve {self.module.module_id} (elapse {time.time() - elapse}s)"
            )
        """
        아직 미사용 하고 있기 때문에 주석 처리 함.
        이후 그래프를 그리는 로직을 정비 한뒤 넣음
        """
        # elif mode == 'test':
        #     self.logger.critical('[cmdutil] start test {}: {}-{}, [{}, {}, {}]\nconfig: {}'.format(
        #         self.module.module_id, self.config['from_date'], self.config['to_date'], opt1, opt2, opt3, self.config))
        #     elapse = time.time()
        #     self.do_test()
        #     self.logger.critical('[cmdutil] end test {} (elapse {}s)'.format(self.module.module_id, (time.time()-elapse)))
        # else:
        #     self.do_user_opt(opt, opt1, opt2, opt3)

    def set_config(self, config):
        self.config = config

    def do_train(self):
        result, data, errno, errmsg = self.module.train(self.logger)
        param = self.module.config

        param["errno"] = errno
        if errmsg is not None:
            param["errmsg"] = errmsg

        if result is not None:
            param["results"] = result

        if data is not None:
            param["body"] = data

        # result file save
        with open(
            os.path.join(param["model_dir"], "model_config.json"), "w"
        ) as outfile:
            json.dump(param, outfile, cls=aicommon.JsonEncoder, ensure_ascii=False)

        # result file save
        with open(
            os.path.join(param["train_dir"], "train_result.json"), "w"
        ) as outfile:
            json.dump(param, outfile, cls=aicommon.JsonEncoder, ensure_ascii=False)

    # def do_test(self, opt_plot, plot_start_ratio, plot_len_ratio):
    #     if not self.config["use_forecast"]:
    #         return None
    #     self.module.init_serving()
    #     result, real, pred, interval_l, interval_u = self.module.test_training()

    #     if opt_plot == 'plot' or opt_plot == 'plotfile':
    #         colors = ['green', 'yellow', 'red', 'cyan', 'magenta', 'pink', 'black']
    #         alphas = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    #         linewdth = 0.2
    #         labels = []
    #         cols = self.module.fcst.y_cols
    #         col_cnt = len(cols)
    #         y_time_cnt = len(self.module.fcst.y_index)

    #         if plot_start_ratio is None:
    #             plot_start = 0
    #         else:
    #             plot_start = int(len(real)*float(plot_start_ratio))

    #         if plot_len_ratio is None:
    #             plot_len = len(real)
    #         else:
    #             plot_len = int(len(real)*float(plot_len_ratio))

    #         for i in range(y_time_cnt):
    #             labels.append('predict +{}'.format(self.module.fcst.y_index[i]))

    #         upper = pred + interval_u
    #         lower = pred + interval_l
    #         upper = upper - lower
    #         lower[lower < 0] = 0

    #         if opt_plot == 'plot':
    #             fig = plt.figure(figsize=(20, 2*col_cnt))
    #         else:
    #             fig = plt.figure(figsize=(30, 4*col_cnt))

    #         for i in range(col_cnt):
    #             ax = fig.add_subplot(col_cnt, 1, i+1)
    #             if i == col_cnt-1:
    #                 ax.set_xlabel("time")
    #             ax.set_ylabel(cols[i])
    #             _ = ax.plot(real[plot_start:plot_start+plot_len, [i]], color='b', linewidth=0.5, label='real')
    #             for j in range(y_time_cnt):
    #                 plot_data = np.array([0] * (self.module.fcst.y_index[j] - self.module.fcst.y_index[0]))
    #                 plot_data = np.append(plot_data, pred[plot_start:plot_start+plot_len, [(j*col_cnt)+i]])
    #                 _ = ax.plot(plot_data, color=colors[j], linewidth=linewdth, label=labels[j])

    #                 #if j == y_time_cnt-1:
    #                 if True:
    #                     plot_upper = np.array([0] * (self.module.fcst.y_index[j] - self.module.fcst.y_index[0]))
    #                     plot_upper = np.append(plot_upper, upper[plot_start:plot_start+plot_len, [(j*col_cnt)+i]])
    #                     plot_lower = np.array([0] * (self.module.fcst.y_index[j] - self.module.fcst.y_index[0]))
    #                     plot_lower = np.append(plot_lower, lower[plot_start:plot_start+plot_len, [(j*col_cnt)+i]])

    #                     ax.stackplot(range(len(plot_data)), np.vstack([plot_lower, plot_upper]),
    #                                  colors=['#FFFFFF', colors[j]], alpha=alphas[j])

    #             if i == 0:
    #                 ax.legend()

    #         if opt_plot == 'plot':
    #             plt.show()
    #         else:
    #             fname = './plot/fcst_{}_{}-{}_{}.png'.format(self.config['target_id'],
    #                                                          self.config['from_date'],
    #                                                          self.config['to_date'],
    #                                                          time.strftime('%Y%m%d%H%M%S'))
    #             plt.savefig(fname)

    def do_serve(self, run_info):

        # target_id = self.config['target_id']
        # from_date = self.config['from_date']
        # to_date = self.config['to_date']

        # self.module.init_serving()

        # if user_df_path is None:
        #     path_list = self.module.get_csvfile_list(os.path.join(self.config['train_dir'], 'was'),
        #                                              self.config['from_date'],
        #                                              self.config['to_date'])
        #     df_was = self.module.load_csvfiles(path_list)

        #     for i in range(len(df_was)):
        #         dict_was = pd.DataFrame(df_was.iloc[i]).to_dict()
        #         input_dict = {}
        #         input_dict['was'] = dict_was[i]
        #         self.logger.debug(input_dict)
        #         result, data, errno, errmsg = self.module.serving(None, input_dict)
        #         self.logger.info(data)
        #         time.sleep(0.1)
        # else:
        #     with open(user_df_path) as jsn_file:
        #         jsn_data = json.load(jsn_file)
        #     self.logger.critical(jsn_data)
        #     # result, pred, errno, errmsg = self.module.serving(None, jsn_data)
        #     # self.logger.critical(pred)
        #     self.module.serving(None, jsn_data)
        pass

    def do_user_opt(self, opt, opt1, opt2, opt3):
        print(f"invalid opt {opt} [{opt1}, {opt2}, {opt3}]")
