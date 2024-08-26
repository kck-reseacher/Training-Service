from analyzer import aimodule
from analyzer.exem_aiops_wclst_db import ExemAiopsWclstDb
from analyzer.exem_aiops_wclst_was import ExemAiopsWclstWas
from common import constants


class ExemAiopsWclst(aimodule.AIModule):
    def __init__(self, config, logger):
        """
        initialize instance attributes using JSON config information and Confluence documentation. AI Server inputs the JSON config information.
        :param config:
        :param logger:
        """
        self.config = config
        self.logger = logger

        if config["inst_type"] == constants.INST_TYPE_WAS:
            self.workload_cluster = ExemAiopsWclstWas(self.config, self.logger)
        elif config["inst_type"] == constants.INST_TYPE_DB:
            self.workload_cluster = ExemAiopsWclstDb(self.config, self.logger)

    def init_train(self):
        pass

    def train(self, stat_logger):

        header, body, errno, errmsg = self.workload_cluster.train(stat_logger)

        return header, body, errno, errmsg

    def end_train(self):
        pass

    def serve(self, serving_df, anls_df, centers_for_serving):
        pass
