import math
import os
import re
import time
from datetime import datetime

import joblib
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathos.multiprocessing import cpu_count, Pool

from common import constants, aicommon


class Log2Template:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.progress_name = constants.MODEL_B_LOGSEQ

        drain_config = TemplateMinerConfig()
        drain_config.load('./algorithms/logseq/config/drain3.ini')
        self.template_miner = TemplateMiner(config=drain_config)
        self.mined_period = None#{'from': '', 'to': ''}
        self.n_templates = None

        self.template_dict = {}

        self.template2vec = None

        self.do_multiprocessing = True
        self.n_proc = math.ceil(cpu_count() * 0.2)

        self.sim_thres = 0.8

    def regex_filtering(self, lines):
        # datetime
        # lines = list(map(lambda x: re.sub(r"\d{4}\-\d{2}\-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", '<DATETIME>', x), lines))
        # lines = list(map(lambda x: re.sub(r"\[\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}\]", '<DATETIME> ', x), lines))
        # lines = list(map(lambda x: re.sub(r"\s{2,}", ' ', x), lines))
        # lines = list(map(lambda x: x.strip(), lines))
        # self.logger.info('[Log2Template] regex_filtering - Datetime filtered')

        lines = list(map(lambda x: re.sub(r"(?<=statement )\[.+\]", '<SQL-STATEMENT>', x), lines))
        self.logger.info('[Log2Template] regex_filtering - SQL statement filtered')

        lines = list(map(lambda x: re.sub(r"at .*", '<ERROR-Traceback>', x), lines))
        lines = list(map(lambda x: re.sub(r"Caused by: .*", '<ERROR-Traceback>', x), lines))
        lines = list(map(lambda x: re.sub(r"\.{3} \d+ more", '<ERROR-Traceback>', x), lines))
        lines = list(map(lambda x: re.sub(r"\s{2,}", ' ', x), lines))
        self.logger.info('[Log2Template] regex_filtering - Error Traceback filtered')

        return lines

    def log2tidx(self, log_df, fitting=False, train_progress=None):
        filtered_lines = self.fit(log_df, train_progress) if fitting else self.regex_filtering(log_df['msg'].values)

        self.logger.info(f"[Log2Template] N_lines to transform = {len(filtered_lines)}")

        time_s = time.time()
        if self.do_multiprocessing and log_df.shape[0] > 500:
            self.logger.info(f"[Log2Template] n_proc for log2tidx() multiprocessing = {self.n_proc}(out of {cpu_count()})")

            # transform via multiprocessing
            pool = Pool(self.n_proc)
            tidxs = pool.map(lambda x: self.transform(x, fitting), filtered_lines)
            pool.close()
            pool.join()
        else:
            tidxs = list(map(lambda x: self.transform(x, fitting), filtered_lines))

        if fitting:
            log_df['tidx'] = tidxs
        else:
            tidxs = np.array(tidxs)
            log_df['tidx_inp'], log_df['tidx_cmp'] = tidxs[:, 0], tidxs[:, 1]  # tidx_inp : input data for model, tidx_cmp : real data for comparison('real' <=> pred)
        log_df.reset_index(inplace=True, drop=True)

        time_e = time.time()
        self.logger.info(f"[Log2Template] transform end (elapsed = {time_e - time_s:.3f}s)")
        return log_df

    def fit(self, log_df, train_progress):
        self.logger.info(f"[Log2Template] regex_filtering start")
        filtered_lines = self.regex_filtering(log_df['msg'].values)
        self.logger.info(f"[Log2Template] regex_filtering end")

        # template_miner
        # if self.load_template_miner(self.config['model_dir']):
        #     self.logger.info(f"[Log2Template] incremental mining")
        #     lines_to_mine = self.regex_filtering(self.preprocess_mining_period(log_df))
        # else:
        self.mined_period = {'from': self.config['date'][0], 'to': self.config['date'][-1]}
        lines_to_mine = filtered_lines#log_df['msg'].values

        self.logger.info(f"[Log2Template] N_lines to mine = {len(lines_to_mine)}")
        if len(lines_to_mine) == 0:
            self.logger.info(f"[Log2Template] No lines to mine. => skip mining")
        else:
            # _ = map(lambda line: self.template_miner.add_log_message(line.strip()), filtered_lines)
            self.logger.info(f"[Log2Template] Mining start")
            time_mining_s = time.time()
            for i, line in enumerate(lines_to_mine):
                self.template_miner.add_log_message(line.strip())
                if i > 0 and i % (len(lines_to_mine) // 20) == 0:
                    self.logger.info(f"[Log2Template] Approx. {int(i/len(lines_to_mine) * 100):>3}%({i:{len(str(len(lines_to_mine)))}} out of {len(lines_to_mine)}) mined")
                    aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress,
                                                                       self.progress_name, int(i/len(lines_to_mine) * 30))  # 0 ~ 30%
            self.logger.info(f"[Log2Template] Mining end (elapsed = {time.time() - time_mining_s:.3f}s)")
            aicommon.Query.update_module_status_by_training_id(self.config['db_conn_str'], self.config['train_history_id'], train_progress, self.progress_name, 30)

        self.n_templates = len(self.template_miner.drain.clusters)
        self.logger.info(f"[Log2Template] Mining finished => n_templates = {self.n_templates}")

        self.logger.debug("="*20 + f" SORTED CLUSTERS " + "="*20)
        # sorted_clusters = sorted(self.template_miner.drain.clusters, key=lambda c: c.size, reverse=True)
        for cluster in self.template_miner.drain.clusters:
            self.logger.debug(cluster)
        self.logger.debug(f"=" * 50)

        # template2vec
        self.logger.info(f"[Log2Template] creating Template2Vec model")
        self.make_template2vec()
        self.logger.info(f"[Log2Template] Template2Vec model created")

        return filtered_lines

    def make_template2vec(self):
        tokenized_templates = [list(filter(lambda x: re.search('[a-zA-Z가-힣]', x) is not None, cluster.get_template().split())) for cluster in self.template_miner.drain.clusters]
        tagged_sents = [TaggedDocument(s, [i+1]) for i, s in enumerate(tokenized_templates)]
        self.template2vec = Doc2Vec(tagged_sents, vector_size=100, window=2, min_count=1, epochs=500)

    def transform(self, filtered_line, fitting=True):
        matched_template = self.template_miner.match(filtered_line)
        tidx = self.get_most_similar_template(filtered_line, fitting) if matched_template is None else matched_template.cluster_id if fitting else np.array([matched_template.cluster_id, matched_template.cluster_id])
        # template_index range (1 ~ n) => softmax_index range (0 ~ n-1)
        return tidx - 1

    def get_most_similar_template(self, line, fitting):
        # self.logger.info(f"[Log2Template] get_most_similar_template - line : {line}")
        sent_template = self.template_miner.drain.get_content_as_tokens(line)
        sent_filtered = list(filter(lambda x: re.search('[a-zA-Z가-힣]', x) is not None, sent_template))
        sent_vec = self.template2vec.infer_vector(sent_filtered)
        most_similar = self.template2vec.docvecs.most_similar(restrict_vocab=50, positive=[sent_vec])[0]  # format : (tidx, proba)
        '''
        return value format example
        * fitting => tidx
        * not fitting
            if similarity >= thres => (tidx, tidx) # ok to use similar template
            else => (tidx, 0) : 0 = dummy tidx for raising anomaly
        '''
        return most_similar[0] if fitting else np.array([most_similar[0], most_similar[0] if most_similar[1] >= self.sim_thres else 0])

    def get_normal_idxs(self):
        thres = self.template_miner.drain.get_total_cluster_size() * 0.01  # 1%
        return [tidx-1 for tidx, templ in sorted(self.template_miner.drain.id_to_cluster.items(), key=lambda x: x[1].size, reverse=True) if templ.size > thres]

    def tidx2template(self, pred):
        return [self.template_dict[str(int(c_idx)+1)].get_template() for c_idx in pred]

    def save(self, model_dir):
        miner_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/template_miner.pkl")
        joblib.dump(self.template_miner, miner_path)
        self.logger.info(f"[Log2Template] template_miner saved (1/4)")

        if not os.path.exists(os.path.join(model_dir, f"{constants.MODEL_S_SPARSELOG}")):
            os.makedirs(os.path.join(model_dir, f"{constants.MODEL_S_SPARSELOG}"))
        sparse_miner_path = os.path.join(model_dir, f"{constants.MODEL_S_SPARSELOG}/template_miner.pkl")
        joblib.dump(self.template_miner, sparse_miner_path)
        self.logger.info(f"[Log2Template] sparselog template_miner saved (1/4)")

        template2vec_path = os.path.join(model_dir, f"{constants.MODEL_S_LOGSEQ}/template2vec.model")
        self.template2vec.save(template2vec_path)
        self.logger.info(f"[Log2Template] template2vec saved (2/4)")

    def load(self, model_dir):
        pass