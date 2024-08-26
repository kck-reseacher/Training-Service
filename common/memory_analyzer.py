import os
import gc
import random
import time

import psutil
from ctypes import cdll, CDLL


class MemoryUtil:
    def __init__(self, logger):
        self.logger = logger

    def prepare_gc(self):
        if not gc.get_debug():
            gc.set_debug(True)
        if gc.isenabled():
            gc.disable()

    def gc_enable(v):
        if not gc.isenabled():
            gc.enable()

    def gc_disable(self):
        if gc.isenabled():
            gc.disable()

    def get_objs(self, gen):
        return gc.get_objects(gen)

    def get_referents(self, obj):
        return gc.get_referents(self, obj)

    def get_referrers(self, obj):
        return gc.get_referrers(self, obj)

    def get_garbage(self):
        garbage = gc.garbage
        self.logger.debug(f"------- gc garbage: {garbage}... --------")
        return garbage

    """
    With no arguments, run a full collection
    """
    def gc_collect(self, gen=None, var=None):
        self.logger.info("------- gc collectiong... --------")
        if gc.isenabled():
            gc.disable()

        if gen is None:
            if var is None:
                gc.collect()
            else:
                del var
                gc.collect()
        else:
            if var is None:
                gc.collect(generation=gen)
            else:
                del var
                gc.collect(generation=gen)
        self.logger.info("------- gc collect done --------\n")

    def gc_info(self):
        self.logger.debug(f">>> gc automatic enable : {gc.isenabled()} <<<")
        self.logger.debug(f">>> gc gen stats : \n{gc.get_stats()} <<<")
        self.logger.debug(f">>> gc reference count: {gc.get_count()} <<<")
        self.logger.debug(f">>> gc threshold: {gc.get_threshold()} <<<\n")

    def tear_down(self):
        gc.set_debug(False)
        if not gc.isenabled():
            self.logger.info("-------- gc automatic enabling... --------")
            gc.enable()
            self.logger.info("-------- gc automatic done --------\n")

    def print_memory(self):
        memory_usage_dict = dict(psutil.virtual_memory()._asdict())
        memory_usage_percent = memory_usage_dict['percent']
        self.logger.info(f"########### memory_usage_percent: {memory_usage_percent}% ##########")

        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
        self.logger.info(f"########### pid {pid}'s current memory alloc size(KB) : "
                    f"{current_process_memory_usage_as_KB: 9.3f} KB ##########\n")

    # def get_var_alloc_size(self, var):
    #     size = asizeof.asizeof(var)
    #     self.logger.info(f"########### variable memory alloc size : {size}B ##########")
    #     return size

    def remove_references(self, lists):
        try:
            for list in lists:
                list = None
                del list
        except Exception as e:
            self.logger.error(e)

class MemoryManager:
    def __init__(self, logger):
        self.logger = logger

    def trim_memory(self):
        try:
            cdll.LoadLibrary("libc.so.6")
            libc = CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception as ignore:
            libc = None


if __name__ == "__main__":
    from resources.logger_manager import Logger
    from common.constants import SystemConstants as sc
    from pathlib import Path

    # GET AIMODULE_HOME
    home = os.environ.get(sc.AIMODULE_HOME)
    if home is None:
        home = os.path.dirname(os.path.abspath(__file__))

    # SET LOG DIR
    log_dir = str(
        Path(home) / "logs" / "memory_usage_logs"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # SET LOGGER
    logger = Logger().get_default_logger(logdir=log_dir, service_name=__name__)

    a_list = [1, 2, 3, 4, 5]
    a_dic = {'a': [1, 2, 3, 4, 5]}
    b = None
    c = None
    d = None
    some_list = []
    for i in range(1000):
        prefix = str(random.randint)
        prefix_var = i
        some_list.append(prefix_var)
    some_list = None

    time.sleep(3)



