from common.error_code import Errors
import logging
import sys


class ModuleException(Exception):

    def __init__(self, exception_code):
        self.exception_code = exception_code

        self.error_code = Errors[exception_code].value
        self.error_msg = Errors[exception_code].desc
        # self.f_name = sys._getframe().f_back.f_code.co_filename

class ModuleAnalyzer():

    def serving(self):

        # try:
        #     2 /0
        # except ZeroDivisionError as e:
        #     raise ModuleException("E705")

        ma = ModuleAlgorithms()

        logging.warning("analyzer any top logic ")

        algo1 = None
        algo2 = None

        try:
            algo1 = ma.predict()
        except ModuleException as me:
            logging.exception(me.error_msg)
        except Exception as e:
            logging.exception(f"Module Exception Cause : {e}")

        try:
            algo2 = ma.predict2()
        except ModuleException as me:
            logging.exception(me.error_msg)
        except Exception as e:
            logging.exception(f"Module Exception Cause : {e}")

        algo1_dict = {}
        algo2_dict = {}

        if algo1 is not None:
            algo1_dict["result"] = "success"
        else:
            algo1_dict["result"] = "fail"

        if algo2 is not None:
            algo2_dict["result"] = "success"
        else:
            algo2_dict["result"] = "fail"

        res = {
            "performance" : algo1_dict,
            "performance2" : algo2_dict
        }

        logging.warning("analyzer any bottom logic")
        return res

class ModuleAlgorithms():

    def predict(self):

        try :
            2 / 0 # cause Exception , try except
        except ZeroDivisionError as me: # developer catch 1
            raise ModuleException("E705")
        except KeyError as ke: # developer catch 2
            raise ModuleException("E900") # 해당되는 error_code 생성 및 적용
        # ....

        logging.warning("predict1 any predict")

        return "predict_success"

    def predict2(self):

        logging.warning("predict2 any predict")

        return "predict_success"


if __name__ == "__main__":
    # print(Errors.E707.value)
    ma = ModuleAnalyzer()

    logging.warning("Start Serving")

    try:
        _body = ma.serving()

    except MemoryError as me:
        logging.exception(f"[Error] Unexpected memory error during serving : {me}")

    except ModuleException as me:
        logging.exception(f"[Error] Unexpected ModuleException during serving cause: {me.error_msg}")

    except Exception as e:
        logging.exception(f"[Error] Unexpected exception during serving : {e}")

    logging.warning("End Serving")
    logging.warning(f"Serving Response {_body}")
