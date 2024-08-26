import os
import onnx
import tf2onnx
import sklearn
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras.models import load_model

class ONNX:

    @staticmethod
    def onnx_save(obj, path, input_signature=None):
        """
            obj : self.models[feat] 알고리즘 인스턴스의 개별 피처 모델 object
            path : {model_dir} / {model_name}.onnx
        """

        if isinstance(obj, tf.keras.Model):
            onnx_model, _ = tf2onnx.convert.from_keras(obj, input_signature=input_signature)
        elif isinstance(obj, tf.Graph):
            return
        elif isinstance(obj, sklearn.base.BaseEstimator):
            return
        elif isinstance(obj, "pickle"):
            return
        else:
            raise Exception("onnx 로 저장할 수 없는 모델입니다.")

        with open(path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
            return onnx_model

    @staticmethod
    def onnx_load(onnx_model_path, h5_model_path=None):
        """
            내부적으로 convert 확인해준다.
            path : {model_dir} / {model_name}.onnx
            name.h5
        """
        def convert_onnx():
            """
                path : 모델이름.h5, -> onnx 모델로 변경 후 저장한다.
                    기존의 h5 모델은 삭제.
            """

            tf_model = load_model(h5_model_path)
            onnx_model, _ = tf2onnx.convert.from_keras(tf_model)
            with open(onnx_model_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())

            return None

        model = None
        try:
            if not os.path.exists(onnx_model_path):  # onnx 모델이 없을 때
                convert_onnx()
        except Exception as e:
            pass

        sess_opt = ort.SessionOptions()
        sess_opt.enable_mem_pattern = False
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.intra_op_num_threads = 1
        onnx_model = onnx.load(onnx_model_path)
        model = ort.InferenceSession(onnx_model.SerializeToString(), sess_options=sess_opt)
        return model

if __name__ == "__main__":
    onnx_path = "path_dir/model.onnx"
    h5_path = "path_dir/model.h5"

    a = ONNX
    a.onnx_load(onnx_path)