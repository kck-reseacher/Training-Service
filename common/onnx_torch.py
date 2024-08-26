import os
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


class ONNXTorch:
    @staticmethod
    def onnx_save(obj, dummy_input, path, input_names="input", output_names="output", opset_version=10, model_type='b'):
        """
            obj : nn.Module type의 torch 모델
            dummy_input : 모델 입력 shape 맞는 더미 값(randn, empty 등 자유롭게 이용 가능)
                ex) torch.randn(1, self.window_size, len(columns_order)).to(self.device)
            path : {model_dir} / {model_name}.onnx
            input_names : 요소로 str 포함하는 list, 입력의 이름은 자유롭게 설정, 개 수는 맞추어야 함.
                ex) 모델 입력이 1 개읜 경우 ["input"]
                ex) 모델 입력이 2 개읜 경우 ["input_1", "input_2"]
            output_names :
            opset_version : 모델에 맞는 opset
        """

        if model_type == 'b':
            if isinstance(obj, nn.Module):
                obj.eval()
                torch.onnx.export(obj, dummy_input, path, input_names=input_names, output_names=output_names, opset_version=opset_version)
            else:
                raise Exception("torch 모델이 아닙니다.")
        elif model_type == 'a':
            if isinstance(obj, nn.Module):
                obj.eval()
                torch.onnx.export(obj, dummy_input, path, opset_version=opset_version)
            else:
                raise Exception("torch 모델이 아닙니다.")


    @staticmethod
    def onnx_load(onnx_model_path):
        """
            onnx_model_path : {model_dir} / {model_name}.onnx
        """
        if not os.path.exists(onnx_model_path):
            raise Exception(f"{onnx_model_path} 모델이 존재하지 않습니다.")
        else:
            sess_opt = ort.SessionOptions()
            sess_opt.enable_mem_pattern = False
            sess_opt.enable_cpu_mem_arena = False
            sess_opt.intra_op_num_threads = 1
            onnx_model = onnx.load(onnx_model_path)
            model = ort.InferenceSession(onnx_model.SerializeToString(), sess_options=sess_opt)
            return model
