import torch
import onnxruntime
from baseline_model_class import CustomBinaryCNN

def convert_pth_to_onnx(model_class, pth_model_path: str, onnx_save_path: str):
    model = model_class()
    model.load_state_dict(torch.load(pth_model_path, map_location=torch.device('cpu')))

    test_size = 64
    dummy_input = (torch.randn(1, 3, test_size, test_size),)
        
    onnx_program = torch.onnx.export(model, dummy_input, dynamo=True)
    onnx_program.save(onnx_save_path)

class ModelController():
    def __init__(self, onnx_model_path: str):
        self.ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

    def run_onnx_model(self, tensor_to_use):
        onnx_input = tensor_to_use.detach().cpu().numpy()
        input_name = self.ort_session.get_inputs()[0].name
        onnxruntime_input = {input_name : onnx_input}
        onnxruntime_output = self.ort_session.run(None, onnxruntime_input)[0]
        return onnxruntime_output

if __name__ == "__main__":
    convert_pth_to_onnx(CustomBinaryCNN, "models/pytorch/baseline_model.pth", "models/onnx/baseline_model.onnx")
