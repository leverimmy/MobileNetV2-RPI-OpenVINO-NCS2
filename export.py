import numpy as np
import torch
import torch.onnx
import torchvision.models as models
import onnx
import onnxruntime

# Download the model
model = models.mobilenet_v2(pretrained=True)
model.eval()

def export_model(model_name):
    """
    Export the PyTorch model to ONNX format
    """
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    # Export the model
    torch.onnx.export(model,                     # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      model_name,                # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      external_data=False,       # *important*: if not set to `False`, then large models will be saved in two files (.onnx and .onnx.data)
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

def verify_model(model_name):
    """
    (Optional) Verify the exported model
    """
    onnx_model = onnx.load(model_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(model_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    
    # Compute PyTorch output prediction
    torch_out = model(x)

    # Compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    MODEL_NAME = 'mobilenet_v2.onnx'
    export_model(MODEL_NAME)
    verify_model(MODEL_NAME)
