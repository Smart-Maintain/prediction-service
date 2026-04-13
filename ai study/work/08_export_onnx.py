import torch
import importlib
import os

print("Starting Step 8: Export Model to ONNX...")

# Access the model module
model_module = importlib.import_module("05_model")
device = torch.device('cpu')

# 1. Initialize the PyTorch model and load the best weights
model = model_module.MultiTaskModel(n_features=14, window_size=30).to(device)
model.load_state_dict(torch.load('best_multitask_model.pth', map_location=device))
model.eval()

# 2. Create a dummy input with the right shape (batch_size=1, window_size=30, n_features=14)
dummy_input = torch.randn(1, 30, 14).to(device)

# 3. Export to ONNX
onnx_file_path = "api/multitask_model.onnx"
torch.onnx.export(
    model,                      # model being run
    dummy_input,                # model input (or a tuple for multiple inputs)
    onnx_file_path,             # where to save the model
    export_params=True,         # store the trained parameter weights inside the model file
    opset_version=11,           # the ONNX version to export the model to
    do_constant_folding=True,   # whether to execute constant folding for optimization
    input_names=['input'],      # the model's input names
    output_names=['rul_output', 'alert_output'], # the model's output names
    dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                  'rul_output': {0: 'batch_size'},
                  'alert_output': {0: 'batch_size'}}
)

print(f"Successfully exported PyTorch model to ONNX format: {onnx_file_path}")
