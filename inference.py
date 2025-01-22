import json
import numpy as np
import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net(device):
    logger.info("Starting model creation.")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)

    model = model.to(device)
    logger.info("Model creation completed.")

    return model

# Load the model
#def model_fn(model_dir):
#    """Load the model from the specified directory."""
#    model_path = f"{model_dir}/model.pth"
#    model = torch.load(model_path)
#    model.eval()
#    return model

# Load the model
def model_fn(model_dir):
    device = "cpu"
    logger.info(f"Device: {device}")

    model = net(device)

    logger.info("Loading model weights")

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()

    return model
    
# Handle input data
def input_fn(input_data, content_type):
    """Deserialize input data."""
    if content_type == 'application/json':
        input_json = json.loads(input_data)
        # Convert input to numpy and then to torch tensor
        inputs = np.array(input_json["inputs"], dtype=np.float32)
        inputs = torch.tensor(inputs)
        return inputs
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# Handle predictions
def predict_fn(input_data, model):
    """Perform inference on the input data using the loaded model."""
    with torch.no_grad():
        outputs = model(input_data)
    return outputs

# Handle output data
def output_fn(prediction, accept):
    """Serialize output data."""
    if accept == 'application/json':
        prediction_list = prediction.tolist()
        return json.dumps({"predictions": prediction_list}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")








