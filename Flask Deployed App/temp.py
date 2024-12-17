import torch

# Define a variable to store the model summary or information
model_info = {}

# Load the model file
model_path = 'plant_disease_model_1_latest.pt'
try:
    loaded_model = torch.load(model_path)
    
    # Check if it contains a model state_dict or a full model object
    if isinstance(loaded_model, dict):
        model_info['type'] = "state_dict"
        model_info['keys'] = list(loaded_model.keys())  # Keys in the state_dict
    else:
        model_info['type'] = "full_model"
        model_info['architecture'] = str(loaded_model.__class__)  # Class of the full model

except Exception as e:
    model_info['error'] = str(e)

model_info
