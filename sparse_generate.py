import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import random
from sparse_auto import SAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


batch_size = 32
learning_rate = 1e-5
l1_lambda = 1e-4  
num_epochs = 1000
hidden_dim_multiplier = 8



sample_texts = [
    "The cat sat on the mat.",
]

layer_index = 15
mlp_layer_name = f"model.layers.{layer_index}.input_layernorm"
layer_index = int(mlp_layer_name.split(".")[2])


activation_data = torch.load("activation_data.pt")
mlp_dim = activation_data.shape[-1]
hidden_dim = hidden_dim_multiplier * mlp_dim

sae = SAE(mlp_dim, hidden_dim).to(device)
sae.load_state_dict(torch.load("sae_model3.pth", map_location=torch.device("cpu")))
sae.eval()  






def influence_model_output(text, feature_index, influence_strength, steps=5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    
    def hook_modify_activation(module, input, output):
        """
        Modify token-level activations for the specified feature.
        """
        
        layer_activations = output[0]  

        
        feature_activations, _ = sae(layer_activations)  
        
        feature_activations[:, feature_index] += influence_strength  

        
        modified_activations = sae.decoder(feature_activations)

        
        return modified_activations.unsqueeze(0)
    
    
    handle = model.get_submodule(f"model.layers.{layer_index}.input_layernorm").register_forward_hook(hook_modify_activation)

    
    with torch.no_grad():
        for _ in range(steps):
            outputs = model(**inputs)
            logits = outputs.logits
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id.unsqueeze(-1)], dim=-1)

    
    handle.remove()

    
    generated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return generated_text

def generate_unmodified_output(text, steps=5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    
    with torch.no_grad():
        for _ in range(steps):
            outputs = model(**inputs)
            logits = outputs.logits
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id.unsqueeze(-1)], dim=-1)
    
    
    return tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

original_text = "My favourite TV channel is"
feature_to_influence = 158
influence_strength = 100.0  

messages = [{"role": "user", "content": original_text}]


modified_text = influence_model_output(original_text, feature_to_influence, influence_strength, steps=20)


normal_text = generate_unmodified_output(original_text, steps=25)

print(f"Original text: {original_text}")
print(f"Modified text: {modified_text}")
print(f"Normal text: {normal_text}")