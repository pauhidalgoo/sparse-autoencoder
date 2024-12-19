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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


batch_size = 32
learning_rate = 1e-5
l1_lambda = 1e-4  
num_epochs = 1000
hidden_dim_multiplier = 4  

def get_activation_data(texts, layer_name, max_length=512):
    all_activations = []
    model.eval()

    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting activations"):
            
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

            
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            
            layer_index = int(layer_name.split(".")[2])  
            layer_activations = hidden_states[layer_index + 1]  
            
            
            attention_mask = inputs["attention_mask"]  
            valid_token_mask = attention_mask.bool()  

            
            for i in range(layer_activations.size(0)):  
                valid_activations = layer_activations[i][valid_token_mask[i]]  
                all_activations.append(valid_activations)

    
    return torch.cat(all_activations, dim=0)


sample_texts = [
    "The cat sat on the mat.",
]


mlp_layer_name = "transformer.h.15.mlp"
layer_index = int(mlp_layer_name.split(".")[2])


activation_data = get_activation_data(sample_texts, mlp_layer_name)
mlp_dim = activation_data.shape[-1]  

class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAE, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded


hidden_dim = hidden_dim_multiplier * mlp_dim

sae = SAE(mlp_dim, hidden_dim).to(device)
sae.load_state_dict(torch.load("sae_model.pth"))
sae.eval()  

def influence_model_output(text, feature_index, influence_strength, steps=5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    
    def hook_modify_activation(module, input, output):
        """
        Modify token-level activations for the specified feature.
        """
        
        layer_activations = output[0]  

        
        feature_activations, _ = sae(layer_activations)  
        print(feature_activations)
        
        feature_activations[:, feature_index] += influence_strength  

        
        modified_activations = sae.decoder(feature_activations)

        
        return modified_activations
    
    
    handle = model.get_submodule(f"model.layers.{layer_index}.mlp").register_forward_hook(hook_modify_activation)

    
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

original_text = "The best civilization in history were named the"
feature_to_influence = 23  
influence_strength = 1000.0  

messages = [{"role": "user", "content": original_text}]


modified_text = influence_model_output(original_text, feature_to_influence, influence_strength, steps=20)


normal_text = generate_unmodified_output(original_text, steps=20)

print(f"Original text: {original_text}")
print(f"Modified text: {modified_text}")
print(f"Normal text: {normal_text}")