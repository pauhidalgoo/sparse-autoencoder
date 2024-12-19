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
learning_rate = 1e-4
l1_lambda = 1e-6
num_epochs = 1000
hidden_dim_multiplier = 4 

mlp_dim = 576
hidden_dim = mlp_dim * hidden_dim_multiplier



sae = SAE(mlp_dim, hidden_dim).to(device)
sae.load_state_dict(torch.load("sae_model.pth"))


def extract_feature_maps(sae, data):
    with torch.no_grad():
        feature_maps, _ = sae(data)
    return feature_maps

def visualize_feature_maps(feature_maps, num_features=25):

    feature_maps = feature_maps.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(feature_maps[:num_features, :].T, aspect='auto', cmap='viridis')
    plt.xlabel("Data Samples")
    plt.ylabel("Features")
    plt.title("Feature Map Visualization")
    plt.colorbar()
    plt.show()


feature_maps = extract_feature_maps(sae, activation_data)


visualize_feature_maps(feature_maps)


def get_top_activating_tokens(text, feature_index, top_k=5):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        layer_activations = hidden_states[layer_index + 1]  

        
        layer_activations = layer_activations[0]  

        
        feature_activations, _ = sae(layer_activations)  

        
        activations_for_feature = feature_activations[:, feature_index]  

        
        token_activations = [(tokens[i], activations_for_feature[i].item()) for i in range(len(tokens))]

    
    token_activations.sort(key=lambda x: x[1], reverse=True)
    return token_activations[:top_k]


def analyze_feature_correlations(feature_maps):
    correlations = np.corrcoef(feature_maps.cpu().numpy().T)
    return correlations


example_text = "The cat sat on the mat."
feature_index = 10  
top_tokens = get_top_activating_tokens(example_text, feature_index)
print(f"Top activating tokens for feature {feature_index}: {top_tokens}")


correlations = analyze_feature_correlations(feature_maps)
plt.figure(figsize=(8, 8))
plt.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix")
plt.colorbar()
plt.show()
