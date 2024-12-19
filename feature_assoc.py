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
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Hyperparameters (should match those used during training)
batch_size = 4096
learning_rate = 5e-5
l1_lambda = 50
num_epochs = 1000
hidden_dim_multiplier = 8
max_length = 512
# Layer to analyze
layer_index = 15
mlp_layer_name = f"model.layers.{layer_index}.input_layernorm"
# Load activation data
activation_data = torch.load("activation_data.pt")
mlp_dim = activation_data.shape[-1]
hidden_dim = hidden_dim_multiplier * mlp_dim


# Load the trained SAE
sae = SAE(mlp_dim, hidden_dim).to(device)
sae.load_state_dict(torch.load("sae_model2.pth", map_location=torch.device("cpu")))
sae.eval()

# Sample texts for analysis (use a diverse set)
sample_texts = list(torch.load("sample_texts.pt"))
sample_texts = [a for a in sample_texts if a != ""]
import random
random.shuffle(sample_texts)
sample_texts= sample_texts[:1000]
print(sample_texts[:2])

# Function to get feature associations (batched version)
def get_feature_associations_batched(data, sae, tokenizer, layer_index, top_k=10, num_features=25, batch_size=32):
    feature_associations = {i: [] for i in range(num_features)}
    
    for batch_start in tqdm(range(0, len(data), batch_size), desc="Analyzing feature associations (batched)"):
        batch_texts = data[batch_start : batch_start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        
        all_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_activations = hidden_states[layer_index + 1]

            # Get feature activations from the SAE for the entire batch
            feature_activations, _ = sae(layer_activations) # Shape: [batch_size, seq_len, hidden_dim]

            for batch_idx in range(feature_activations.shape[0]):
              for feature_index in range(num_features):
                  activations_for_feature = feature_activations[batch_idx, :, feature_index]
                  token_activations = [(all_tokens[batch_idx][i], activations_for_feature[i].item()) for i in range(len(all_tokens[batch_idx]))]
                  token_activations.sort(key=lambda x: x[1], reverse=True)
                  feature_associations[feature_index].extend(token_activations[:top_k])

    # Remove duplicates and keep only top_k for each feature
    for feature_index in feature_associations:
        feature_associations[feature_index] = list({t[0]: t for t in feature_associations[feature_index]}.values())
        feature_associations[feature_index].sort(key=lambda x: x[1], reverse=True)
        feature_associations[feature_index] = feature_associations[feature_index][:top_k]

    return feature_associations

# Analyze feature associations
num_features_to_analyze = 200
top_k_tokens_per_feature = 5
batch_size_analysis = 8 # Adjust this based on your GPU memory
feature_associations = get_feature_associations_batched(
    sample_texts, sae, tokenizer, layer_index, top_k=top_k_tokens_per_feature, num_features=num_features_to_analyze, batch_size=batch_size_analysis
)

# Print the feature associations
for feature_index, tokens in feature_associations.items():
    print(f"Feature {feature_index}: {[(token, activation) for token, activation in tokens]}")

with open("feature_associations.json", "w") as f:
    json.dump(feature_associations, f, indent=4)