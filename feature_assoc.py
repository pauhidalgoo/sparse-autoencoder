
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
learning_rate = 1e-3
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



sample_texts = [
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Space, the final frontier. These are the voyages of the starship Enterprise.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Early to bed and early to rise makes a man healthy, wealthy, and wise.",
    "A journey of a thousand miles begins with a single step.",
    "When in Rome, do as the Romans do.",
    "The pen is mightier than the sword.",
    "Actions speak louder than words.",
    "You can't judge a book by its cover.",
    "Where there's a will, there's a way.",
    "I love Leo Messi."
]


def get_feature_associations(data, sae, tokenizer, layer_index, top_k=10, num_features=25):
    feature_associations = {i: [] for i in range(num_features)}

    for text in tqdm(data, desc="Analyzing feature associations"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_activations = hidden_states[layer_index + 1]  

            
            feature_activations, _ = sae(layer_activations[0])  
            
            for feature_index in range(num_features):
                
                activations_for_feature = feature_activations[:, feature_index]

                
                token_activations = [(tokens[i], activations_for_feature[i].item()) for i in range(len(tokens))]

                
                token_activations.sort(key=lambda x: x[1], reverse=True)

                
                feature_associations[feature_index].extend(token_activations[:top_k])

    
    for feature_index in feature_associations:
        feature_associations[feature_index] = list({t[0]: t for t in feature_associations[feature_index]}.values())
        feature_associations[feature_index].sort(key=lambda x: x[1], reverse=True)
        feature_associations[feature_index] = feature_associations[feature_index][:top_k]

    return feature_associations


num_features_to_analyze = 25
top_k_tokens_per_feature = 5
feature_associations = get_feature_associations(
    sample_texts, sae, tokenizer, layer_index, top_k=top_k_tokens_per_feature, num_features=num_features_to_analyze
)


for feature_index, tokens in feature_associations.items():
    print(f"Feature {feature_index}: {[(token, activation) for token, activation in tokens]}")

