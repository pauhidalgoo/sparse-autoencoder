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
learning_rate = 1e-4
l1_lambda = 1e-6
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

"""
model.layers.18.input_layernorm
model.layers.18.post_attention_layernorm
model.layers.19
model.layers.19.self_attn
model.layers.19.self_attn.q_proj
model.layers.19.self_attn.k_proj
model.layers.19.self_attn.v_proj
model.layers.19.self_attn.o_proj
model.layers.19.self_attn.rotary_emb
model.layers.19.mlp
model.layers.19.mlp.gate_proj
model.layers.19.mlp.up_proj
model.layers.19.mlp.down_proj
model.layers.19.mlp.act_fn
model.layers.19.input_layernorm
model.layers.19.post_attention_layernorm
model.layers.20
model.layers.20.self_attn
model.layers.20.self_attn.q_proj
model.layers.20.self_attn.k_proj
model.layers.20.self_attn.v_proj
model.layers.20.self_attn.o_proj
model.layers.20.self_attn.rotary_emb
model.layers.20.mlp
model.layers.20.mlp.gate_proj
model.layers.20.mlp.up_proj
model.layers.20.mlp.down_proj
model.layers.20.mlp.act_fn
model.layers.20.input_layernorm
model.layers.20.post_attention_layernorm
model.layers.21
model.layers.21.self_attn
model.layers.21.self_attn.q_proj
model.layers.21.self_attn.k_proj
model.layers.21.self_attn.v_proj
model.layers.21.self_attn.o_proj
model.layers.21.self_attn.rotary_emb
model.layers.21.mlp
model.layers.21.mlp.gate_proj
model.layers.21.mlp.up_proj
model.layers.21.mlp.down_proj
"""

mlp_layer_name = "model.layers.29.mlp"
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
print(mlp_dim)
print(hidden_dim)
sae = SAE(mlp_dim, hidden_dim).to(device)

def train_sae(sae, data, num_epochs, batch_size, learning_rate, l1_lambda):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch, in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch[0].to(device)  

            encoded, decoded = sae(batch)
            mse_loss = criterion(decoded, batch)
            l1_loss = l1_lambda * torch.norm(encoded, p=1)
            loss = mse_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}")

train_sae(sae, activation_data, num_epochs, batch_size, learning_rate, l1_lambda)

torch.save(sae.state_dict(), "sae_model.pth")
