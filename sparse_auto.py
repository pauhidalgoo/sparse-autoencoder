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

# Hyperparameters (adjusted based on model size and Scaling Monosemanticity)
batch_size = 4096  # Critical batch size mentioned, but can be reduced
learning_rate = 5e-5
l1_lambda = 5 # As recommended in the paper. Start at 0 and linearly ramp up to 5 over 5% of training
num_epochs = 500 # This is going to be tuned based on how the model performs
hidden_dim_multiplier = 4  # Hidden dimension of SAE (e.g., 4 * MLP layer size)
max_length = 512
training_steps = 50000
warmup_steps = int(training_steps * 0.05) # 5% of total training steps
lr_decay_steps = int(training_steps*0.2) # 20% of total training steps

def get_activation_data_batched(texts, layer_name, batch_size=64, max_length=512):
    """
    Extracts and normalizes activation data from a specified layer of a transformer model in batches.

    Args:
        texts: A list of input texts.
        layer_name: The name of the layer to extract activations from (e.g., "model.layers.15.mlp.fc_in").
        batch_size: The batch size to use for processing the texts.
        max_length: The maximum sequence length for tokenization.

    Returns:
        A PyTorch tensor containing the normalized activation data.
    """
    all_activations = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations in batches"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            layer_index = int(layer_name.split(".")[2])
            layer_activations = hidden_states[layer_index + 1]  # Shape: (batch_size, seq_len, hidden_dim)

            attention_mask = inputs["attention_mask"]  # Shape: (batch_size, seq_len)
            valid_token_mask = attention_mask.bool()

            # Iterate through the batch and extract valid activations
            for j in range(layer_activations.size(0)):
                valid_activations = layer_activations[j][valid_token_mask[j]]  # Shape: (num_valid_tokens, hidden_dim)
                all_activations.append(valid_activations)

    activation_data = torch.cat(all_activations, dim=0)

    return activation_data


# SAE Model (with modifications from Scaling Monosemanticity)
class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SAE, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def get_learned_dict(self):
      # See last section of the prompt
      W_e = self.encoder.weight.data
      b_e = self.encoder.bias.data
      W_d = self.decoder.weight.data
      b_d = self.decoder.bias.data

      W_d_normalized = nn.functional.normalize(W_d, p=2, dim=0)
      W_e_normalized = W_e * torch.norm(W_d, p=2, dim=0)
      b_e_normalized = b_e * torch.norm(W_d, p=2, dim=0)

      return W_e_normalized, b_e_normalized, W_d_normalized, b_d

def sparse_loss(x, encoded, decoded, l1_lambda, W_d):
    mse_loss = nn.MSELoss()(decoded, x)
    l1_loss = l1_lambda * torch.mean(torch.norm(encoded, p=1))
    l2_loss = torch.sum(W_d ** 2)
    total_loss = mse_loss + l1_loss + l2_loss
    return total_loss, mse_loss, l1_loss, l2_loss





import os
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    model_path = os.path.join(checkpoint_dir, f"checkpoint_model_epoch_{epoch}_step_{step}.pth")
    torch.save(sae.state_dict(), model_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Training function (with modifications and logging)
def train_sae(sae, data, num_epochs, batch_size, learning_rate, l1_lambda):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=lr_decay_steps)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    step = 0
    for epoch in range(num_epochs):
        for batch, in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch[0].to(device)

            current_l1_lambda = l1_lambda * min(1.0, step / warmup_steps) # Ramp up l1 lambda

            encoded, decoded = sae(batch)
            W_d = sae.decoder.weight
            loss, mse_loss, l1_loss, l2_loss = sparse_loss(batch, encoded, decoded, current_l1_lambda, W_d)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0) # Clip gradients
            optimizer.step()
            
            if step < lr_decay_steps:
              scheduler.step()

            step += 1

            if step % 1000 == 0:  # Save checkpoint every 1000 steps
                save_checkpoint(sae, optimizer, epoch, step, checkpoint_dir)

            if step >= training_steps:
                print("Reached maximum training steps.")
                return

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}, L2 Loss: {l2_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]}")



if __name__ == "__main__":
    from datasets import load_dataset

    # Load the WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    def preprocess_function(examples):
        return {"text": examples["text"]}

    dataset = dataset.map(preprocess_function, batched=True)

    dataset.shuffle()
    sample_texts = dataset["text"][:10000]

    torch.save(sample_texts, "sample_texts.pt")

    layer_index = 15
    mlp_layer_name = f"model.layers.{layer_index}.input_layernorm"
    layer_index = int(mlp_layer_name.split(".")[2])
    print(layer_index)

    # Extract and save activations
    #activation_data = get_activation_data_batched(sample_texts, mlp_layer_name)
    activation_data = torch.load("activation_data.pt")
    mlp_dim = activation_data.shape[-1]
    #torch.save(activation_data, "activation_data.pt")

    # Train the SAE
    hidden_dim = hidden_dim_multiplier * mlp_dim
    print(mlp_dim)
    print(hidden_dim)
    sae = SAE(mlp_dim, hidden_dim).to(device)
    train_sae(sae, activation_data, num_epochs, batch_size, learning_rate, l1_lambda)

    # Save the trained SAE
    torch.save(sae.state_dict(), "sae_model2.pth")