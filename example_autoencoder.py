
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Linear(embedding_dim, hidden_dim)
        # Decoder
        self.decoder = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # Encoding
        encoded = torch.relu(self.encoder(x))
        # Decoding
        decoded = torch.relu(self.decoder(encoded))
        return decoded

n_modalities = 2
embedding_dim = 1024 * n_modalities  # Dimension of each unimodal embedding
hidden_dim = 1024# Dimension of multimodal embedding

# Model
model = Autoencoder(embedding_dim, hidden_dim)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example unimodal embeddings
embeddings = torch.rand(embedding_dim)  # Single example

# Training Loop
for epoch in range(100):
    optimizer.zero_grad()

    # Forward pass
    reconstructed = model(embeddings)

    # Compute reconstruction loss
    loss = criterion(reconstructed, embeddings)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/100], Loss: {loss.item()}")
# the autoencoder tries to learn to reconstruct the original unimodal embeddings from their combined form. The hidden layer (hidden_dim) represents the multimodal embedding.
