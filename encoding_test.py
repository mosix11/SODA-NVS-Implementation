import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math

class ScaledSinusoidalPositionEmbedBlock(nn.Module):
    def __init__(self, dim, s=0.0001):
        """
        Scaled sinusoidal positional encoding for normalized inputs in range [-1, 1]
        based on the descriptions of the SODA paper. 
        
        Args:
            dim (int): The dimensionality of the positional encoding (must be even).
            s (float): Scaling factor for the arguments of sin and cos.
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        self.dim = dim
        self.s = s

        # Precompute the frequencies for half of the dimensions
        half_dim = dim // 2
        self.register_buffer(
            'frequencies',
            1.0 / ((10000 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim)) * 2 * torch.pi * self.s)
        )

    def forward(self, positions):
        """
        Args:
            positions (torch.Tensor): Input positions (batch_size, seq_length) normalized to [-1, 1].
            
        Returns:
            torch.Tensor: Scaled sinusoidal positional encodings (batch_size, seq_length, dim).
        """
        # Scale positions
        scaled_positions = positions.unsqueeze(-1)  # (batch_size, seq_length, 1)
        
        # Compute sinusoidal embeddings
        sinusoidal = scaled_positions * self.frequencies.unsqueeze(0)  # Broadcast frequencies
        embeddings = torch.cat([torch.cos(sinusoidal), torch.sin(sinusoidal)], dim=-1)  # (batch_size, seq_length, dim)
        # TODO check the impact of the order of concatenation between first `sin` then `cos` vs first `cos` then `sin`. It seems
        # int the original paper they have used `cos` first based on Figure 10 of the SODA paper.
        return embeddings

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        """
        * `dim` is the number of dimensions in the embedding
        """
        super().__init__()
        self.dim = dim
        self.encoding = ScaledSinusoidalPositionEmbedBlock(dim=dim)
        self.lin1 = nn.Linear(self.dim, self.dim)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.dim, self.dim)

    def forward(self, t):
        t = t * 2 - 1 # Original t is between [0, 1] we scale it to [-1, 1]
        emb = self.encoding(t)
        # Transform with the MLP
        # emb = self.act(self.lin1(emb))
        # emb = self.lin2(emb)
        return emb

# class ScaledSinusoidalPositionEmbedBlock(nn.Module):
#     def __init__(self, dim, input_dim=1, s=0.0001):
#         """
#         Scaled sinusoidal positional encoding for normalized inputs in range [-1, 1].
        
#         Args:
#             dim (int): The dimensionality of the positional encoding (must be even).
#             s (float): Scaling factor for the arguments of sin and cos.
#             input_dim (int): The dimensionality of the input vectors to encode (e.g., 1 for time and 6 for r= [o, d]).
#         """
#         super().__init__()
#         assert dim % 2 == 0, "Dimension must be even"
#         assert dim >= input_dim, "Encoding dimensionality must be >= input dimensionality"
#         self.dim = dim
#         self.s = s
#         self.input_dim = input_dim

#         # Precompute the frequencies for half of the dimensions
#         half_dim = dim // 2
#         self.register_buffer(
#             'frequencies',
#             1.0 / ((10000 ** (torch.arange(half_dim, dtype=torch.float32) / half_dim)) * 2 * torch.pi * self.s)
#         )

#     def forward(self, positions):
#         """
#         Args:
#             positions (torch.Tensor): Input positions (..., input_dim), normalized to [-1, 1].
            
#         Returns:
#             torch.Tensor: Scaled sinusoidal positional encodings (..., dim).
#         """
#         # Ensure positions match the expected input dimensions
#         assert positions.shape[-1] == self.input_dim, \
#             f"Expected input_dim={self.input_dim}, but got positions with shape {positions.shape[-1]}"
#         original_shape = positions.shape
        
#         # Scale positions and repeat frequencies to match input dimensions
        
#         frequencies = self.frequencies.unsqueeze(0).unsqueeze(0).expand(-1, self.input_dim, -1)  # Broadcast for input_dim

#         # Compute sinusoidal embeddings
#         sinusoidal = positions * frequencies  # (batch_size, seq_length, input_dim, half_dim)
#         embeddings = torch.cat([
#             torch.cos(sinusoidal),
#             torch.sin(sinusoidal)
#         ], dim=-1)  # (batch_size, seq_length, input_dim, dim)

#         # Reshape to match output dimension
#         embeddings = embeddings.view(*positions.shape[:-1], self.dim)  # (..., dim)
#         return embeddings

    
# Initialize the encoding class
num_dimensions = 512  # Total dimensions of the encoding
# pos_encoding = ScaledSinusoidalPositionEmbedBlock(num_dimensions, input_dim=2)
pos_encoding = TimeEmbedding(num_dimensions)

# Generate positions normalized in the range [-1, 1]
positions = torch.linspace(0, 1, steps=1000)  # Add a batch dimension
# positions_2D = torch.cat([positions, positions], dim=1)
# print(positions_2D.shape)
# Apply positional encoding
encodings = pos_encoding(positions)
print(encodings.shape)

# Convert encodings to a 2D array for visualization
# Rows represent encoding dimensions, and columns represent positions
encodings_np = encodings.squeeze().detach().numpy().T  # Shape: (num_dimensions, num_positions)

# Visualize encodings
plt.figure(figsize=(6, 10))
plt.imshow(encodings_np, cmap='inferno', aspect='auto', extent=[-1, 1, 0, num_dimensions])
# plt.imshow(encodings_np, cmap='inferno', aspect='auto')
plt.gca().invert_yaxis()
plt.colorbar(label='Encoding Value')
plt.title("Complex Sinusoidal Positional Encodings")
plt.xlabel("Normalized Position")
plt.ylabel("Encoding Dimensions")
plt.show()