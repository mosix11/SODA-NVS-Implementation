import torch
from torchmetrics.image.fid import FrechetInceptionDistance

# Initialize FID metric
fid = FrechetInceptionDistance(feature=64, normalize=True)  # Using a small feature size for speed

# Generate two completely disjoint distributions
# Dataset 1: Centered around 0 with small variance
data1 = torch.zeros(128, 3, 299, 299)

# Dataset 2: Centered far from Dataset 1 with large variance
data2 = torch.ones(128, 3, 299, 299)

# Update FID for both datasets
fid.update(data1, real=True)
fid.update(data2, real=False)

# Compute FID score
max_fid_score = fid.compute()
print(max_fid_score.item())