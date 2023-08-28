import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import  numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  
)
        
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3


in_channels = 1  # Grayscale input image
out_channels = 1  # Grayscale mask output

# Create an instance of the UNet model
model = UNet(in_channels, out_channels)
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_size=(512, 512)):
        self.data_dir = data_dir
        self.cta_image_path = os.path.join(data_dir, 'cta_image')
        self.brain_mask_path = os.path.join(data_dir, 'brain_mask_path')
        self.cta_image_filenames = [f for f in os.listdir(self.cta_image_path) if f.endswith('_MIP.png')]
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.cta_image_filenames)
    
    def __getitem__(self, idx):
        cta_image_name = self.cta_image_filenames[idx]
        cta_image_path = os.path.join(self.cta_image_path, cta_image_name)
        
        # Extract the prefix to find the corresponding mask image
        prefix = cta_image_name.split('_MIP.png')[0]
        brain_mask_name = f"{prefix}_ROI.nii_MIP.png"  # Construct brain mask filename with ROI naming
        brain_mask_path = os.path.join(self.brain_mask_path, brain_mask_name)
        
        # Load the images
        cta_image = Image.open(cta_image_path).convert('L')
        brain_mask_image = Image.open(brain_mask_path).convert('L')
        
        # Resize the images to the target size
        cta_image = cta_image.resize(self.target_size)
        brain_mask_image = brain_mask_image.resize(self.target_size)

        # Apply transformations 
        if self.transform:
            cta_image = self.transform(cta_image)
            brain_mask_image = self.transform(brain_mask_image)
            
        return {'cta_image': cta_image, 'brain_mask_image': brain_mask_image}

data_dir = r'C:\Proxmed\Dataset Proxmed\extracts\New folder'

# transformations 
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    
])

# Create custom dataset
dataset = CustomDataset(data_dir, transform=transform)

# Create a data loader

batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load and preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
])

# Create the training dataset
target_size = (512,512)
train_dataset = CustomDataset(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_idx, batch_data in enumerate(train_loader):
            
        cta = batch_data['cta_image'].to(device)
        true_mask = batch_data['brain_mask_image'].to(device)

        # Resize input images to match target mask size
        cta_resized = torch.nn.functional.interpolate(cta, size=true_mask.shape[2:], mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        predicted_mask = model(cta_resized)

        # Resize predicted mask to match target mask size
        predicted_mask_resized = torch.nn.functional.interpolate(predicted_mask, size=true_mask.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(predicted_mask_resized, true_mask)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "unet_brain_mask_model.pth")



# model = UNet(in_channels=1, out_channels=1)
# model.load_state_dict(torch.load(r"C:\Proxmed\unet_brain_mask_model.pth"))
# model.eval()

# input_image_path = r"C:\Proxmed\Dataset Proxmed\extracts\New folder\cta_image\Anon1.jpg"
# input_image = Image.open(input_image_path).convert('L')

# transform = transforms.Compose([
#     transforms.Resize((512, 512)),  
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  
# ])

# input_image = transform(input_image)
# input_image = input_image.unsqueeze(0)  
# input_image = input_image.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to device

# # Make predictions
# with torch.no_grad():
#     predicted_mask = model(input_image)

# # Post-process the output (apply thresholding)
# threshold = 0.5
# binary_mask = (predicted_mask > threshold).float()

# # Convert to numpy array
# binary_mask = binary_mask.squeeze().cpu().numpy()
# binary_mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))

# # # Save the PIL Image as a JPG file
# # binary_mask_pil.save("predicted_mask.jpg")

# # print("Predicted mask saved as predicted_mask.jpg")

# # Visualize the input image and predicted mask
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(input_image.squeeze().cpu().numpy(), cmap='gray')
# axes[0].set_title("Input Image")
# axes[1].imshow(binary_mask_pil, cmap='gray')
# axes[1].set_title("Predicted Mask")
# plt.show()
