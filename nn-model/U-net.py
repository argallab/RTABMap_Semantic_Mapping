import os
import torch
from PIL import Image
import glob
import pytorch_msssim
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

# -----------------------------
# Dataset
# -----------------------------


class IRDotDataset(Dataset):
    def __init__(self, dot_files, clean_files, transform=None):
        self.with_dots_images = dot_files
        self.clean_images = clean_files
        self.transform = transform

        assert len(self.with_dots_images) == len(self.clean_images), \
            f"Mismatch: {len(self.with_dots_images)} vs {len(self.clean_images)} images"

    def __len__(self):
        return len(self.with_dots_images)

    def __getitem__(self, idx):
        with Image.open(self.with_dots_images[idx]) as img:
            img_with_dots = img.convert("L").copy()
        with Image.open(self.clean_images[idx]) as img:
            img_clean = img.convert("L").copy()

        if self.transform:
            img_with_dots = self.transform(img_with_dots)
            img_clean = self.transform(img_clean)

        return img_with_dots, img_clean

def _pad_input(x):
    return F.pad(x, (0, 0, 4, 4), mode='reflect')  # 4 pixels top, 4 bottom
    
def _crop_output(x):
    return x[:, :, 4:364, :]  # [batch, channels, 4:364, :]

# -----------------------------
# Improved U-Net Model
# -----------------------------
class UNet(nn.Module):
    def __init__(self, dropout=0.1):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, dropout=0.0):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # Encoder
        self.enc1 = CBR(1, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = CBR(512, 1024, dropout)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = CBR(1024, 512, dropout)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256, dropout)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)

        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                    # 368x640
        e2 = self.enc2(self.pool(e1))        # 184x320
        e3 = self.enc3(self.pool(e2))        # 92x160
        e4 = self.enc4(self.pool(e3))        # 46x80
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))   # 23x40
        
        # Decoder with skip connections
        d4 = torch.cat((self.upconv4(b), e4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = torch.cat((self.upconv3(d4), e3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = torch.cat((self.upconv2(d3), e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = torch.cat((self.upconv1(d2), e1), dim=1)
        d1 = self.dec1(d1)
        
        output = torch.sigmoid(self.conv_last(d1))
        return torch.clamp(output, 0, 1)


# -----------------------------
# Loss: L1 + SSIM
# -----------------------------
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):  # Balanced SSIM and L1
        super(CombinedLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.ssim = pytorch_msssim.SSIM(
            data_range=1.0, size_average=True, channel=1)
        self.alpha = alpha

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


# -----------------------------
# Evaluation Function
# -----------------------------
def save_sample_predictions(model, dataloader, device, epoch, num_samples=4):
    """Save sample predictions for visual inspection"""
    model.eval()
    samples_saved = 0

    os.makedirs("validation_samples", exist_ok=True)

    with torch.no_grad():
        for imgs_with_dots, imgs_clean in dataloader:
            if samples_saved >= num_samples:
                break

            imgs_with_dots = imgs_with_dots.to(device)
            imgs_clean = imgs_clean.to(device)

            imgs_with_dots_padded = _pad_input(imgs_with_dots)

            outputs = model(imgs_with_dots_padded)
            outputs = _crop_output(outputs)

            # Save each sample in the batch
            for i in range(imgs_with_dots.size(0)):
                if samples_saved >= num_samples:
                    break

                # Convert to PIL images
                input_img = transforms.ToPILImage()(imgs_with_dots[i].cpu())
                output_img = transforms.ToPILImage()(outputs[i].cpu())
                target_img = transforms.ToPILImage()(imgs_clean[i].cpu())

                # Create side-by-side comparison
                width, height = input_img.size
                comparison = Image.new('L', (width * 3, height))
                comparison.paste(input_img, (0, 0))
                comparison.paste(output_img, (width, 0))
                comparison.paste(target_img, (width * 2, 0))

                # Save
                comparison.save(
                    f"validation_samples/epoch_{epoch:03d}_sample_{samples_saved}.png")
                samples_saved += 1

    print(
        f"  -> Saved {samples_saved} validation samples to validation_samples/")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    num_samples = 0

    with torch.no_grad():
        for imgs_with_dots, imgs_clean in dataloader:
            imgs_with_dots = imgs_with_dots.to(device, non_blocking=True)
            imgs_clean     = imgs_clean.to(device, non_blocking=True)

            imgs_with_dots_padded = _pad_input(imgs_with_dots)

            with torch.autocast(device_type=device.type):
                outputs = model(imgs_with_dots_padded)
                outputs = _crop_output(outputs)
                loss    = criterion(outputs, imgs_clean)

            # Calculate PSNR
            mse = F.mse_loss(outputs, imgs_clean) + 1e-10
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

            total_loss += loss.item()
            total_psnr += psnr.item()
            num_samples += imgs_with_dots.size(0)

    return total_loss / len(dataloader), total_psnr / len(dataloader)


# -----------------------------
# Training Function
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1]
    ])

    # Load dataset
    all_with_dots = sorted(
        glob.glob("/app/dataset/captured_images/input/*.jpg"))
    all_clean = sorted(glob.glob("/app/dataset/captured_images/output/*.jpg"))

    # Split into train/validation
    train_with_dots, val_with_dots, train_clean, val_clean = train_test_split(
        all_with_dots, all_clean, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = IRDotDataset(train_with_dots, train_clean, transform)
    val_dataset = IRDotDataset(val_with_dots, val_clean, transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    # Model, loss, optimizer
    model = UNet(dropout=0.1).to(device)
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Training loop
    epochs = 1000
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    print(
        f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for imgs_with_dots, imgs_clean in train_loader:

            imgs_with_dots = imgs_with_dots.to(device, non_blocking=True)
            imgs_clean     = imgs_clean.to(device, non_blocking=True)

            imgs_with_dots_padded = _pad_input(imgs_with_dots)

            optimizer.zero_grad()
            outputs = model(imgs_with_dots_padded)
            outputs = _crop_output(outputs)
            loss = criterion(outputs, imgs_clean)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss, val_psnr = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Save sample predictions every 5 epochs or when we get a new best model
        should_save_samples = (epoch + 1) % 5 == 0

        # Print progress
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        if device.type == "cuda":
            alloc  = torch.cuda.memory_allocated(device) / 1024**2
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"  GPU mem: {alloc:.0f} MB alloc / {peak:.0f} MB peak")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'best_val_loss': best_val_loss,
                'hyperparameters': {
                    'lr': 2e-4,
                    'batch_size': 4,
                    'dropout': 0.1,
                    'alpha': 0.7,
                    'weight_decay': 1e-5,
                },
            }
            torch.save(checkpoint, "best-unet-24.pth")
            patience_counter = 0
            print("  -> New best model saved!")
            should_save_samples = True  # Always save samples for best model
        else:
            patience_counter += 1

        # Save visual samples
        if should_save_samples:
            save_sample_predictions(model, val_loader, device, epoch)

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

        print("-" * 50)

    # Save final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'val_psnr': val_psnr,
        'hyperparameters': {
            'lr': 2e-4,
            'batch_size': 4,
            'dropout': 0.1,
            'alpha': 0.7,
            'weight_decay': 1e-5,
        },
    }
    torch.save(final_checkpoint, "final-unet-24.pth")
    print("Training completed!")

    model.eval()
    example_input = torch.randn(1, 1, 368, 640).to(device)
    scripted_model = torch.jit.script(model)
    scripted_model.save("unet-24.pt")


if __name__ == "__main__":
    # For training
    train()

