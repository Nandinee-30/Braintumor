import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# DATASET
# =========================
class BrainDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".h5")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = h5py.File(self.files[idx], "r")

        image = file["image"][:]
        mask = file["mask"][:]

        # 🔥 NORMALIZE
        image = image / (np.max(image) + 1e-8)
        mask = mask / (np.max(mask) + 1e-8)

        image = np.transpose(image, (2,0,1))

        if len(mask.shape) == 3:
            mask = mask[:,:,0]

        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# =========================
# SIMPLE UNET (CORRECT)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(4, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))

        m = self.middle(self.pool3(d3))

        u1 = self.up1(m)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.conv3(u3)

        return self.final(u3)


# =========================
# LOSS
# =========================
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)


# =========================
# TRAIN
# =========================
def train():

    train_loader = DataLoader(BrainDataset("dataset/train"), batch_size=2, shuffle=True)
    val_loader = DataLoader(BrainDataset("dataset/val"), batch_size=2, shuffle=False)
    test_loader = DataLoader(BrainDataset("dataset/test"), batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(20):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = bce(out, y) + dice_loss(out, y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # TEST
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = bce(out, y) + dice_loss(out, y)

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print(f"🔥 Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()