import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_model import UNet

# =========================
# LOAD MODEL
# =========================
model = UNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# =========================
# SELECT FILE (CHANGE HERE)
# =========================
file = h5py.File("dataset/test/volume_101_slice_50.h5", "r")

image = file["image"][:]
mask = file["mask"][:]

# preprocess
image = np.transpose(image, (2,0,1))
image = np.expand_dims(image, axis=0)

if len(mask.shape) == 3:
    mask = mask[:,:,0]

image = torch.tensor(image, dtype=torch.float32)

# =========================
# PREDICTION
# =========================
with torch.no_grad():
    pred = model(image)

pred = torch.sigmoid(pred)
pred = pred.numpy()[0][0]
img = image.numpy()[0][0]

# =========================
# SHOW RESULT
# =========================
plt.subplot(1,3,1)
plt.title("MRI")
plt.imshow(img, cmap="gray")

plt.subplot(1,3,2)
plt.title("Prediction")
plt.imshow(pred > 0.2)

plt.subplot(1,3,3)
plt.title("Heatmap")
plt.imshow(pred)

plt.show()