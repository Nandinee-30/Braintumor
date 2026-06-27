from flask import Flask, render_template, request, redirect
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json, os
import torch
import cv2

from reportlab.platypus import SimpleDocTemplate, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from train_model import UNet

app = Flask(__name__)

HISTORY_FILE = "history.json"

# ===== LOAD HISTORY =====
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
else:
    history = []

# ===== LOAD MODEL =====
model = UNet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# ===== METRICS =====
def dice_score(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    intersection = np.sum(pred * true)
    return (2 * intersection) / (np.sum(pred) + np.sum(true) + 1e-8)

def iou_score(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    intersection = np.sum(pred * true)
    union = np.sum(pred) + np.sum(true) - intersection
    return intersection / (union + 1e-8)

# ===== GRAPH =====
def save_graph(prob):
    labels = ["Tumor", "Normal"]
    values = [prob, 100 - prob]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Confidence Graph")
    plt.ylabel("Percentage")
    plt.savefig("static/graph.png")
    plt.close()

# ===== PDF =====
def generate_pdf(prob, dice=None, iou=None):
    doc = SimpleDocTemplate("static/report.pdf")
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Brain Tumor Detection Report", styles['Title']))
    content.append(Paragraph(f"Probability: {prob}%", styles['Normal']))

    if dice is not None:
        content.append(Paragraph(f"Dice Score: {dice}", styles['Normal']))
    if iou is not None:
        content.append(Paragraph(f"IoU Score: {iou}", styles['Normal']))

    content.append(Image("static/mri.png", width=200, height=200))
    content.append(Image("static/overlay.png", width=200, height=200))
    content.append(Image("static/mask.png", width=200, height=200))
    content.append(Image("static/heatmap.png", width=200, height=200))
    content.append(Image("static/graph.png", width=200, height=200))

    doc.build(content)

# ===== MODEL FUNCTION =====
def run_model(file):

    with h5py.File(file, 'r') as f:
        img = f['image'][:]
        true_mask = f['mask'][:] if 'mask' in f else None

    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.tensor(img, dtype=torch.float32)

    with torch.no_grad():
        pred = model(img_tensor)

    pred = torch.sigmoid(pred).numpy()[0][0]

    # MASK
    mask = (pred > 0.3).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)

    # NORMALIZE
    original = img_tensor.numpy()[0][0]
    original = (original - original.min())/(original.max()-original.min()+1e-8)

    prob = round(float(np.mean(mask)*100),2)

    # OVERLAY
    overlay = np.stack([original]*3, axis=-1)
    overlay[mask > 0] = [1,0,0]

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_uint8 = (overlay*255).astype(np.uint8)
    cv2.drawContours(overlay_uint8, contours, -1, (0,255,0), 2)
    overlay = overlay_uint8/255.0

    # SAVE IMAGES
    plt.imsave("static/mri.png", original, cmap='gray')
    plt.imsave("static/mask.png", mask*255, cmap='gray')
    plt.imsave("static/heatmap.png", pred, cmap='jet', vmin=0, vmax=1)
    plt.imsave("static/overlay.png", overlay)

    # METRICS
    dice, iou = None, None
    if true_mask is not None:
        if len(true_mask.shape)==3:
            true_mask = true_mask[:,:,0]
        true_mask = (true_mask>0).astype(np.uint8)

        dice = round(dice_score(mask, true_mask),3)
        iou = round(iou_score(mask, true_mask),3)

    save_graph(prob)
    generate_pdf(prob, dice, iou)

    return prob, dice, iou

# ===== ROUTES =====
@app.route("/", methods=["GET","POST"])
def index():
    global history

    result, dice, iou = None, None, None

    if request.method=="POST":
        file=request.files["file"]
        file.save("temp.h5")

        result, dice, iou = run_model("temp.h5")

        history.insert(0,{
            "time":str(datetime.now())[:19],
            "prob":result
        })

        with open(HISTORY_FILE,"w") as f:
            json.dump(history,f)

    return render_template("index.html",
                           result=result,
                           history=history,
                           dice=dice,
                           iou=iou)

@app.route("/delete/<int:id>")
def delete(id):
    global history
    if id < len(history):
        history.pop(id)

    with open(HISTORY_FILE,"w") as f:
        json.dump(history,f)

    return redirect("/")

# ===== RUN =====
if __name__=="__main__":
    app.run(debug=True, use_reloader=False)