"""Road anomaly detector: Streamlit web application (deployment of notebooks/RoadAnomaly_complete.ipynb).

Use cases:
  UC1  Analyse a road image: anomaly mask + image-level alert       (features F1, F3)
  UC2  Adjust the sensitivity (decision threshold)                    (F2)
  UC3  See the anomaly score map                                      (F2)
  UC4  Explain the detection with Grad-CAM                            (F4)
  UC5  Inspect a batch of images and download a CSV report            (F3, F5)
  UC6  Export the alert as JSON for another system                    (F6)
  UC7  Read the model card: test performance and limits               (trust)

Run locally:   pip install -r requirements.txt   then   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud from a GitHub repository (see README.md)
"""
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from roadanomaly_core import Detector, alert_from, heat_image, overlay

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "roadanomaly_model.pt"
EXAMPLES = {p.stem: p for p in sorted((HERE / "examples").glob("*.jpg"))}

st.set_page_config(page_title="Road anomaly detector", layout="wide")


def model_url():
    # where to download the model when it is not next to the app: Streamlit secret or environment variable MODEL_URL
    try:
        url = st.secrets.get("MODEL_URL", "")
    except Exception:                                         # no secrets file
        url = ""
    return url or os.environ.get("MODEL_URL", "")


@st.cache_resource(show_spinner="Loading the model…")
def load_detector():
    if not MODEL_FILE.exists():
        url = model_url()
        if not url:
            st.error("roadanomaly_model.pt is missing. Put it next to streamlit_app.py, or set the MODEL_URL secret "
                     "to a direct download link (for example a GitHub release asset). See README.md.")
            st.stop()
        urllib.request.urlretrieve(url, MODEL_FILE)
    return Detector(MODEL_FILE)


det = load_detector()


@st.cache_data(max_entries=16, show_spinner=False)
def scores_of(image_bytes):
    rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return det.score_map(rgb)


@st.cache_data(max_entries=8, show_spinner=False)
def cam_of(image_bytes, threshold):
    rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return det.grad_cam(rgb, threshold)


# ---------------------------------------------------------------- sidebar: sensitivity (UC2)
PRESETS = {"High sensitivity: miss fewer obstacles": 0.25,
           f"Balanced: validated threshold ({det.threshold:.2f})": round(det.threshold, 2),
           "Few false alarms: only confident detections": 0.70}
with st.sidebar:
    st.header("Sensitivity")
    preset = st.radio("Preset", list(PRESETS), index=1)
    threshold = st.slider("Decision threshold (lower = more sensitive)", 0.05, 0.95, PRESETS[preset], 0.01,
                          key=f"thr_{preset}")
    st.caption("A lower threshold misses fewer obstacles but raises more false alarms. Only the balanced value "
               "was measured on the validation set.")
    explain = st.checkbox("Explain the detection (Grad-CAM)", value=True)
    st.divider()
    st.caption(f"Model: {det.name} · runs on {det.device.type.upper()}")

st.title("Road anomaly detector")
st.write("Upload a road photo: every pixel that looks like an unexpected obstacle is marked in red, and an alert "
         "tells you how many obstacles there are and where.")

tab_one, tab_batch, tab_card = st.tabs(["Analyse an image", "Batch inspection", "Model card"])

# ---------------------------------------------------------------- UC1, UC3, UC4, UC6
with tab_one:
    source = st.radio("Image", ["Upload a photo", "Use a test image"], horizontal=True,
                      index=0 if not EXAMPLES else 1)
    image_bytes, name = None, None
    if source == "Upload a photo":
        up = st.file_uploader("Road image", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            image_bytes, name = up.getvalue(), up.name
    elif EXAMPLES:
        choice = st.selectbox("Test image from RoadAnomaly", list(EXAMPLES))
        image_bytes, name = EXAMPLES[choice].read_bytes(), EXAMPLES[choice].name
        st.caption("Example photos from Wikimedia Commons, CC licences: see examples/CREDITS.txt.")

    if image_bytes is None:
        st.info("Upload a road image to start.")
    else:
        rgb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with st.spinner("Detecting anomalies…"):
            prob, latency = scores_of(image_bytes)
        mask, info = alert_from(prob, threshold)

        if info["alert"]:
            n = info["objects"]
            st.error(f"**ALERT: {n} unexpected obstacle{'s' if n > 1 else ''} on the road.** Largest one: "
                     f"{info['main position']}.")
        else:
            st.success(f"No alert: no anomaly above the threshold {threshold:.2f}.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Obstacles", info["objects"])
        m2.metric("Anomaly area", f"{info['anomaly area (%)']} %")
        m3.metric("Threshold", f"{threshold:.2f}")
        m4.metric("Inference", f"{latency:.0f} ms")

        c1, c2 = st.columns(2)
        c1.image(overlay(rgb, mask), caption="Detected anomalies (red)")
        c2.image(heat_image(prob), caption="Anomaly score map: dark = normal, light yellow = anomaly")
        if explain:
            with st.spinner("Computing Grad-CAM…"):
                cam = cam_of(image_bytes, round(threshold, 2))
            c3, c4 = st.columns(2)
            c3.image(heat_image(cam, rgb), caption="Grad-CAM: regions the model relied on")
            c4.image(rgb, caption="Original photo")

        report = {"model": det.name, "image": name, "image size": [rgb.width, rgb.height],
                  "latency (ms)": round(latency, 1), **info}
        with st.expander("Alert details and export for another system (JSON)"):
            st.json(report)
            st.download_button("Download the alert (JSON)", json.dumps(report, indent=2),
                               file_name=f"{Path(name).stem}_alert.json", mime="application/json")
            st.caption("A fleet back-end can read this JSON. For a live API, the same Detector class of "
                       "roadanomaly_core.py can be served with FastAPI (see README.md).")

# ---------------------------------------------------------------- UC5
with tab_batch:
    st.write("For a road inspection run: add several images, get one line per image, the images with an alert "
             "and a CSV report. The threshold of the sidebar is used.")
    ups = st.file_uploader("Road images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
    if st.button("Inspect all images", type="primary", disabled=not ups):
        rows, flagged = [], []
        bar = st.progress(0.0)
        for k, up in enumerate(ups, start=1):
            try:
                data = up.getvalue()
                rgb = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                rows.append({"image": up.name, "status": "unreadable file"})
                continue
            prob, latency = scores_of(data)
            mask, info = alert_from(prob, threshold)
            rows.append({"image": up.name, "status": "ALERT" if info["alert"] else "ok", "objects": info["objects"],
                         "anomaly area (%)": info["anomaly area (%)"], "main position": info["main position"],
                         "max score": round(float(prob.max()), 3), "latency (ms)": round(latency, 1)})
            if info["alert"]:
                flagged.append((overlay(rgb, mask), f"{up.name}: {info['objects']} obstacle(s), {info['main position']}"))
            bar.progress(k / len(ups))
        table = pd.DataFrame(rows)
        n_alert = int((table["status"] == "ALERT").sum())
        st.subheader(f"{len(table)} images inspected, {n_alert} with an alert")
        st.dataframe(table, hide_index=True)
        st.download_button("Download the report (CSV)", table.to_csv(index=False), file_name="road_inspection_report.csv",
                           mime="text/csv")
        if flagged:
            cols = st.columns(3)
            for k, (img, cap) in enumerate(flagged):
                cols[k % 3].image(img, caption=cap)

# ---------------------------------------------------------------- UC7
with tab_card:
    st.subheader(det.name)
    st.write("Binary semantic segmentation: for every pixel, the probability that it belongs to an unexpected "
             "obstacle on the road (animal, rock, cone, lost tire, unusual vehicle).")
    st.table(pd.DataFrame({"value": {
        "Architecture": f"U-Net decoder on an ImageNet-pretrained encoder ({det.pkg['arch']})",
        "Input": f"any RGB image, resized to {det.w}×{det.h} and normalised like ImageNet",
        "Validated threshold": f"{det.threshold:.3f} (chosen on the validation set)",
        "Training data": "RoadAnomaly: 60 annotated images (40 train / 10 validation / 10 test)"}}))
    if det.metrics:
        st.write("**Test results (10 images)**")
        st.table(pd.DataFrame({"value": {k: f"{v:.3f}" for k, v in det.metrics.items()}}))
    st.warning("**Limits.** Trained on 60 daytime web images: not validated for night, rain, other cameras or video. "
               "People and ordinary vehicles are not labelled as anomalies but can trigger alerts. Recall is about "
               "0.75: roughly one anomaly pixel in four is missed. Decision-support prototype, not a safety-certified "
               "component.")
