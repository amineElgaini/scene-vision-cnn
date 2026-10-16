import base64
import io
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

MODEL_PATH = "models/final_model.keras"
CLASSES_PATH = "models/class_names.json"
SAMPLE_DIR = Path("data/seg_pred/seg_pred")

COLORS = {
    "buildings": "#B5533C",
    "forest": "#2F6B3F",
    "glacier": "#7CC4DE",
    "mountain": "#6B7A8F",
    "sea": "#0F5F8C",
    "street": "#D9962B",
}
LABELS = {
    "buildings": "Buildings",
    "forest": "Forest",
    "glacier": "Glacier",
    "mountain": "Mountain",
    "sea": "Sea",
    "street": "Street",
}

st.set_page_config(page_title="SceneVision", page_icon="🏔️", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Instrument Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 880px; padding-top: 2.5rem; padding-bottom: 3rem; }

h1.title {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800; font-size: 3rem; letter-spacing: -0.03em;
    margin: 0 0 0.25rem 0; color: #14263A;
}
p.subtitle { color: #5B6B7B; font-size: 1.1rem; margin: 0 0 1.5rem 0; max-width: 34rem; }

.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.75rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: #FFFFFF; border-radius: 999px; padding: 0.3rem 0.8rem;
    font-size: 0.9rem; color: #14263A;
}
.dot { width: 0.65rem; height: 0.65rem; border-radius: 50%; display: inline-block; }

.photo { width: 100%; border-radius: 18px; display: block; }

.result { border-radius: 18px; padding: 1.5rem 1.5rem 1.25rem 1.5rem; }
.result .name {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800; font-size: 2.6rem; letter-spacing: -0.03em;
    color: #14263A; line-height: 1.05; margin: 0;
}
.result .conf { color: #14263A; font-size: 1.05rem; margin: 0.35rem 0 1.1rem 0; }
.result .hint { color: #14263A; font-size: 0.95rem; margin-top: 1rem; }

.row { display: flex; align-items: center; gap: 0.75rem; margin: 0.45rem 0; font-size: 0.95rem; color: #14263A; }
.row .lab { width: 5.5rem; }
.row .track { flex: 1; height: 0.6rem; background: rgba(20,38,58,0.10); border-radius: 999px; overflow: hidden; }
.row .fill { height: 100%; border-radius: 999px; }
.row .pct { width: 3rem; text-align: right; font-variant-numeric: tabular-nums; }

.empty {
    border: 2px dashed #B9C7D3; border-radius: 18px; padding: 2.5rem 1.5rem;
    text-align: center; color: #5B6B7B;
}
.note { color: #5B6B7B; font-size: 0.9rem; margin-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    class_names = json.load(open(CLASSES_PATH))
    return model, class_names


def preprocess(img):
    arr = tf.convert_to_tensor(np.array(img.convert("RGB")))
    arr = tf.image.resize(arr, (150, 150)) / 255.0
    return tf.expand_dims(arr, 0)


def to_data_uri(img, width=600):
    img = img.convert("RGB")
    h = int(img.height * width / img.width)
    img = img.resize((width, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


model, class_names = load_model()

st.markdown('<h1 class="title">SceneVision</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a photo and the model tells you what kind of place it shows.</p>',
    unsafe_allow_html=True,
)
chips = "".join(
    f'<span class="chip"><span class="dot" style="background:{COLORS[c]}"></span>{LABELS[c]}</span>'
    for c in class_names
)
st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

samples = list(SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
if samples and st.button("Try a random photo"):
    st.session_state["sample"] = str(random.choice(samples))

img = None
if file:
    img = Image.open(file)
elif "sample" in st.session_state:
    img = Image.open(st.session_state["sample"])

if img is None:
    st.markdown(
        '<div class="empty">Drop a photo above to see the prediction.</div>',
        unsafe_allow_html=True,
    )
else:
    probs = model.predict(preprocess(img), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = class_names[order[0]]
    second = class_names[order[1]]
    p_top = float(probs[order[0]])
    color = COLORS[top]

    left, right = st.columns([5, 6], gap="large")

    with left:
        st.markdown(
            f'<img class="photo" src="{to_data_uri(img)}" />',
            unsafe_allow_html=True,
        )

    with right:
        rows = "".join(
            f'<div class="row"><span class="lab">{LABELS[class_names[i]]}</span>'
            f'<span class="track"><span class="fill" style="width:{probs[i]*100:.1f}%;'
            f'background:{COLORS[class_names[i]]}; display:block"></span></span>'
            f'<span class="pct">{probs[i]:.0%}</span></div>'
            for i in order
        )
        hint = ""
        if p_top < 0.6:
            hint = (
                f'<div class="hint">The model is not sure. '
                f'It also sees {LABELS[second].lower()}.</div>'
            )
        st.markdown(
            f'<div class="result" style="background:{color}26">'
            f'<p class="name">{LABELS[top]}</p>'
            f'<p class="conf">{p_top:.1%} sure</p>'
            f'{rows}{hint}</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="note">The model knows 6 places only. For any other photo, it still picks one of them.</p>',
    unsafe_allow_html=True,
)