import json
import itertools
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="WardrobeIQ | Your Digital Closet",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/wardrobeiq_mobilenetv2.keras")
LABEL_PATH = Path("models/class_names.json")

CATEGORIES = ["top", "shirt", "t-shirt", "jeans", "trousers", "skirt", "dress", "jacket", "shoes", "accessory"]
OCCASIONS = ["Casual", "College", "Formal", "Party", "Festive", "Date"]
STYLES = ["Minimal", "Streetwear", "Classic", "Smart casual", "Ethnic", "Trendy"]

# ---------- Core helpers ----------
def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, x))) for x in rgb)

def extract_color(img):
    im = img.convert("RGB").resize((50, 50))
    arr = np.asarray(im).reshape(-1, 3)
    return rgb_to_hex(np.median(arr, axis=0))

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def color_harmony(a, b):
    x, y = np.array(hex_rgb(a)), np.array(hex_rgb(b))
    dist = float(np.linalg.norm(x - y) / (255 * np.sqrt(3)))
    sat_x, sat_y = [max(hex_rgb(c)) - min(hex_rgb(c)) for c in (a, b)]
    neutral_bonus = 0.18 if sat_x < 35 or sat_y < 35 else 0
    return max(0.0, min(1.0, 1.0 - dist + neutral_bonus))

def predict_category(img):
    if not (MODEL_PATH.exists() and LABEL_PATH.exists()):
        return None, None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        labels = json.loads(LABEL_PATH.read_text())
        im = img.convert("RGB").resize((224, 224))
        x = np.asarray(im, dtype=np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        probs = model.predict(np.expand_dims(x, 0), verbose=0)[0]
        idx = int(np.argmax(probs))
        return labels[idx], float(probs[idx])
    except Exception as exc:
        st.warning(f"The image model could not be loaded. You can still add the item manually. Details: {exc}")
        return None, None

# ---------- Session state ----------
for key, default in {
    "wardrobe": [],
    "history": [],
    "feedback": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Beige editorial design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --ink: #29251f;
    --muted: #82786d;
    --paper: #f5f0e8;
    --card: #fffdf9;
    --line: #e7ded2;
    --accent: #806b55;
    --accent-dark: #4c3c2e;
}
.stApp { background: var(--paper); color: var(--ink); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--ink); font-family: 'Playfair Display', serif !important; letter-spacing: -0.03em; }
h1 { font-size: 3rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.35rem !important; }
p, label, .stCaption { color: var(--muted); }
[data-testid="stSidebar"] { background: #ebe2d6; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .45rem .2rem; }
.stButton > button, .stDownloadButton > button {
    background: var(--accent-dark); color: #fffaf3; border: 1px solid var(--accent-dark);
    border-radius: 7px; padding: .55rem 1.1rem; font-weight: 600; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #6a5745; color: white; border-color: #6a5745;
}
.stTextInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea {
    background: #fffdf9; border-radius: 7px;
}
[data-testid="stFileUploader"] { background: #fffdf9; border: 1px dashed #cbbba8; border-radius: 12px; padding: 1rem; }
[data-testid="stMetric"] { background: #fffdf9; border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #fffdf9; border-radius: 13px; border-color: var(--line) !important; }
hr { border-color: var(--line); }
.wiq-brand { font-family: 'Playfair Display', serif; font-size: 1.75rem; font-weight: 700; color: #342a21; letter-spacing: -.04em; }
.wiq-kicker { color: #9a8269; text-transform: uppercase; letter-spacing: .19em; font-size: .68rem; font-weight: 700; }
.wiq-hero { background: #e9dfd2; border: 1px solid #dfd2c2; border-radius: 16px; padding: 2rem 2.2rem; }
.wiq-hero-title { font-family: 'Playfair Display', serif; font-size: 2.5rem; line-height: 1.12; color: #30271f; }
.wiq-hero-copy { color: #766858; max-width: 650px; font-size: .98rem; line-height: 1.7; }
.wiq-section-note { color: #8a7c6e; font-size: .9rem; }
.wiq-pill { display: inline-block; border: 1px solid #d9cbbb; color: #725e4a; border-radius: 30px; padding: .25rem .7rem; font-size: .75rem; }
[data-testid="stImage"] img { border-radius: 9px; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar navigation ----------
with st.sidebar:
    st.markdown('<div class="wiq-brand">WardrobeIQ<span style="color:#a58b70">.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="wiq-kicker" style="margin: .35rem 0 1.8rem">THE INTELLIGENT CLOSET</div>', unsafe_allow_html=True)
    page = st.radio(
        "MENU",
        ["Overview", "My Wardrobe", "AI Stylist", "Style DNA", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown('<div class="wiq-kicker">YOUR CLOSET</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(st.session_state.wardrobe)}** pieces saved")
    st.caption("A little more style, a little less decision-making.")

# ---------- Header ----------
st.markdown('<div class="wiq-kicker">A MORE THOUGHTFUL WAY TO GET DRESSED</div>', unsafe_allow_html=True)
st.markdown('<div class="wiq-hero"><div class="wiq-hero-title">Style starts with<br>what you already own.</div><p class="wiq-hero-copy">Your personal digital wardrobe, with thoughtful outfit ideas built around your clothes, your taste and your day.</p><span class="wiq-pill">YOUR PERSONAL AI STYLE COMPANION</span></div>', unsafe_allow_html=True)
st.write("")

wardrobe = st.session_state.wardrobe

if page == "Overview":
    st.subheader("Your wardrobe, at a glance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pieces in your closet", len(wardrobe))
    c2.metric("Outfits worn", len(st.session_state.history))
    c3.metric("Saved preferences", len(st.session_state.feedback))
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### A closet that works harder")
        st.markdown('<p class="wiq-section-note">Add your favourite pieces and let WardrobeIQ help you make more of them.</p>', unsafe_allow_html=True)
        if st.button("＋  Add a clothing item", key="overview_add"):
            st.session_state["go_to_wardrobe"] = True
            st.rerun()
        if wardrobe:
            st.markdown("#### Recently added")
            recent = wardrobe[-4:][::-1]
            cols = st.columns(min(4, len(recent)))
            for col, item in zip(cols, recent):
                with col:
                    st.image(item["image"], use_container_width=True)
                    st.caption(item["name"])
        else:
            st.info("Your digital closet is ready. Add a few pieces to get started.")
    with right:
        st.markdown("### Your style notes")
        if wardrobe:
            counts = Counter(x["category"] for x in wardrobe)
            st.write("**Most common category**")
            st.write(max(counts, key=counts.get).title())
            st.write("**Styles in your closet**")
            st.write(" · ".join(sorted(set(x["style"] for x in wardrobe))))
        else:
            st.write("Your style profile will take shape as you add clothes.")
        st.markdown("---")
        st.markdown("### A good place to begin")
        st.write("Add one top, one bottom and a pair of shoes. Then visit AI Stylist to create your first look.")

elif page == "My Wardrobe":
    st.subheader("My Wardrobe")
    st.markdown('<p class="wiq-section-note">Every piece you own, thoughtfully organised in one place.</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Add a new piece")
        uploaded = st.file_uploader("Choose a clothing photo", type=["jpg", "jpeg", "png"], key="garment")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            pred, conf = predict_category(image)
            preview_col, details_col = st.columns([1, 1.6])
            with preview_col:
                st.image(image, use_container_width=True)
            with details_col:
                if pred:
                    st.success(f"Model prediction: {pred} · {conf:.1%} confidence")
                else:
                    st.caption("No trained image classifier found. Select the category manually.")
                default_cat = pred if pred in CATEGORIES else "top"
                category = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(default_cat))
                color = extract_color(image)
                st.markdown(f"**Detected dominant colour:** <span style='color:{color}'>●</span> `{color}`", unsafe_allow_html=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    style = st.selectbox("Style", STYLES)
                with col_b:
                    occasion = st.selectbox("Best for", OCCASIONS)
                item_name = st.text_input("Give this piece a name", value=uploaded.name.rsplit(".", 1)[0])
                if st.button("Add to my wardrobe", type="primary"):
                    st.session_state.wardrobe.append({
                        "id": max([x["id"] for x in st.session_state.wardrobe], default=0) + 1,
                        "name": item_name or category, "category": category, "color": color,
                        "style": style, "occasion": occasion, "image": image.copy()
                    })
                    st.success("Added to your wardrobe.")
                    st.rerun()
    st.write("")
    st.markdown(f"### Your pieces  <span class='wiq-pill'>{len(wardrobe)} ITEMS</span>", unsafe_allow_html=True)
    if wardrobe:
        filter_cat = st.selectbox("Filter by category", ["All"] + CATEGORIES, key="filter_cat")
        shown = [x for x in wardrobe if filter_cat == "All" or x["category"] == filter_cat]
        cols = st.columns(4)
        for i, item in enumerate(shown):
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(item["image"], use_container_width=True)
                    st.markdown(f"**{item['name']}**")
                    st.caption(f"{item['category'].title()} · {item['style']}")
                    st.caption(f"Colour {item['color']}")
                    if st.button("Remove piece", key=f"remove_{item['id']}"):
                        st.session_state.wardrobe = [x for x in st.session_state.wardrobe if x["id"] != item["id"]]
                        st.rerun()
    else:
        st.info("Your wardrobe is empty. Upload your first clothing piece above.")

elif page == "AI Stylist":

    st.subheader("The AI Stylist")
    st.markdown(
        '<p class="wiq-section-note">A considered outfit for wherever the day takes you.</p>',
        unsafe_allow_html=True
    )

    if not wardrobe:
        st.info("Add clothing pieces to your wardrobe to get outfit recommendations.")

    else:
        f1, f2, f3 = st.columns(3)

        with f1:
            pref_style = st.selectbox(
                "Your preferred style",
                ["Any"] + STYLES,
                key="pref_style"
            )

        with f2:
            pref_occ = st.selectbox(
                "Occasion",
                ["Any"] + OCCASIONS,
                key="pref_occ"
            )

        with f3:
            avoid_repeat = st.checkbox(
                "Avoid recently worn looks",
                value=True
            )

        # Separate clothing categories correctly
        tops = [
            x for x in wardrobe
            if x["category"] in ["top", "shirt", "t-shirt"]
        ]

        bottoms = [
            x for x in wardrobe
            if x["category"] in ["jeans", "trousers", "skirt"]
        ]

        dresses = [
            x for x in wardrobe
            if x["category"] == "dress"
        ]

        shoes = [
            x for x in wardrobe
            if x["category"] == "shoes"
        ]

        combos = []

        # OPTION 1: TOP + BOTTOM + SHOES
        if tops and bottoms:
            for top, bottom in itertools.product(tops, bottoms):

                for shoe in (shoes or [None]):

                    parts = [top, bottom]

                    if shoe:
                        parts.append(shoe)

                    combos.append(parts)

        # OPTION 2: DRESS + SHOES
        # Never recommend jeans, trousers or skirts with a dress.
        for dress in dresses:

            for shoe in (shoes or [None]):

                parts = [dress]

                if shoe:
                    parts.append(shoe)

                combos.append(parts)

        # Apply filters and calculate outfit scores
        scored_combos = []

        for parts in combos:

            if pref_style != "Any":
                if not any(p["style"] == pref_style for p in parts):
                    continue

            if pref_occ != "Any":
                if not any(p["occasion"] == pref_occ for p in parts):
                    continue

            # Colour compatibility
            if len(parts) >= 2:
                colour_score = color_harmony(
                    parts[0]["color"],
                    parts[1]["color"]
                )
            else:
                colour_score = 1.0

            # Style consistency
            main_style = parts[0]["style"]

            style_score = sum(
                p["style"] == main_style for p in parts
            ) / len(parts)

            # Occasion matching
            if pref_occ == "Any":
                occasion_score = sum(
                    p["occasion"] == parts[0]["occasion"]
                    for p in parts
                ) / len(parts)
            else:
                occasion_score = sum(
                    p["occasion"] == pref_occ for p in parts
                ) / len(parts)

            key = "-".join(str(p["id"]) for p in parts)

            repeat_penalty = (
                0.25
                if key in st.session_state.history[-5:]
                else 0
            )

            score = (
                0.55 * colour_score
                + 0.25 * style_score
                + 0.20 * occasion_score
                - (repeat_penalty if avoid_repeat else 0)
            )

            scored_combos.append((score, key, parts))

        if not scored_combos:
            st.warning(
                "No complete outfits available. Add a top and bottom "
                "or a dress to your wardrobe."
            )

        else:
            scored_combos.sort(
                key=lambda x: x[0],
                reverse=True
            )

            for rank, (score, key, parts) in enumerate(
                scored_combos[:5], 1
            ):

                with st.container(border=True):

                    st.markdown(
                        f"### Look {rank} "
                        f"<span class='wiq-pill'>MATCH {score:.0%}</span>",
                        unsafe_allow_html=True
                    )

                    cols = st.columns(len(parts))

                    for col, item in zip(cols, parts):

                        with col:
                            st.image(
                                item["image"],
                                use_container_width=True
                            )

                            st.caption(
                                f"{item['name']} · "
                                f"{item['category'].title()}"
                            )

                    st.caption(
                        "Outfits are generated using clothing category "
                        "compatibility, colour harmony, style and occasion."
                    )

                    b1, b2, b3 = st.columns([1, 1, 3])

                    with b1:
                        if st.button("♡ Like", key=f"like_{key}"):
                            st.session_state.feedback[key] = "liked"
                            st.success("Saved to your preferences.")

                    with b2:
                        if st.button("Wear this", key=f"wear_{key}"):
                            st.session_state.history.append(key)
                            st.success("Added to outfit history.")

                    with b3:
                        if st.button("Skip", key=f"skip_{key}"):
                            st.session_state.feedback[key] = "skipped"
                            st.info("Feedback saved.")
    
elif page == "Style DNA":
    st.subheader("Your Style DNA")
    st.markdown('<p class="wiq-section-note">A snapshot of the style taking shape in your wardrobe.</p>', unsafe_allow_html=True)
    if wardrobe:
        styles = Counter(x["style"] for x in wardrobe)
        categories = Counter(x["category"] for x in wardrobe)
        a, b = st.columns(2)
        with a:
            st.markdown("#### Style mix")
            st.bar_chart(pd.DataFrame({"Pieces": styles}).sort_values("Pieces", ascending=False))
        with b:
            st.markdown("#### Wardrobe balance")
            st.bar_chart(pd.DataFrame({"Pieces": categories}).sort_values("Pieces", ascending=False))
        st.markdown("---")
        st.metric("Outfit history entries", len(st.session_state.history))
        st.metric("Feedback entries", len(st.session_state.feedback))
    else:
        st.info("Add some clothes to discover your Style DNA.")
    if st.button("Clear outfit history"):
        st.session_state.history = []
        st.success("Outfit history cleared.")

else:
    st.subheader("A little about WardrobeIQ")
    st.markdown("""
    WardrobeIQ is a digital wardrobe and outfit recommendation prototype designed to help people make more of the clothes they already own.

    **How it works**
    1. Upload clothing photos and add their category, style and suitable occasion.
    2. If a trained MobileNetV2 model and its class labels are installed, the app can predict the garment category.
    3. WardrobeIQ extracts an approximate dominant colour from each image.
    4. The outfit engine ranks available top-bottom-shoe combinations using colour harmony, style consistency and occasion matching.
    5. Likes, skips and worn looks are tracked during the current session.

    **Prototype note:** If no trained classifier files are present in the `models` folder, clothing categories must be selected manually. Outfit compatibility scores are rule-based estimates, not learned model predictions. Wardrobe data is currently held in the active app session and is not permanent storage.
    """)
