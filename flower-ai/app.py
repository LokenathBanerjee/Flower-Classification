import io
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.gradcam import apply_heatmap, build_gradcam_models, compute_heatmap
from src.predictor import FlowerPredictor
from src.rag_pipeline import FlowerRAG
from src.utils import format_confidence, get_api_key, get_flower_info, load_json_file

MODEL_PATH = "model/final_model.keras"
CLASS_NAMES_PATH = "data/class_names.json"
FLOWER_INFO_PATH = "data/flower_info.json"
VECTOR_DB_PATH = "vector_db"
MAX_PREVIEW_MB = 10

st.set_page_config(page_title="Flower AI", page_icon="🌸", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #1F1B2E; }

    .block-container { max-width: 760px; margin: 0 auto; padding-top: 2rem; }

    hr.accent {
        border: none;
        height: 4px;
        width: 110px;
        margin: 0.4rem auto 2rem;
        border-radius: 999px;
        background: linear-gradient(90deg, #5B3E8C, #3F9468);
    }

    .pill {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem 0.3rem 0.15rem 0;
    }
    .pill-confident { background: #E3F1E8; color: #2F6B49; }
    .pill-unsure { background: #FBEAD9; color: #A85B22; }
    .pill-source { background: #F1ECF9; color: #5B3E8C; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the flower classifier...")
def load_predictor():
    try:
        return FlowerPredictor(MODEL_PATH, CLASS_NAMES_PATH), None
    except Exception as e:
        return None, str(e)


@st.cache_resource(show_spinner="Connecting the RAG chatbot...")
def load_rag():
    try:
        api_key = get_api_key("GOOGLE_API_KEY")
        return FlowerRAG(VECTOR_DB_PATH, api_key), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_flower_info():
    try:
        return load_json_file(FLOWER_INFO_PATH)
    except Exception:
        return {}


predictor, predictor_error = load_predictor()
rag, rag_error = load_rag()
flower_info_lookup = load_flower_info()

st.title("🌸 Flower AI")
st.caption("Upload a flower photo, get an instant prediction, and ask anything about it.")
st.markdown('<hr class="accent">', unsafe_allow_html=True)

if predictor_error:
    st.error(f"The flower classifier couldn't load: {predictor_error}")
    st.info("Check that model/final_model.keras exists and matches data/class_names.json, then restart the app.")
    st.stop()


@st.cache_resource(show_spinner=False)
def _build_gradcam_cached(_model):
    return build_gradcam_models(_model)

# --- session state -----------------------------------------------------
for key, default in [
    ("uploader_key", 0),
    ("file_fingerprint", None),
    ("image", None),
    ("prediction", None),
    ("gradcam_result", None),   
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def reset_downstream_state():
    st.session_state.prediction = None
    st.session_state.gradcam_result = None
    st.session_state.chat_history = []


# --- Step 1: upload + predict ------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a flower photo",
    type=["jpg", "jpeg", "png", "webp"],
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded_file is not None:
    fingerprint = (uploaded_file.name, uploaded_file.size)

    if fingerprint != st.session_state.file_fingerprint:
        st.session_state.file_fingerprint = fingerprint
        reset_downstream_state()
        try:
            st.session_state.image = Image.open(uploaded_file)
        except (UnidentifiedImageError, OSError):
            st.session_state.image = None
            st.error("That file doesn't look like a valid image. Please try a JPG, PNG, or WEBP photo.")

    if uploaded_file.size > MAX_PREVIEW_MB * 1024 * 1024:
        st.warning(f"That's a large photo ({MAX_PREVIEW_MB}MB+) — prediction may take a little longer than usual.")

if st.session_state.image is not None:
    st.image(st.session_state.image, caption="Uploaded photo", width='stretch')

    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing the photo..."):
            try:
                st.session_state.prediction = predictor.predict(st.session_state.image)
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- Step 2: result, info card, and RAG chat ----------------------------
prediction = st.session_state.prediction

if prediction:
    flower = prediction["flower"]
    display_name = flower.title()

    with st.container(border=True):
        st.subheader(f"🌼 {display_name}")
        st.progress(prediction["confidence"])

        if prediction["is_confident"]:
            st.markdown(
                f'<span class="pill pill-confident">'
                f'Confidence {format_confidence(prediction["confidence"])} · Most confident prediction</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="pill pill-unsure">'
                f'Confidence {format_confidence(prediction["confidence"])} · Not very sure</span>',
                unsafe_allow_html=True,
            )
            st.caption("The model isn't very confident about this one — a clearer, well-lit photo usually helps.")
            with st.expander("See other possibilities"):
                for name, conf in prediction["top_predictions"][1:]:
                    st.write(f"{name.title()} — {format_confidence(conf)}")

    info = get_flower_info(flower, flower_info_lookup)
    with st.container(border=True):
        st.subheader(f"About {display_name}")
        if info:
            st.write(info.get("description", ""))
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Scientific name**  \n{info.get('scientific_name', '—')}")
                st.markdown(f"**Family**  \n{info.get('family', '—')}")
                st.markdown(f"**Blooming season**  \n{info.get('blooming_season', '—')}")
            with col2:
                st.markdown(f"**Native region**  \n{info.get('native_region', '—')}")
                st.markdown(f"**Colors**  \n{', '.join(info.get('colors', [])) or '—'}")
                st.markdown(f"**Uses**  \n{', '.join(info.get('uses', [])) or '—'}")
        else:
            st.info(f"No reference info found for '{display_name}' yet — add an entry in data/flower_info.json.")

    # --- Grad-CAM section -----------------------------------------------
    with st.container(border=True):
        st.subheader("🔍 What the model looked at")
        st.caption("Red / Yellow = areas the model focused on most · Blue = least important")

        if st.session_state.gradcam_result is None:
            with st.spinner("Computing Grad-CAM..."):
                try:
                    img_arr   = predictor.preprocess(st.session_state.image)
                    conv_m, pre_m, W, b = _build_gradcam_cached(predictor.model)
                    heatmap   = compute_heatmap(img_arr, conv_m, pre_m, W, b)
                    # img_arr is float32 0-255; squeeze + cast to uint8 for display
                    resized   = np.squeeze(img_arr).astype(np.uint8)
                    heat_rgb, overlay = apply_heatmap(resized, heatmap)
                    st.session_state.gradcam_result = {
                        "resized":  resized,
                        "heat_rgb": heat_rgb,
                        "overlay":  overlay,
                    }
                except Exception as e:
                    st.session_state.gradcam_result = {"error": str(e)}

        gcam = st.session_state.gradcam_result
        if "error" in gcam:
            st.warning(
                f"Grad-CAM couldn't run: {gcam['error']}. "
                f"This usually means the model architecture isn't the standard "
                f"EfficientNetB0 + head layout. Everything else still works fine."
            )
        else:
            c1, c2, c3 = st.columns(3)
            c1.image(gcam["resized"],  caption="Original (224×224)",  width='stretch')
            c2.image(gcam["heat_rgb"], caption="Heatmap",             width='stretch')
            c3.image(gcam["overlay"],  caption="Overlay",             width='stretch')

            buf = io.BytesIO()
            Image.fromarray(gcam["overlay"]).save(buf, format="PNG")
            st.download_button(
                "⬇ Download overlay",
                data=buf.getvalue(),
                file_name=f"gradcam_{flower.lower().replace(' ', '_')}.png",
                mime="image/png",
            )

    st.subheader(f"💬 Ask anything about {display_name}")

    if rag_error:
        st.warning(
            f"The RAG chatbot isn't available right now ({rag_error}). "
            f"Everything above still works — see the README to enable Q&A."
        )
    else:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                if turn["sources"]:
                    tags = " ".join(f'<span class="pill pill-source">{s}</span>' for s in turn["sources"])
                    st.markdown(tags, unsafe_allow_html=True)

        question = st.chat_input(f"Ask something about {display_name}...")
        if question:
            with st.spinner("Searching the knowledge base..."):
                try:
                    result = rag.answer(display_name, question)
                    st.session_state.chat_history.append(
                        {"question": question, "answer": result["answer"], "sources": result["sources"]}
                    )
                except Exception as e:
                    st.session_state.chat_history.append(
                        {"question": question, "answer": f"Couldn't get an answer: {e}", "sources": []}
                    )
            st.rerun()

    st.divider()
    if st.button("Try another photo"):
        st.session_state.uploader_key += 1  
        st.session_state.file_fingerprint = None
        st.session_state.image = None
        reset_downstream_state()
        st.rerun()
