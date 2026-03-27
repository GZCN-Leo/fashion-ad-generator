"""
Streamlit app – Fashion Product Ad Generator

Pipelines (from DL_Project_V3.0):
  1. Image Classification   – Fine-tuned ViT  (Leoinhouse/ImagineClassification-finetuned-model)
  2. Image Captioning       – BLIP            (Salesforce/blip-image-captioning-base)
  3. Ad Copy Generation     – Qwen2.5-0.5B   (Qwen/Qwen2.5-0.5B-Instruct)

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import random
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")

import streamlit as st
import torch
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    GenerationConfig,
    pipeline as hf_pipeline,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
VIT_MODEL_ID = "Leoinhouse/ImagineClassification-finetuned-model"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Model loaders (cached so they load only once)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_vit():
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_ID)
    model = ViTForImageClassification.from_pretrained(VIT_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_ID, tie_word_embeddings=False,
    ).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_qwen():
    return hf_pipeline(
        "text-generation", model=QWEN_MODEL_ID,
        torch_dtype=torch.float16, device_map="auto",
    )


# ──────────────────────────────────────────────
# Pipeline functions (identical to DL_Project_V3.0)
# ──────────────────────────────────────────────

def classify_image(image, vit_processor, vit_model):
    inputs = vit_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = vit_model(**inputs).logits
    return vit_model.config.id2label[torch.argmax(logits, dim=-1).item()]


def generate_product_description(image, blip_processor, blip_model, max_length=50):
    inputs = blip_processor(
        image, text="a product on a white background of", return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(
            **inputs, max_length=max_length,
            num_beams=5, repetition_penalty=1.5, do_sample=False,
        )
    return blip_processor.decode(out[0], skip_special_tokens=True).capitalize() + "."


def _clean_caption(caption):
    text = caption.strip().rstrip(" .")

    for p in ["a product on a white background of ",
              "a product photo on a white background of ",
              "a product photo of ", "a photo of ", "a picture of "]:
        if text.lower().startswith(p):
            text = text[len(p):]
            break

    lower = text.lower()
    for s in [" on a white background", " with a white background",
              " against a white background", " in front of a white background",
              " on white background", " white background"]:
        if lower.endswith(s):
            text = text[:len(text) - len(s)]
            break

    m = re.match(r"^a\s+(man|woman|boy|girl)\s+(?:in|wearing|with)\s+",
                 text, re.IGNORECASE)
    if m:
        g = m.group(1).lower()
        text = text[m.end():] + (" for men" if g in ("man", "boy") else " for women")

    return text.strip(" .,").capitalize()


_AD_FEW_SHOTS = [
    ("Footwear", "a pair of men's shoes",
     "Step up your style with these versatile men's shoes."),
    ("Apparel", "a blue dress",
     "Turn heads in this elegant blue dress for any occasion."),
    ("Accessories", "a brown leather handbag",
     "Carry your essentials in style with this classic brown leather handbag."),
    ("Personal Care", "a black bottle",
     "Elevate your grooming routine with this sleek black bottle."),
    ("Footwear", "a pair of black and red sneakers",
     "Bold black and red sneakers built for comfort and style."),
    ("Apparel", "a soft knit sweater",
     "Stay cozy and chic in this soft knit sweater all season long."),
    ("Accessories", "a silver necklace",
     "Complete your look with this stunning silver necklace."),
    ("Personal Care", "a perfume bottle",
     "Discover your signature scent with this elegant perfume bottle."),
]


def generate_product_ad(category, description, text_gen):
    clean_desc = _clean_caption(description)

    shots = list(_AD_FEW_SHOTS)
    random.Random(clean_desc).shuffle(shots)

    examples = "\n".join(
        f"Category: {c} | Description: {d} \u2192 {a}" for c, d, a in shots)
    system_prompt = (
        "You write one-sentence product advertisements for an "
        "e-commerce fashion store. Given a product CATEGORY and "
        "DESCRIPTION, write exactly ONE short, appealing sentence "
        "that highlights the product's key feature. Rules:\n"
        "- ONE sentence only, no more.\n"
        "- Focus on the product, NOT people or backgrounds.\n"
        "- Keep it under 20 words.\n"
        "- Make it sound attractive to shoppers.\n\n" + examples
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Category: {category} | Description: {clean_desc} \u2192"},
    ]
    gen_config = GenerationConfig(
        max_new_tokens=40, do_sample=False,
        repetition_penalty=1.5, no_repeat_ngram_size=3,
    )
    raw = text_gen(messages, generation_config=gen_config)
    first_line = raw[0]["generated_text"][-1]["content"].strip().split("\n")[0].strip()
    if first_line and first_line[-1] not in ".!":
        first_line += "."
    return first_line


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Fashion Ad Generator",
    page_icon="\U0001F6CD\uFE0F",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align:center;'>"
    "\U0001F6CD\uFE0F Fashion Product Ad Generator"
    "</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style='text-align:center; color:grey; font-size:1.05rem;'>
    Automate product advertisement copy from a single image — powered by
    three deep-learning pipelines.
    </p>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to use", expanded=False):
    st.markdown(
        """
        This app helps you quickly generate marketing content for your products:

        1. **Product Category** — automatically identifies what type of product it is.
        2. **Product Description** — generates a concise description of the product.
        3. **Ad Copy** — creates a one-sentence advertisement ready for your listing.

        **Getting started:** Upload a clear product image below, then click **Generate**.
        For best results, use a photo with a plain background.

        **Current support:** This app currently supports 4 categories only:
        Apparel, Accessories, Footwear, and Personal Care.
        """
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, PNG, WEBP. Best results with a clear product photo on a plain background.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

generate_clicked = st.button(
    "\u26A1 Generate",
    type="primary",
    use_container_width=True,
    disabled=(uploaded_file is None),
)

if generate_clicked and uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    try:
        with st.status("Running pipelines — this may take a moment on first launch...",
                        expanded=True) as status:

            st.write("\U0001F4E6 Loading classification model (ViT)...")
            vit_proc, vit_mdl = load_vit()

            st.write("\U0001F4F7 Loading captioning model (BLIP)...")
            blip_proc, blip_mdl = load_blip()

            st.write("\U0001F4DD Loading ad-copy model (Qwen)...")
            qwen_gen = load_qwen()

            st.write("\U0001F50D Classifying product...")
            category = classify_image(image, vit_proc, vit_mdl)

            st.write("\U0001F4AC Generating description...")
            raw_caption = generate_product_description(image, blip_proc, blip_mdl)
            clean_caption = _clean_caption(raw_caption)

            st.write("\u2728 Crafting advertisement...")
            ad_copy = generate_product_ad(category, clean_caption, qwen_gen)

            status.update(label="All pipelines complete!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"Something went wrong during generation: {e}")
        st.stop()

    st.divider()
    st.subheader("Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Category", value=category)
    with col2:
        st.metric(label="Words in Ad", value=len(ad_copy.split()))

    st.markdown("**Product Description**")
    st.info(clean_caption)

    st.markdown("**Advertisement Copy**")
    st.success(ad_copy)

    st.divider()
    st.caption(
        "Models: ViT (Leoinhouse/ImagineClassification-finetuned-model) · "
        "BLIP (Salesforce/blip-image-captioning-base) · "
        "Qwen2.5-0.5B-Instruct"
    )

elif generate_clicked and uploaded_file is None:
    st.warning("Please upload a product image first.")
