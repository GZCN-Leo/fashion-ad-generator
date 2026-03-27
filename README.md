# Fashion Product Ad Generator

A Streamlit web app that automatically generates **product category**, **description**, and **one-sentence advertisement copy** from a single fashion product image — powered by three deep-learning pipelines.

## Pipelines

| Step | Task | Model | Source |
|------|------|-------|--------|
| 1 | Image Classification | Fine-tuned ViT | [Leoinhouse/ImagineClassification-finetuned-model](https://huggingface.co/Leoinhouse/ImagineClassification-finetuned-model) |
| 2 | Image Captioning | BLIP (zero-shot) | [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| 3 | Ad Copy Generation | Qwen2.5-0.5B-Instruct (few-shot) | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

**Supported categories:** Apparel · Footwear · Accessories · Personal Care

## How It Works

1. Upload a product image (JPG / PNG / WEBP).
2. Click **Generate**.
3. The app classifies the product, generates a natural-language description, and produces a short advertisement sentence.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies
- A GPU is recommended for faster inference (CPU works but is slower)

## Project Context

This app is built as part of the **ISOM 5240 Deep Learning Business Applications with Python** course project. The ViT classifier is loaded from [Leoinhouse/ImagineClassification-finetuned-model](https://huggingface.co/Leoinhouse/ImagineClassification-finetuned-model), fine-tuned on the [Fashion Product Images (Small)](https://huggingface.co/datasets/ashraq/fashion-product-images-small) dataset with balanced sampling across 4 categories (2 000 images each).

## License

For academic use.
