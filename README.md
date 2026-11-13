# Saving Texas

Streamlit app for comparing before-and-after flood imagery and producing a cautious visual triage report with a local Qwen vision-language model.

## Features

- Browse linked before/response pairs from [UTSA Flood Images](https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images). Metadata is loaded first; image files download on demand.
- Upload an image pair for a separate comparison.
- Ask a local Qwen vision model about visible household and city impacts, infrastructure hazards, and an illustrative cost range.
- Reports contain three tabs: impact, observed changes, and short image-based limitations.
- Download the result as JSON.

The dataset includes 1,204 image metadata rows with explicit `paired_image_id` links. It combines aerial imagery from different dates and sources. A visible change is not proof that flooding caused it. The dataset is licensed CC BY-NC 4.0; commercial use is not allowed under that license.

## Requirements

- Python 3.10 or newer
- A local OpenAI-compatible Qwen vision endpoint, such as vLLM, serving the model `Qwen/Qwen3-VL-8B-Instruct`
- Network access the first time dataset metadata/images are downloaded

The app does not download the model. Set `QWEN_BASE_URL` and `QWEN_MODEL` when your local server uses different values.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export QWEN_BASE_URL=http://127.0.0.1:8000/v1
export QWEN_MODEL=Qwen/Qwen3-VL-8B-Instruct
streamlit run app.py
```

By default Streamlit listens on `0.0.0.0:8501`, allowing access from a reachable network interface. Protect the server with an authenticated reverse proxy or firewall before exposing it beyond a trusted network. Keep the model endpoint private.

## Data and model behavior

The dataset is used for image pair selection and retrieval, not as verified damage labels. The model is instructed to distinguish visible evidence from unknown conditions. It cannot determine personal injuries from imagery and must mark that field as not assessable. Financial output is an illustrative range only; actual repair costs require local inspections, measured quantities, and current unit prices. Do not use the app to certify infrastructure safety or replace emergency response.

Reports use only the selected before-and-after images for triage. The local model does not browse the web or retrieve external incident, cost, or safety sources.

## Project layout

```text
app.py                  Streamlit UI
flood_app/models.py     Typed domain objects
flood_app/dataset.py    Hugging Face pair and image access
flood_app/analyzer.py   Local Qwen API client and assessment prompt
```

