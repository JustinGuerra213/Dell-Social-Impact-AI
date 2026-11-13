# Saving Texas

Streamlit app for comparing before-and-after flood imagery and producing a cautious visual triage report with a local Qwen vision-language model.

## Features

- Browse linked before/response pairs from [UTSA Flood Images](https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images). Metadata is loaded first; image files download on demand.
- Upload an image pair for a separate comparison.
- Ask a local Qwen vision model about visible household and city impacts, infrastructure hazards, and an illustrative cost range.
- For georeferenced UTSA pairs, fetch active point alerts from NWS and county disaster declarations from FEMA; link historical regional infrastructure, contamination, insurance, and evacuation sources in the report.
- Display citizen and city cost categories separately. Unverified costs remain unpriced rather than being inferred from aerial imagery.
- Build an emergency response checklist from visible hazards and official alerts without generating unsafe road-by-road routing.
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

Research in the report is contextual. FEMA declarations are county-level and do not establish that a particular property was impacted. Historical Harvey contamination, drainage, or cost reports must not be treated as current incidents or site-specific costs. Active alerts are queried from the NWS for the dataset coordinate at analysis time. The report links Ready Harris evacuation information, TCEQ spill reporting, FEMA claims and mold resources, and HCFCD infrastructure sources. Verify current hazards with local emergency management before acting.

## Project layout

```text
app.py                  Streamlit UI
flood_app/models.py     Typed domain objects
flood_app/dataset.py    Hugging Face pair and image access
flood_app/analyzer.py   Local Qwen API client and assessment prompt
flood_app/research.py   NWS alerts, FEMA declarations, and area report sources
```

