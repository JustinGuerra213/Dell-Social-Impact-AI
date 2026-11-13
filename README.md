# Saving Texas

Streamlit app for comparing before-and-after flood imagery and producing a cautious visual triage report with a local Qwen vision-language model.

## Features

- Browse linked before/response pairs from [UTSA Flood Images](https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images). Metadata is loaded first; image files download on demand.
- Ask a local Qwen vision model about visible household and city impacts, infrastructure hazards, and approximate image-based impact metrics.
- The triage report includes impact, observed changes, limitations, metrics, and area map tabs.
- Explore daily temperature and rainfall around each image date in an interactive JavaScript chart, using the pair's dataset coordinates.
- View the selected dataset coordinate on an interactive JavaScript area map in the triage report.
- Download the result as JSON.

The dataset includes 1,204 image metadata rows with explicit `paired_image_id` links. It combines aerial imagery from different dates and sources. A visible change is not proof that flooding caused it. The dataset is licensed CC BY-NC 4.0; commercial use is not allowed under that license.

## Requirements

- Python 3.10 or newer
- A local OpenAI-compatible Qwen vision endpoint, such as vLLM, serving the model `Qwen/Qwen3-VL-8B-Instruct`
- Network access the first time dataset metadata/images are downloaded
- Network access to retrieve historical weather context from Open-Meteo

The app does not download the model. Set `QWEN_BASE_URL` and `QWEN_MODEL` when your local server uses different values.
The metrics tab reports cautious visual percentage ranges for the visible image footprint and visible road segments, plus Qwen's confidence and evidence basis. These are not surveyed measurements. When coordinates and image dates are available, the app retrieves a 30-day historical temperature and rainfall distribution from Open-Meteo and supplies it to Qwen as context; weather values do not establish visible flood extent or road damage. It uses Open-Meteo's default Best Match gridded dataset, which combines available IFS, ERA5, and ERA5-Land data; the model and grid resolution can vary by date and location. These are regional reanalysis estimates, not local gauge measurements. See the [Historical Weather API documentation](https://open-meteo.com/en/docs/historical-weather-api). If weather data is unavailable, Qwen uses only the before/response images. The area map uses Leaflet and OpenStreetMap tiles, shows the coordinate supplied by the dataset, and does not require a Google Maps API key. OpenStreetMap requires visible attribution and has a [tile usage policy](https://operations.osmfoundation.org/policies/tiles); use a dedicated tile provider if the app's usage grows beyond interactive evaluation.

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

The local model does not browse the web. The app can retrieve historical temperature and rainfall from Open-Meteo using the selected pair's coordinates and date, then include those readings as limited context. It does not retrieve external incident, cost, or safety reports.

## Project layout

```text
app.py                  Streamlit UI
flood_app/models.py     Typed domain objects
flood_app/dataset.py    Hugging Face pair and image access
flood_app/analyzer.py   Local Qwen API client and assessment prompt
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module responsibilities, service flow, deployment tunnel, and network boundaries.

