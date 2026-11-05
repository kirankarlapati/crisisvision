# CrisisVision — Satellite Disaster Damage Assessment & Hotspot Risk Mapping

CrisisVision is a satellite-image-based disaster damage assessment system that analyzes pre- and post-disaster satellite imagery, extracts individual building footprints, performs pixel-level damage classification, generates Gaussian spatial damage hotspots, and provides natural-language damage assessments and interactive incident chat powered by local **Llama 3.2** via **Ollama**.

---

## Prerequisites

1. **Python 3.11+**
2. **Ollama** installed on your system ([Download Ollama](https://ollama.com/download))
3. *(Optional)* **MongoDB** running locally on port `27017` (the application runs and displays all predictions even if MongoDB is offline)

---

## Quick Start Commands

### 1. Start Ollama and Download Llama 3.2

Open a terminal and start the Ollama service (if not already running as a background service):

```powershell
# Start Ollama service
ollama serve
```

In another terminal, ensure the `llama3.2` model is downloaded:

```powershell
ollama pull llama3.2
```

To verify the model is available:

```powershell
ollama list
```

---

### 2. Activate Virtual Environment & Install Dependencies

From the project root directory (`satelitte`):

#### On Windows (PowerShell):
```powershell
# Activate existing virtual environment
.venv\Scripts\Activate.ps1

# (Optional) If packages need to be updated:
pip install -r requirements.txt
```

#### On Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

#### On Linux / macOS:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Run the CrisisVision Web Application

Start the Flask development server:

```powershell
python app.py
```

Once started, open your browser and navigate to:

👉 **http://127.0.0.1:5000**

---

### 4. Run the Test Suite

Execute all 73 automated unit and integration tests:

```powershell
python -m pytest tests/ -v
```

To run individual test modules:

```powershell
# Test change detection & model inference
python -m pytest tests/test_change_detection.py -v

# Test building extraction & coordinate geometry
python -m pytest tests/test_building_extraction.py -v

# Test Gaussian hotspot risk mapping
python -m pytest tests/test_risk_mapping.py -v

# Test Llama 3.2 assistant & chat routes
python -m pytest tests/test_llm_assistant.py -v
```

---

## Usage Workflow

1. Navigate to **http://127.0.0.1:5000**.
2. Upload a **Pre-disaster** satellite image (RGB, e.g., PNG/JPG/TIFF).
3. Upload a **Post-disaster** satellite image of the same area.
4. Click **Analyze Damage**.
5. Inspect:
   - **Interactive Overlay Viewport**: Toggle between Building Damage, Hotspot Heatmap, Centroids & Bounding Boxes, or Base Post-Disaster Imagery.
   - **Executive Damage Explanation**: Generated automatically by Llama 3.2.
   - **Incident Chat Assistant**: Ask follow-up questions about buildings, hotspots, and damage classifications.
   - **Hotspot Priority & Building Tables**: Filter by damage class or priority tier.

---

## Configuration (Environment Variables)

Optional environment variables (can be set in PowerShell or `.env`):

| Variable | Default Value | Description |
|:---|:---|:---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model tag to use in Ollama |
| `OLLAMA_TIMEOUT` | `60.0` | Timeout in seconds for LLM generation |
| `MONGO_URI` | `mongodb://127.0.0.1:27017/` | MongoDB connection URI |
| `MONGO_DATABASE` | `crisis_management` | Database name |
| `SECRET_KEY` | `dev-only-change-this...` | Flask session secret key |

