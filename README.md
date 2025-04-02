# Mix Master: Natural Language to Professional Mixing Terminology Tool

## User Guide

### 1. Introduction

Mix Master is a tool designed specifically for mixing beginners, capable of transforming everyday language expressions into professional mixing terminology and parameters. Whether you're a newcomer to audio production or a music enthusiast looking to enhance your mixing skills, this tool helps you easily express your mixing needs and obtain professional advice.

### 2. Installation Guide

#### System Requirements
- Python 3.9 or higher
- Stable internet connection (for model access)

#### Core Dependencies
```bash
# Install core dependencies
pip install streamlit      # For building interactive user interfaces
pip install dashscope      # For AI model access
pip install pandas         # For data processing
pip install torch          # For deep learning model support
```

#### Installation Steps

**Method 1: Using conda**
```bash
# Clone the repository
git clone https://github.com/YottaCheng/MixMaster-Finetune.git
cd MixMaster-Finetune

# Create and activate environment
conda env create -f environment.yml
conda activate mixmaster
```

**Method 2: Using pip**
```bash
# Clone the repository
git clone https://github.com/YottaCheng/MixMaster-Finetune.git
cd MixMaster-Finetune

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage Guide

#### Launching the Application
```bash
# Use absolute path to app_streamlit.py
streamlit run /path/to/MixMaster-Finetune/scripts/web/app_streamlit.py
```
After executing the above command, the application will automatically open in your default browser, typically at http://localhost:8501

#### Basic Workflow

1. **Enter Natural Language Description**
   - Input your description of the sound in the text box, for example: "Make the drums more powerful" or "Make the vocals sound warmer"

2. **Get Professional Conversion Results**
   - Click the "Convert" button, and the system will process your description
   - Wait a few seconds, and professional terminology and suggested parameters will appear in the results area

3. **Use the Conversion Results**
   - Click the "Copy" button to copy the results to your clipboard
   - Apply the suggested parameters in your mixing software

### 4. Feature Details

#### Natural Language Conversion
The system can understand and convert various mixing-related natural language expressions, such as:
- "Make the guitar sound brighter" → "Boost 3-5kHz range by 2-3dB, slightly cut 250-500Hz"
- "Reduce vocal harshness" → "Apply light 2:1 compression in the 4-6kHz range, threshold set to -15dB"
- "Give the drums more impact" → "Enhance drum transients, attack time set to 5-10ms, release time 50ms"

#### Dataset Inspection Tool
The label_checker.py in the scripts/data_prepare directory is a tool for checking and validating training datasets, primarily intended for project developers.

### 5. Custom Model Path

If you need to use a custom model, modify the model path in the `predict.py` file:

1. Open the file: `/path/to/MixMaster-Finetune/scripts/web/predict.py`
2. Find the `class MixingLabelPredictor:` section
3. Modify the initialization method's model path to your local path:
   ```python
   def __init__(self, model_dir="/your/local/path/to/model"):
   ```