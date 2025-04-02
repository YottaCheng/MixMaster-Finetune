# Mix Master

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview

Mix Master is a natural language conversion tool designed for mixing beginners that transforms casual expressions into professional mixing terminology and parameters.

- **Background**: Currently there is a lack of tools in the mixing field that convert ordinary language into professional terminology, making it especially difficult for beginners
- **Core Function**: Transform natural language descriptions from mixing beginners into professional mixing parameters and terminology
- **Target Users**: Mixing beginners, music production enthusiasts, creators without professional mixing knowledge
- **Problem Solved**: Helps users accurately express their mixing needs and lowers the barrier to entry for mixing technology

## Features

- **Natural Language Conversion**: Analyzes non-professional requirement expressions from mixing beginners, such as converting "make my voice sound sweeter" into "moderate gain in the high frequency range (8kHz-12kHz) to enhance brightness and clarity"
- **Professional Model Support**: Fine-tuned based on DeepSeek Distill and Qwen 1.5B models, specifically optimized for terminology recognition and conversion in the mixing field
- **User-Friendly Interface**: Provides a clean and intuitive user interface through Streamlit, allowing users to get professional advice without complex operations
- **Real-time Conversion**: Quickly responds to user input, providing immediate professional terminology conversion results
- **Copyable Results**: Generated professional terminology and parameters can be copied with one click for use in mixing software

## Technology Stack

- **Programming Language**: Python
- **Frontend Framework**: Streamlit
- **Backend Framework**: Streamlit
- **AI Models**: DeepSeek-R1-1.5B-Distill, Qwen 1.5B (fine-tuned versions)
- **Other Tools/Libraries**:
  - transformers
  - PEFT
  - bitsandbytes
  - torch
  - numpy
  - pandas

## System Architecture

Mix Master employs a streamlined modular design, primarily consisting of the following components:

- **User Interface Layer**: Interactive interface built with Streamlit
- **Natural Language Processing Layer**: Processes and parses user-input mixing requirements
- **Model Inference Layer**: Uses fine-tuned language models to convert requirements into professional terminology
- **Data Processing Layer**: Processes model output and formats it into user-friendly responses

```
┌─────────────────────┐
│  User Interface (Streamlit)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Natural Language Processing │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Fine-tuned LLM (DeepSeek/Qwen) │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Professional Term Conversion  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Result Display & Export  │
└─────────────────────┘
```

## Installation Guide

### Requirements

- **Operating System**: macOS / Windows / Linux
- **Runtime Environment**: Python 3.8+
- **Hardware Requirements**:
  - Recommended: Dedicated GPU with CUDA support (VRAM ≥ 8GB)
  - Minimum: Supports basic CPU inference (slower processing)

### Dependency Installation

```bash
# Clone the project
git clone https://github.com/your-username/mix-master.git
cd mix-master

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Environment Variables**:
   Create a `.env` file and set necessary environment variables:
   ```
   MODEL_PATH=/path/to/model/weights
   DEVICE=cuda  # or cpu
   ```

2. **Model Weights Download**:
   - Automatic: Fine-tuned model weights will be downloaded automatically on first run
   - Manual: Model weights can be downloaded from the [release page](link) and placed in the `models/` directory

3. **Logging Configuration**:
   - Log files are saved in the `logs/` directory
   - Log levels and output methods can be adjusted by modifying `config/logging.yaml`

## Usage Guide

### Starting the Project

```bash
# Navigate to scripts/web directory
cd scripts/web

# Launch Streamlit application
streamlit run app.py
```

After launching, your browser will automatically open http://localhost:8501, displaying the Mix Master web interface.

### Basic Operation Flow

1. Enter your mixing requirements in the text input box, such as "make the vocals clearer and more prominent"
2. Click the "Generate Professional Terminology" button
3. View the generated professional terminology and parameter suggestions
4. Click the "Copy to Clipboard" button to copy the results for reference in your mixing software

### Example Scenarios

#### Scenario 1: Adjusting Vocal Tone

User input:
```
I want my vocals to have more space and thickness
```

Mix Master output:
```
Recommended vocal processing:
1. Add moderate reverb with pre-delay of 20-30ms and decay time of 1.2-1.8 seconds
2. Use a stereo widener set to 120-140% width
3. Boost 2-3dB in the 300-500Hz range to add warmth and thickness
4. Consider adding slight delay effects, set to eighth or sixteenth note rhythm-synced delay
```

#### Scenario 2: Processing Drum Sounds

User input:
```
The drums sound too muffled, I want them to have more power and impact
```

Mix Master output:
```
Recommended drum processing:
1. Add 2-3dB at 60-80Hz on the kick drum to enhance low-frequency energy
2. Boost 3-4dB in the 2-4kHz range to increase attack characteristics
3. Apply compression with a ratio of 4:1, fast attack (5-10ms) and medium release (80-150ms)
4. Consider parallel compression to maintain dynamics while adding punch
```

## Project Structure

```
mix-master/
│
├── docs/                 # Documentation
├── src/                  # Source code
├── scripts/              # Scripts
│   └── web/              # Web application
│       └── app.py        # Streamlit application entry point
├── models/               # Model weights and configurations
├── tests/                # Test files
├── config/               # Configuration files
├── .gitignore
├── requirements.txt
└── README.md
```

## Testing

To test if the model is working correctly:

```bash
# Navigate to the test directory
cd scripts/web

# Run the test script
python test_model.py
```

This will run a series of predefined test cases to verify that the model is correctly converting natural language to professional mixing terminology.

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2025-04-01 | Initial release |
| 0.9.0 | 2025-03-15 | Beta version |

## Contact Information

- Project Lead: [Your Name]
- Email: [your.email@example.com]
- Project Team: [Team Name/Link]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- DeepSeek AI for the base model
- Qwen team for their language model
- All contributors and testers who helped improve this project