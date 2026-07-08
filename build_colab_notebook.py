import json
import os

files_to_write = [
    "requirements.txt",
    "config.py",
    "src/pdf_processor.py",
    "src/chunking.py",
    "src/embeddings.py",
    "src/vector_db.py",
    "src/retriever.py",
    "src/rag_pipeline.py",
    "src/summarizer.py",
    "app.py"
]

cells = []

# Header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🧠 Research Paper Intelligence Engine (Standalone Colab Version)\n",
        "\n",
        "This notebook is completely self-contained. Run all cells from top to bottom. It will write the source code to the Colab environment, install dependencies, and launch the Streamlit web app."
    ]
})

# Create src directory
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!mkdir -p src\n",
        "!mkdir -p data\n",
        "!mkdir -p documents\n",
        "!mkdir -p embeddings\n",
        "!mkdir -p vector_store"
    ]
})

# Write files
for filepath in files_to_write:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # We need to prepend %%writefile
        cell_source = f"%%writefile {filepath}\n" + content
        
        # Jupyter stores source as a list of lines (optional, but standard)
        lines = [line + "\n" for line in cell_source.split('\n')]
        # Remove trailing newline from the last element if present
        if lines and lines[-1].endswith("\n"):
            lines[-1] = lines[-1][:-1]
            
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

# Install dependencies
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Install Dependencies"
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install -r requirements.txt\n",
        "!npm install localtunnel"
    ]
})

# Endpoint IP
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Get your Endpoint IP\n",
        "Copy the IP address below. You will need to paste it into the localtunnel website."
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import urllib\n",
        "print(\"Password/Endpoint IP for localtunnel is:\", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())"
    ]
})

# Run App
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Launch App\n",
        "Click the `loca.lt` link below and paste the IP address!"
    ]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Research_Engine_Standalone.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully!")
