# Tacotron: PyTorch Implementation

## Overview

This project is a PyTorch re-implementation of [Tacotron](https://arxiv.org/abs/1703.10135), an end-to-end deep learning model for text-to-speech (TTS) synthesis. Tacotron maps input text directly to mel-spectrograms, which can be converted to human speech, eliminating the need for complex feature engineering and traditional synthesis pipelines.

Our implementation closely follows the original [Tacotron paper](https://arxiv.org/abs/1703.10135) by Google (2017), with modular PyTorch code and detailed explanations.

---

## Features

- **Encoder-decoder architecture** with location-sensitive attention
- Full **CBHG module** (Convolution Bank, Highway networks, BiGRU)
- Implements **Bahdanau-style attention** (location-sensitive)
- PyTorch-based, easy to extend or modify
- Trained on the open-source [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- Includes code for Griffin-Lim vocoder to reconstruct audio
- Jupyter Notebook format for interactive exploration

---

## Project Structure

- `Tacotron_article_implementation.ipynb` &nbsp;—&nbsp; Full code and explanations for data loading, model architecture, training loop, and results visualization.

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/tacotron-pytorch
   cd tacotron-pytorch
   ```

2. **Install dependencies** (recommend using a virtual environment):
   ```bash
   pip install torch numpy matplotlib librosa tqdm
   ```
   - For audio playback and spectrograms: `librosa`
   - For GPU training: [PyTorch with CUDA](https://pytorch.org/get-started/locally/)

3. **Download the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)**  
   Place the audio and metadata files in a folder called `data/LJSpeech-1.1`.

4. **Open the notebook**:
   ```
   jupyter notebook Tacotron_article_implementation.ipynb
   ```
   - Step through the notebook to preprocess data, train the model, and generate samples.
   - Hyperparameters and settings are documented in the notebook.

---

## Dataset

- **LJ Speech Dataset**: 13,100 short English audio clips by a single speaker.
- This project used a subset (approx. 45%) for training due to hardware constraints.
- Data preprocessing and loading steps are in the notebook.

---

## Training

- Training is performed for 50 epochs on a subset of LJ Speech (or adjust as resources allow).
- Hardware: Model was trained using an Nvidia P100 GPU.
- Training and validation losses are tracked per epoch.

---

## Results

- The model successfully learns to align text and audio, with loss and attention plots included in the notebook.
- Audio samples may still sound noisy or unintelligible with limited training (full speech quality requires significantly longer training and larger datasets).
- See notebook for spectrogram visualizations and attention heatmaps.

---

## Limitations & Future Work

- Model performance is limited by available GPU compute and training time.
- Training on a small subset and for limited epochs results in noisy outputs, as expected for TTS models.
- For best results, train for hundreds of epochs on the full dataset with powerful GPUs.

---

## Credits

- Based on [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135) by Google (2017).

---

## License

MIT License.

---
