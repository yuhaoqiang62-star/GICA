# GICA: Gate-Controlled Information Bottleneck Cross-Modal Attention Network

A multimodal sentiment analysis model based on information bottleneck theory.

## Overview

Your model description text here. GICA employs the gated information bottleneck 
method with cross-modal attention to form compact and informative latent states...

![Model Architecture](assets/GICA.png)

## Requirements
- Python 3.8+
- PyTorch 1.12+
- transformers 4.x (DeBERTa)

## Project Structure
<pre>
├── GICA.py                  # Core GICA model (bottleneck, attention, fusion)
├── deberta_GICA.py          # DeBERTa integration wrapper
├── train.py                 # Training, evaluation, and ablation scripts
├── global_configs.py        # Dataset-specific dimension configs
├── requirements.txt         # Python dependencies
├── datasets/                # Place CMU-MOSI / CMU-MOSEI .pkl files here
└── saved_models/            # Saved checkpoints (created at runtime)
</pre>



## Quick Start

1. Clone the repository and install dependencies:
git clone https://github.com/yuhaoqiang62-star/GICA.git
pip install -r requirements.txt

2. Download the datasets to `./datasets` by running `download_datasets.sh`. For details, see [here](https://github.com/A2Text/MOSI).

3. Train the model:
python train.py --dataset mosi
python train.py --dataset mosei

## Training Options
- Adjust `--batch_size` to fit memory constraints
- Adjust `--epochs` for training duration
