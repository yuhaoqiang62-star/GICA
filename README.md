# GICA: Gated Information Bottleneck Network with Cross-Modal Attention

A multimodal sentiment analysis model based on information bottleneck theory.

## Requirements
- Python 3.8+
- PyTorch 1.12+
- transformers 4.x (DeBERTa)

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
