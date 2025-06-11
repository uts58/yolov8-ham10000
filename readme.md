# YOLOv8-Based Deep Learning Approach for Real-Time Skin Lesion Classification Using the HAM10000 Dataset

This repository contains code for the paper titled `YOLOv8-Based Deep Learning Approach for Real-Time Skin Lesion Classification Using the HAM10000 Dataset`. The ISIC2018 Task3 dataset (HAM10000) is used for this paper.

## Dataset

The project uses the ISIC2018 Task3 dataset (HAM10000), which contains skin lesion images classified into 7 different categories:
- Melanoma
- Melanocytic nevus
- Basal cell carcinoma
- Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
- Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)
- Dermatofibroma
- Vascular lesion

## Project Structure

```
├── data_processing.py       # Script for processing ISIC2018 dataset
├── train.py                 # Script for training YOLOv8 models
├── test.py                  # Script for evaluating trained models
├── train_with_aug.log       # Training log with data augmentation
├── train_without_aug.log    # Training log without data augmentation
├── result_with_aug/         # Results directory for models trained with augmentation
├── result_without_aug/      # Results directory for models trained without augmentation
└── runs/                    # Directory containing visualization results
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/skin-cancer-classification.git
cd skin-cancer-classification
```

2. Install the required packages:
```bash
pip install ultralytics scikit-learn matplotlib seaborn pandas
```

## Usage

### Data Preparation

The `data_processing.py` script organizes the ISIC2018 dataset into the appropriate directory structure:

```bash
python data_processing.py
```

### Training

To train YOLOv8 models on the skin cancer dataset:

```bash
python train.py
```

This script trains five different YOLOv8 model sizes (nano, small, medium, large, and extra large) on the dataset for 30 epochs.

### Evaluation

To evaluate the trained models:

```bash
python test.py
```

This script generates classification reports and confusion matrices for each model.

## Models

The project uses the following YOLOv8 classification models:
- YOLOv8n (nano): 1.4M parameters
- YOLOv8s (small): ~6M parameters
- YOLOv8m (medium): ~17M parameters
- YOLOv8l (large): ~37M parameters
- YOLOv8x (extra large): ~68M parameters

## Results

The results are saved in the following directories:
- `result_with_aug/`: Contains models trained with data augmentation
- `result_without_aug/`: Contains models trained without data augmentation

For each model, the following files are generated:
- Classification report (CSV)
- Confusion matrix (PNG)

## Citation
Cite our paper by
```
@INPROCEEDINGS{10880715,
  author={Saha, Utsha and Ahamed, Imtiaj Uddin and Imran, Md Ashique and Ahamed, Imam Uddin and Hossain, Al-Amin and Gupta, Ucchash Das},
  booktitle={2024 IEEE International Conference on E-health Networking, Application & Services (HealthCom)}, 
  title={YOLOv8-Based Deep Learning Approach for Real-Time Skin Lesion Classification Using the HAM10000 Dataset}, 
  year={2024},
  volume={},
  number={},
  pages={1-4},
  keywords={Measurement;Deep learning;Accuracy;Computational modeling;Skin;Real-time systems;Lesions;Ensemble learning;Skin cancer;Diseases;Skin Cancer;YOLOv8-Classification;Skin Lesion Classification;HAM10000 Dataset;Transfer Learning;Deep Learning},
  doi={10.1109/HealthCom60970.2024.10880715}}
```

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ISIC2018 Challenge](https://challenge.isic-archive.com/landing/2018/)
