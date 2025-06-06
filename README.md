# SlideGelSight Training Pipeline

This repository contains scripts to train multimodal neural networks on GelSight tactile data.

## Contents
- `flow_processor.py` – Extracts spatial and global deformation features from flow JSON files.
- `flow_networks.py` – Light‐weight CNN/MLP modules for processing flow features.
- `multimodal_dataset.py` – Loads synchronized image and flow sequences.
- `multimodal_network.py` – Encodes image and flow sequences and fuses them for classification.
- `split_dataset.py` – Utility to split raw data into training and testing sets.
- `train_multimodal.py` – High level training script.
- `test_imports.py` – Sanity check that all modules import correctly.

## Preparing the Data
1. Arrange your raw dataset as:
   ```
   dataset_root/
       material_1/
           cycle_001/
               frame_000.png
               frame_001.png
               ...
               flow_000.json
               flow_001.json
               ...
       material_2/
           cycle_001/
           ...
   ```
   Each `material_X/cycle_Y` directory should contain a sequence of PNG frames and their corresponding flow JSON files.

2. Run the dataset splitting utility. This will create `train` and `test` folders containing symlinks (default) or copies of the data and also produces file lists.
   ```bash
   python split_dataset.py --root path/to/dataset_root --output dataset_split --ratio 0.9
   ```
   Adjust `--ratio` to change the train/test split and use `--mode copy` if you prefer actual copies instead of symlinks.

## Training
1. Verify that required Python packages are installed (PyTorch, torchvision, NumPy, pandas, scikit‑learn, matplotlib, seaborn, tqdm). Install them with pip if necessary:
   ```bash
   pip install torch torchvision torchaudio numpy pandas scikit-learn matplotlib seaborn tqdm
   ```

2. Edit the configuration dictionary at the bottom of `train_multimodal.py` if you need to modify hyperparameters or paths. By default it expects the split data to be located in `./dataset_split`.

3. Start training:
   ```bash
   python train_multimodal.py
   ```
   Model checkpoints and training curves will be saved under `./results/` with a timestamped folder name.

## Notes
- `test_imports.py` can be run to quickly check that all modules are discoverable:
  ```bash
  python test_imports.py
  ```
- Training assumes a CUDA‑capable GPU but will fall back to CPU if unavailable (slower).

