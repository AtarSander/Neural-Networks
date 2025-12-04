# Neural Networks Labs

Collection of university lab assignments exploring neural network models across tabular, vision, text, and generative tasks. Each lab ships both notebooks (`exercise.ipynb`) and lightweight Python modules for reproducible runs and final `pred.csv` submissions.

## Repository layout
- `Lab1 - MLP tabular classifier`: Multilayer perceptron for housing affordability classification (cheap/average/expensive) from tabular features. Includes `preprocessing.py` for cleaning/encoding, `model.py` for the MLP, `experiment.py` to train/evaluate, and final predictions in `pred.csv`.
- `Lab2 - ResNet image classifier`: Custom residual CNN (no off-the-shelf backbone) for a 50-class image dataset. `preprocessing.py` defines torchvision transforms/loaders, `model.py` implements residual blocks, `experiment.py` handles training/checkpoints, with submission files `pred.csv`.
- `Lab3 - Image GenAI comparison`: Side-by-side implementation of VAE (`vae.py`), GAN (`gan.py`), and conditional diffusion (`diffusion_model.py`) on 32×32 traffic sign images (`data/trafic_32`). Utilities in `utils.py` cover visualization, conditioning vectors, and FID measurement; generated samples and weights live under `weights` and `generated_images.pt`.
- `Lab4 - RNN text classifier`: Sequence classifier over pre-tokenized sequences (`train.pkl`, `test_no_target.pkl`) using LSTM/GRU variants in `model.py`, dataloaders/utilities in `utils.py`, and inference outputs in `pred.csv`.
- `Lab5 - Finetuning with LoRA`: Parameter-efficient finetuning of transformer text classifiers with Hugging Face `transformers` + `peft`. `utils.py` wraps dataset handling, LoRA configuration, custom weighted loss trainer, and accuracy reporting; checkpoints and experiment outputs are stored in the `lora_results*` directories with `pred.csv` for submissions.

## Tech stack
- Python 3.x
- PyTorch, Torchvision, Torchmetrics
- NumPy, scikit-learn, Matplotlib
- Hugging Face Transformers + PEFT (LoRA)
