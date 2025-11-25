# lipid-diffusion

A denoising diffusion probabilistic model (DDPM) for generating new lipid conformations from a library of existing structures.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the example with synthetic data:

```bash
python run.py
```

### Using Your Own Data

Edit `run.py` and uncomment the file loading section:

```python
file_paths = [
    'data/popc_conf_001.xyz',
    'data/popc_conf_002.xyz',
    # ... add more files
]
coords_normalized = preprocessor.process_dataset(file_paths, format='xyz')
```

Supported formats: XYZ, PDB, NPY

## Repository Structure

- `run.py` - Main training script
- `lipid_diffusion/` - Core package
  - `data/` - Data loading and preprocessing
  - `models/` - Model architectures
  - `training/` - Training and sampling utilities
- `outputs/` - Generated results and checkpoints

## Model Architecture

- **Transformer-based** architecture with self-attention between atoms
- **DDPM** with 1000 timesteps and linear noise schedule
- **Kabsch alignment** for structural superposition

## Output

After training, the following files are saved to `outputs/`:
- `generated_conformations.npy` - Generated lipid structures
- `diffusion_model.pt` - Trained model weights
- `training_loss.png` - Training loss curve

## Extending to Membrane Assembly

This single-lipid model can be extended to full membrane assembly by:
- Adding inter-lipid interaction terms
- Conditioning on membrane geometry
- Implementing packing density constraints
- Adding physics-based energy terms
