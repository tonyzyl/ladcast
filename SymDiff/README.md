# SymDiff: Equivariant Diffusion via Stochastic Symmetrisation

Official code release for the paper: [SymDiff: Equivariant Diffusion via Stochastic Symmetrisation](https://arxiv.org/abs/2410.06262).

## Description

We propose SymDiff, a method for constructing equivariant diffusion models using the recently introduced framework of stochastic symmetrisation. Notably, in contrast to previous work, SymDiff typically does not require any neural network components that are intrinsically equivariant, avoiding the need for complex parameterisations and the use of higher-order geometric features. Instead, our method can leverage highly scalable modern architectures, such as Diffusion Transformers, as drop-in replacements for these more constrained alternatives, where we demonstrate that this additional flexibility yields significant empirical benefit on E(3)-equivariant molecular generation tasks.

This codebase is heavily based on the [EDM codebase](https://github.com/ehoogeboom/e3_diffusion_for_molecules) and the official [DiT implementation](https://github.com/facebookresearch/DiT/tree/main). We provide an implementation of our example architecture in `sym_nn/sym_nn` which we use as a drop-in replacement for the EGNN model used in the original EDM framework.

## Installation

To download QM9, GEOM-Drugs and install RDKit, see the instructions from the EDM codebase. For further dependencies, see `requirements.txt`.

## Training:

We provide training scripts for our experiments in `scripts`.

## Testing

To analyze the sample quality of generated molecules and test NLL

```python eval_analyze.py --model_path outputs/YOUR_EXP_NAME --n_samples 10000 --datadir YOUR_DATADIR```

We provide our pretrained SymDiff model for QM9 [here](https://drive.google.com/drive/folders/1QfgBrTZnGY0mx6ATamzXyUjHQI0dZQil?usp=sharing). To evaluate this model,
download the SymDiff folder, place it in `./outputs/` and run the above command with `YOUR_EXP_NAME=SymDiff`.

## Citations

If you find this work useful, you can cite us by

```
@misc{zhang2024symdiffequivariantdiffusionstochastic,
      title={SymDiff: Equivariant Diffusion via Stochastic Symmetrisation}, 
      author={Leo Zhang and Kianoosh Ashouritaklimi and Yee Whye Teh and Rob Cornish},
      year={2024},
      eprint={2410.06262},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.06262}, 
}
```

## Licences

This repo is licensed under the [MIT License](https://opensource.org/license/mit/).