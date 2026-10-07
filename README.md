# Gaussian Splatting

Implementation of a differentiable Gaussian splatting pipeline for fitting 2D images and reconstructing a 3D scene from posed images. Includes adaptive densification, alpha compositing, camera projection, and PSNR evaluation.

## Files

- `gaussian.py` — Implements Gaussian covariances, pixel weights, alpha compositing, quaternion rotations, and 3D camera projection.
- `densification.py` — Implements cloning, splitting, and pruning for 2D Gaussians.
- `densification_3d.py` — Implements density control for 3D Gaussians.
- `train_2d.py` — Initializes and trains 2D Gaussians and runs image-fitting experiments.
- `train_3d.py` — Loads camera poses, trains the 3D reconstruction, and evaluates training and held-out views.
- `novel_view_evaluate.py` — Loads a saved 3D model and generates held-out comparisons and novel-view renders.
- `utils.py` — Provides device selection, image loading/saving, and PSNR calculation.
- `data/` — Contains the coffee, astronaut, and cat target images.
- `spheres/` — Contains `cameras.json` and the provided `train/` and `val/` images.
- `results/` — Contains generated renders, checkpoints, plots, and evaluation results.

## Reproducing Results

Install the required Python packages:

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy pillow matplotlib
```

Run all commands from the repository root.

### 2D Image Fitting

Configure the experiment in `train_2d.py`, then run:

```bash
python -u train_2d.py
```

The 2D experiments use 128×128 images, 2000 optimization steps, Adam with a learning rate of `0.01`, and seed `0`.

For the P4 densification comparison:

- Start with 256 Gaussians and a budget of 1024.
- Use a position-gradient threshold of `1e-5`.
- Densify every 200 steps, excluding the final step.
- Compare against a fixed-count fit using the adaptive run’s final count.

For the P5 quality-versus-count experiment, select `run_p5()` in the script’s execution block. Each target is independently fitted using 256, 512, and 1024 Gaussians without densification.

### 3D Reconstruction

The sphere dataset should have this structure:

```text
spheres/
├── cameras.json
├── train/
└── val/
```

Select the plain P7 experiment using `main()` in `train_3d.py`, then run:

```bash
python -u train_3d.py
```

The reported P7 model uses 4000 Gaussians, 1500 optimization steps, and Adam with a learning rate of `0.01`. Gaussian centers are initialized uniformly in `[-1.5, 1.5]^3`, with initial scales of `0.08`, identity quaternions, gray colors, and low opacity.

For P8, select `run_p8()` in the execution block. It loads the saved P7 checkpoint and trains an adaptive model starting with 1000 Gaussians and a budget of 4000. Use a gradient threshold of `2e-4`, a world-space size threshold of `0.08`, a split scale factor of `1.6`, and an opacity-pruning threshold of `0.005`. Adam is restarted after each density-control pass.

### Novel-View Evaluation

After saving the P7 checkpoint, run:

```bash
python -u novel_view_evaluate.py
```

Check that the checkpoint path points to the model being evaluated. This script evaluates the held-out cameras and renders additional novel viewpoints without retraining.

## Evaluation Output

Training prints PSNR at regular intervals. For 3D training, these progress values correspond to the currently sampled camera.

Final evaluation reports:

- Full-image PSNR for each 2D fit.
- PSNR versus Gaussian count for all three targets.
- Mean PSNR across 44 training views and 12 held-out views.
- Final Gaussian counts for plain and densified models.

Renders, comparison images, plots, metrics, and model checkpoints are saved under `results/`. Additional novel views have no reference images and are evaluated visually.
