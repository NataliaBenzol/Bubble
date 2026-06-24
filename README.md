# Bubble 
# Bubble Concentration Classification Pipeline

This project implements a computer vision pipeline for the automatic classification of bubble concentration in liquid solutions based on video recordings. The system extracts frames from videos, filters out irrelevant frames, and trains convolutional neural networks to classify solutions into seven concentration categories: 0%, 5%, 12.5%, 25%, 50%, 75%, and 100%.

The architecture is built on the principle of separation of concerns. The domain layer in `src/bubble/domain/` describes the core business entities, including concentrations, image sizes, and frame samples. The repository layer in `src/bubble/repositories/` encapsulates file system and video operations through the `FrameSource` and `ImageRepository` protocols. The `src/bubble/ml/` module handles data models, neural network architectures, training, and evaluation. Reporting functionality is isolated in `src/bubble/reporting/`, which manages CSV metric exports and plot generation.

The pipeline consists of four sequential stages. The `extract` command retrieves frames from videos at a uniform step, skipping dark and irrelevant frames. The `filter` command removes completely black frames and images without visible bubbles using OpenCV morphological operations. The `resize` command standardizes all images to 224x224 pixels while preserving aspect ratios and adding black padding where necessary. Finally, the `train` command trains the models using transfer learning on ImageNet-pretrained weights.

Training utilizes two architectures: ResNet50 and VGG16. Both models are frozen upon loading, and only the classification head is replaced. ResNet50 receives a new head consisting of two fully connected layers with dropout, while VGG16 replaces only the final layer of its classifier. The data is split chronologically in a 70/15/15 ratio for training, validation, and testing, which prevents data leakage between adjacent video frames. Training employs early stopping with a configurable patience parameter, defaulting to 10 epochs. Random seeds are fixed to ensure result reproducibility.

The performance results of the two trained models are presented below.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| ResNet50 | 81.01% | 83.27% | 81.01% | 80.51% |
| VGG16 | 78.40% | 78.20% | 78.40% | 77.03% |

ResNet50 demonstrates higher accuracy across all metrics, which aligns with the known characteristics of this architecture for image classification tasks.

## Installation

The project requires Python version 3.11 or higher. After cloning the repository, you need to create a virtual environment and install the dependencies. The most convenient way is to use the `uv` package manager.

```bash
uv python install 3.11
uv sync
```

Alternatively, you can use the standard pip.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

The core dependencies include PyTorch and torchvision for neural networks, OpenCV for video and image processing, scikit-learn for classification metrics, matplotlib and seaborn for visualization, and pandas for handling CSV reports.

## Usage

Before running the pipeline, you must prepare your videos. They should be placed in the `videos/` directory, organized into class-specific subdirectories. The subdirectory names must exactly match the concentration labels: `solution_0pct`, `solution_5pct`, `solution_12.5pct`, `solution_25pct`, `solution_50pct`, `solution_75pct`, and `solution_100pct`. Supported formats are mp4, avi, and mov.

The pipeline is then executed by running four commands sequentially.

```bash
python -m bubble.cli extract --input-dir videos --output-dir output_frames
python -m bubble.cli filter --input-dir output_frames --output-dir filtered_frames
python -m bubble.cli resize --input-dir filtered_frames --output-dir resized_frames
python -m bubble.cli train --config configs/default.toml
```

All parameters are configured via the `configs/default.toml` file. It defines brightness and bubble area thresholds for filtering, target image sizes, the list of architectures to train, batch size, learning rate, number of epochs, and the patience parameter for early stopping. Any parameter can be overridden via command-line arguments, such as `--epochs 20` or `--architectures resnet50`.

Upon completion of training, the root directory will contain `best_resnet50_model.pth` and `best_vgg16_model.pth` with the best model weights, a `results.csv` file with a summary table of metrics, JSON files with per-epoch training history, and PNG plots of the training curves and confusion matrices for each architecture.

Project Structure

The code is organized in the `src/bubble/` package by functional domain. The entry point is the `cli.py` module, which parses command-line arguments and invokes the corresponding pipeline functions. Configurations are defined as immutable dataclasses with validation in `config.py`. Domain entities like `Concentration`, `ImageSize`, and `FrameSample` are strictly typed and independent of external libraries, making it easy to write unit tests.

The testing infrastructure is configured via pytest with settings in `pyproject.toml`. Static analysis is handled by ruff for linting and mypy for type checking. All tools are configured within the same `pyproject.toml` file to simplify project maintenance.

## License

The project is distributed under the MIT License.
