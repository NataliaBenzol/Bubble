import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from bubble import extract, filtering, resize
from bubble.config import (
    ExtractionConfig,
    TrainConfig,
    load_extraction_config,
    load_train_config,
)
from bubble.domain import Concentration, ImageSize
from bubble.ml.backbones import create_backbone
from bubble.ml.checkpoint import BestValAccuracySelector, CheckpointManager
from bubble.ml.data import BubbleDataset, get_data_transforms, load_and_split_data
from bubble.ml.evaluator import evaluate_model
from bubble.ml.trainer import Trainer
from bubble.reporting.csv_reporter import CsvReporter
from bubble.reporting.plotter import Plotter
from bubble.utils import seed_everything


def get_video_list_from_dir(input_dir: str) -> list[tuple[str, str]]:
    input_path = Path(input_dir)
    videos = []
    for ext in (".mp4", ".avi", ".mov"):
        for video_path in input_path.rglob(f"*{ext}"):
            class_name = (
                video_path.parent.name
                if video_path.parent.name != input_path.name
                else "unknown_class"
            )
            videos.append((str(video_path), class_name))
    return videos


def _train_single_architecture(
    architecture: str,
    config: TrainConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Training architecture: {architecture}")
    print(f"{'=' * 60}")

    model = create_backbone(architecture, num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    save_path = config.output_path / f"best_{architecture}_model.pth"
    checkpoint_manager = CheckpointManager(
        save_path=str(save_path),
        selector=BestValAccuracySelector(),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        device=device,
        patience=config.patience,
    )

    start_time = time.time()
    history = trainer.train(num_epochs=config.num_epochs)
    training_time = time.time() - start_time

    print(f"\nLoading best {architecture} model for final evaluation...")
    checkpoint_manager.load_best(model)
    metrics = evaluate_model(model, test_loader, class_names, device)

    csv_reporter = CsvReporter()
    results_csv_path = config.output_path / "results.csv"
    csv_reporter.save_results(metrics, architecture, str(results_csv_path))

    history_path = config.output_path / f"{architecture}_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plotter = Plotter()
    plotter.plot_training_history(history, str(config.output_path / f"{architecture}_training_history.png"))
    plotter.plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        str(config.output_path / f"{architecture}_confusion_matrix.png"),
    )

    print(f"\n{architecture.upper()} SUCCESS. Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Total training time: {training_time / 60:.1f} minutes")


def run_training_pipeline(config: TrainConfig) -> None:
    seed_everything(config.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = Concentration.all_labels()

    train_samples, val_samples, test_samples = load_and_split_data(
        config.data_dir, config.random_state
    )

    transforms_dict = get_data_transforms(config.image_size)
    train_dataset = BubbleDataset(train_samples, transform=transforms_dict["train"])
    val_dataset = BubbleDataset(val_samples, transform=transforms_dict["val"])
    test_dataset = BubbleDataset(test_samples, transform=transforms_dict["test"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    for architecture in config.architectures:
        _train_single_architecture(
            architecture=architecture,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bubble", description="Bubble concentration classification pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    p_extract = subparsers.add_parser("extract", help="Extract frames from videos")
    p_extract.add_argument("--config", default="configs/default.toml", help="Path to config file")
    p_extract.add_argument("--input-dir", default=None, help="Directory containing videos")
    p_extract.add_argument("--output-dir", default=None, help="Output directory for frames")

    p_filter = subparsers.add_parser("filter", help="Filter black images and check for bubbles")
    p_filter.add_argument("--input-dir", default=None, help="Input directory with extracted images")
    p_filter.add_argument("--output-dir", default=None, help="Output directory for filtered images")

    p_resize = subparsers.add_parser("resize", help="Resize images with padding")
    p_resize.add_argument("--input-dir", default=None, help="Input directory with filtered images")
    p_resize.add_argument("--output-dir", default=None, help="Output directory for resized images")

    p_train = subparsers.add_parser("train", help="Train the model(s)")
    p_train.add_argument("--config", default="configs/default.toml", help="Path to config file")
    p_train.add_argument("--data-dir", default=None, help="Directory with resized frames")
    p_train.add_argument("--output-dir", default=None, help="Directory to save models, csv, and plots")
    p_train.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        help="Backbone architectures to train (e.g. resnet50 vgg16)",
    )
    p_train.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p_train.add_argument("--lr", type=float, default=None, help="Learning rate")
    p_train.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p_train.add_argument("--patience", type=int, default=None, help="Early stopping patience")

    args = parser.parse_args()

    try:
        if args.command == "extract":
            config = load_extraction_config(args.config)
            video_list = get_video_list_from_dir(args.input_dir or "videos")
            if not video_list:
                print("No videos found. Please check the input directory.")
                sys.exit(1)
            extract.process_videos(
                video_list=video_list,
                output_root=args.output_dir or "output_frames",
                config=config,
            )

        elif args.command == "filter":
            filtering.filter_images(
                input_folder=args.input_dir or "output_frames",
                output_folder=args.output_dir or "filtered_frames",
            )

        elif args.command == "resize":
            image_size = ImageSize(224, 224)
            resize.resize_images(
                input_folder=args.input_dir or "filtered_frames",
                output_folder=args.output_dir or "resized_frames",
                output_size=image_size,
            )

        elif args.command == "train":
            overrides = {}
            if args.data_dir is not None:
                overrides["data_dir"] = args.data_dir
            if args.output_dir is not None:
                overrides["output_dir"] = args.output_dir
            if args.architectures is not None:
                overrides["architectures"] = args.architectures
            if args.epochs is not None:
                overrides["num_epochs"] = args.epochs
            if args.batch_size is not None:
                overrides["batch_size"] = args.batch_size
            if args.lr is not None:
                overrides["learning_rate"] = args.lr
            if args.seed is not None:
                overrides["random_state"] = args.seed
            if args.patience is not None:
                overrides["patience"] = args.patience

            config = load_train_config(args.config, **overrides)
            run_training_pipeline(config)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()