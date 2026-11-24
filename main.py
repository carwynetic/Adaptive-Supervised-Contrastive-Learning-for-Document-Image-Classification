from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _run_standard_train(args: argparse.Namespace) -> None:
    from data_processing import get_transforms
    from model import DIC, DICWithoutGroupwise
    from train_test.train import create_dry_run_loaders
    from train_test.trainer import DICTrainer
    from utils.dataloader import build_dataloader

    if args.train_type == "groupwise":
        model = DIC(num_classes=args.num_classes, exit_layers=args.exit_layers, projection_dim=128)
    else:
        model = DICWithoutGroupwise(num_classes=args.num_classes, exit_layers=args.exit_layers, projection_dim=128)

    trainer = DICTrainer(
        model=model,
        train_type=args.train_type,
        exit_layers=args.exit_layers,
        num_classes=args.num_classes,
        smoothing_coeff=args.smoothing_coeff,
        temperature=args.temperature,
        device=args.device,
        gradient_accumulation_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
    )

    if args.dry_run:
        train_loader_s1, train_loader_s2 = create_dry_run_loaders()
        val_loader = train_loader_s2
    else:
        train_loader_s1 = build_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_contrastive=True,
            transform=get_transforms(),
            max_samples=args.max_samples,
        )
        train_loader_s2 = build_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_contrastive=False,
            transform=None,
            max_samples=args.max_samples,
        )
        val_loader = build_dataloader(
            data_dir=args.data_dir,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_contrastive=False,
            transform=None,
            max_samples=args.max_samples,
        )

    trainer.train_stage1(
        train_loader=train_loader_s1,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=max(1, int(args.epochs * args.warmup_ratio)),
        use_lars=args.use_lars,
        save_dir=args.save_dir,
    )
    trainer.train_stage2(
        train_loader=train_loader_s2,
        val_loader=val_loader,
        epochs=args.epochs_stage2,
        lr=args.lr_stage2,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
    )


def _run_standard_eval(args: argparse.Namespace) -> None:
    from model.DIC_without_groupwise import DICWithoutGroupwise
    from train_test.evaluate import EarlyExitEvaluator
    from utils.dataloader import build_dataloader

    model = DICWithoutGroupwise(num_classes=args.num_classes, exit_layers=args.exit_layers)
    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    evaluator = EarlyExitEvaluator(model=model, exit_layers=args.exit_layers, device=args.device)
    if args.dry_run:
        dataloader = [
            {
                "pixel_values": torch.randn(4, 3, 224, 224),
                "input_ids": torch.zeros((4, 16), dtype=torch.long),
                "bbox": torch.zeros((4, 16, 4), dtype=torch.long),
                "labels": torch.tensor([0, 5, 10, 15]),
            }
        ]
    else:
        dataloader = build_dataloader(
            data_dir=args.data_dir,
            split="test",
            batch_size=args.batch_size,
            is_contrastive=False,
            shuffle=False,
        )

    all_logits, all_labels = evaluator.collect_all_logits(dataloader)
    static_results = evaluator.evaluate_static_exits(all_logits, all_labels)
    for l, m in static_results.items():
        print(f"Exit L{l:2d}: Acc={m['accuracy']:.2f}% | Conf={m['avg_confidence']:.4f} | Ent={m['avg_entropy']:.4f}")
    if args.sweep:
        sweep_data = evaluator.sweep_thresholds(all_logits, all_labels, criterion=args.criterion)
        for s in sweep_data:
            print(f"t={s['threshold']:.2f} | Acc={s['adaptive_accuracy']:.2f}% | AvgL={s['avg_exit_layer']:.2f} | Saved={s['compute_saved_percent']:.2f}%")
    else:
        res = evaluator.evaluate_adaptive_inference(all_logits, all_labels, criterion=args.criterion, threshold=args.threshold)
        print(f"Adaptive Acc={res['adaptive_accuracy']:.2f}% | AvgL={res['avg_exit_layer']:.2f} | Saved={res['compute_saved_percent']:.2f}%")


def main_app(args: argparse.Namespace) -> None:
    if args.mode == "prepare_data":
        from data_processing.prepare_rvl_subset import prepare_rvl_cdip_subset

        counts = prepare_rvl_cdip_subset(
            output_dir=args.data_dir,
            dataset_name=args.dataset_name,
            ocr_engine=args.ocr_engine,
            max_samples_per_split=args.max_samples,
            seed=args.seed,
        )
        print(f"Prepared dataset at {args.data_dir}: {counts}")
    elif args.mode == "exp1_hamed_ce":
        from train_test.exp_runner import run_exp1_hamed_ce

        run_exp1_hamed_ce(
            data_dir=args.data_dir,
            output_dir=args.exp1_dir,
            exit_layers=args.exit_layers,
            num_classes=args.num_classes,
            weighting_mode=args.weighting_mode,
            gamma=args.gamma,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            probe_every_n_steps=args.probe_every_n_steps,
            landscape_grid_size=args.landscape_grid_size,
            landscape_span=args.landscape_span,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            use_amp=args.use_amp,
            use_checkpointing=args.use_checkpointing,
            dry_run=args.dry_run,
        )
    elif args.mode == "exp2_supcon":
        from train_test.exp_runner import run_exp2_supcon

        run_exp2_supcon(
            data_dir=args.data_dir,
            output_dir=args.exp2_dir,
            exit_layers=args.exit_layers,
            num_classes=args.num_classes,
            weighting_mode=args.weighting_mode,
            gamma=args.gamma,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            epochs_stage2=args.epochs_stage2,
            lr=args.lr,
            lr_stage2=args.lr_stage2,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            temperature=args.temperature,
            smoothing_coeff=args.smoothing_coeff,
            use_lars=args.use_lars,
            probe_every_n_steps=args.probe_every_n_steps,
            landscape_grid_size=args.landscape_grid_size,
            landscape_span=args.landscape_span,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            use_amp=args.use_amp,
            use_checkpointing=args.use_checkpointing,
            dry_run=args.dry_run,
            run_stage2_probe=args.run_stage2_probe,
        )
    elif args.mode == "run_both":
        from train_test.exp_runner import run_exp1_hamed_ce, run_exp2_supcon
        from train_test.plot_gradient_conflict import generate_all_visualizations

        run_exp1_hamed_ce(
            data_dir=args.data_dir,
            output_dir=args.exp1_dir,
            exit_layers=args.exit_layers,
            num_classes=args.num_classes,
            weighting_mode=args.weighting_mode,
            gamma=args.gamma,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            probe_every_n_steps=args.probe_every_n_steps,
            landscape_grid_size=args.landscape_grid_size,
            landscape_span=args.landscape_span,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            use_amp=args.use_amp,
            use_checkpointing=args.use_checkpointing,
            dry_run=args.dry_run,
        )
        run_exp2_supcon(
            data_dir=args.data_dir,
            output_dir=args.exp2_dir,
            exit_layers=args.exit_layers,
            num_classes=args.num_classes,
            weighting_mode=args.weighting_mode,
            gamma=args.gamma,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            epochs=args.epochs,
            epochs_stage2=args.epochs_stage2,
            lr=args.lr,
            lr_stage2=args.lr_stage2,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            temperature=args.temperature,
            smoothing_coeff=args.smoothing_coeff,
            use_lars=args.use_lars,
            probe_every_n_steps=args.probe_every_n_steps,
            landscape_grid_size=args.landscape_grid_size,
            landscape_span=args.landscape_span,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            use_amp=args.use_amp,
            use_checkpointing=args.use_checkpointing,
            dry_run=args.dry_run,
            run_stage2_probe=args.run_stage2_probe,
        )
        generate_all_visualizations(
            exp1_log_path=args.exp1_dir,
            exp2_log_path=args.exp2_dir,
            output_dir=args.figures_dir,
        )
    elif args.mode == "visualize_all":
        from train_test.plot_gradient_conflict import generate_all_visualizations

        res = generate_all_visualizations(
            exp1_log_path=args.exp1_dir,
            exp2_log_path=args.exp2_dir,
            output_dir=args.figures_dir,
        )
        print(f"Saved figures and summary tables to: {res['output_dir']}")
    elif args.mode == "train_standard":
        _run_standard_train(args)
    elif args.mode == "evaluate_standard":
        _run_standard_eval(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Contrastive Early-Exiting DIC Runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="run_both",
        choices=[
            "prepare_data",
            "exp1_hamed_ce",
            "exp2_supcon",
            "run_both",
            "visualize_all",
            "train_standard",
            "evaluate_standard",
        ],
    )
    parser.add_argument("--data_dir", type=str, default="data/rvl_cdip_100")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp1_dir", type=str, default="outputs/exp1_hamed_ce")
    parser.add_argument("--exp2_dir", type=str, default="outputs/exp2_supcon")
    parser.add_argument("--figures_dir", type=str, default="outputs/figures")

    parser.add_argument("--dataset_name", type=str, default="jordyvl/rvl_cdip_100_examples_per_class")
    parser.add_argument("--ocr_engine", type=str, default="tesseract", choices=["tesseract", "easyocr", "pp_ocrv6"])

    parser.add_argument("--exit_layers", type=int, nargs="+", default=[0, 1, 4, 8, 12])
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--weighting_mode", type=str, default="subgraph_paper", choices=["uniform", "subgraph_paper", "subgraph_repo"])
    parser.add_argument("--gamma", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epochs_stage2", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_stage2", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--smoothing_coeff", type=float, default=0.1)
    parser.add_argument("--use_lars", action="store_true", default=False)

    parser.add_argument("--probe_every_n_steps", type=int, default=10)
    parser.add_argument("--landscape_grid_size", type=int, default=11)
    parser.add_argument("--landscape_span", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_stage2_probe", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train_type", type=str, default="without_groupwise", choices=["without_groupwise", "groupwise"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/run_01")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/run_01/best_model.pt")
    parser.add_argument("--criterion", type=str, default="max_confidence", choices=["max_confidence", "entropy", "top12_margin"])
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--sweep", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main_app(args)
