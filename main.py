from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import EEBaselineConfig, Stage1SupConConfig, Stage2ClassifierConfig


def _arg(args: argparse.Namespace, primary: str, fallback: str | None, default: Any) -> Any:
    if hasattr(args, primary):
        val = getattr(args, primary)
        if val is not None or primary == "s1_base_temperature":
            return val
    if fallback is not None and hasattr(args, fallback):
        return getattr(args, fallback)
    return default


def _run_standard_train(args: argparse.Namespace) -> None:
    from data_processing import get_transforms
    from model import DIC, DICWithoutGroupwise
    from train_test.train import create_dry_run_loaders
    from train_test.trainer import DICTrainer
    from utils.dataloader import build_dataloader

    s1_def = Stage1SupConConfig()
    s2_def = Stage2ClassifierConfig()

    s1_batch_size = _arg(args, "s1_batch_size", "batch_size", s1_def.batch_size)
    s1_grad_accum = _arg(args, "s1_grad_accum_steps", "grad_accum_steps", s1_def.grad_accum_steps)
    s1_epochs = _arg(args, "s1_epochs", "epochs", s1_def.epochs)
    s1_lr = _arg(args, "s1_lr", "lr", s1_def.lr)
    s1_wd = _arg(args, "s1_weight_decay", "weight_decay", s1_def.weight_decay)
    s1_warmup = _arg(args, "s1_warmup_ratio", "warmup_ratio", s1_def.warmup_ratio)
    s1_temp = _arg(args, "s1_temperature", "temperature", s1_def.temperature)
    s1_base_temp = _arg(args, "s1_base_temperature", None, s1_def.base_temperature)
    s1_smooth = _arg(args, "s1_smoothing_coeff", "smoothing_coeff", s1_def.smoothing_coeff)
    s1_use_lars = _arg(args, "s1_use_lars", "use_lars", s1_def.use_lars)
    s1_lars_eta = _arg(args, "s1_lars_eta", None, s1_def.lars_eta)
    s1_excl_bn = _arg(args, "s1_exclude_bias_norm", None, s1_def.exclude_bias_norm)
    s1_max_gn = _arg(args, "s1_max_grad_norm", None, s1_def.max_grad_norm)
    s1_shear = _arg(args, "s1_max_shear", None, s1_def.max_shear)
    s1_blur = _arg(args, "s1_p_blur", None, s1_def.p_blur)
    s1_noise = _arg(args, "s1_noise_std", None, s1_def.noise_std)
    s1_erase = _arg(args, "s1_p_erase", None, s1_def.p_erase)

    s2_batch_size = _arg(args, "s2_batch_size", "batch_size", s2_def.batch_size)
    s2_eval_bs = _arg(args, "s2_eval_batch_size", "batch_size", s2_def.eval_batch_size)
    s2_grad_accum = _arg(args, "s2_grad_accum_steps", "grad_accum_steps", s2_def.grad_accum_steps)
    s2_epochs = _arg(args, "s2_epochs", "epochs_stage2", s2_def.epochs)
    s2_lr = _arg(args, "s2_lr", "lr_stage2", s2_def.lr)
    s2_wd = _arg(args, "s2_weight_decay", "weight_decay", s2_def.weight_decay)
    s2_ls = _arg(args, "s2_label_smoothing", None, s2_def.label_smoothing)
    s2_use_aug = _arg(args, "s2_use_aug", None, s2_def.use_augmentation)

    if args.train_type == "groupwise":
        model = DIC(num_classes=args.num_classes, exit_layers=args.exit_layers, projection_dim=s1_def.projection_dim)
    else:
        model = DICWithoutGroupwise(
            num_classes=args.num_classes,
            exit_layers=args.exit_layers,
            projection_dim=s1_def.projection_dim,
        )

    trainer = DICTrainer(
        model=model,
        train_type=args.train_type,
        exit_layers=args.exit_layers,
        num_classes=args.num_classes,
        smoothing_coeff=s1_smooth,
        temperature=s1_temp,
        base_temperature=s1_base_temp,
        label_smoothing_stage2=s2_ls,
        device=args.device,
        gradient_accumulation_steps=s1_grad_accum,
        use_amp=args.use_amp,
    )

    aug_tf = get_transforms(
        max_shear=s1_shear,
        brightness_range=s1_def.brightness_range,
        p_erase=s1_erase,
        noise_std=s1_noise,
        p_blur=s1_blur,
        blur_sigma=s1_def.blur_sigma,
    )

    if args.dry_run:
        train_loader_s1, train_loader_s2 = create_dry_run_loaders()
        val_loader = train_loader_s2
    elif (Path(args.data_dir) / "ocr_cache" / "train.jsonl").exists():
        from torch.utils.data import DataLoader
        from data_processing import MultimodalCollator, PKBatchSampler, RVLCDIPCachedDataset

        collator = MultimodalCollator(max_length=512)
        pin = torch.cuda.is_available()
        s1_sampler = _arg(args, "s1_sampler", None, s1_def.sampler)
        s1_pk_p = _arg(args, "s1_pk_p", None, s1_def.pk_p)
        s1_pk_k = _arg(args, "s1_pk_k", None, s1_def.pk_k)

        ds_s1 = RVLCDIPCachedDataset(
            data_dir=args.data_dir,
            split="train",
            transform=aug_tf,
            is_contrastive=True,
            max_samples=args.max_samples,
        )
        if s1_sampler == "pk" and len(ds_s1) > 0:
            pk = PKBatchSampler([int(r["label"]) for r in ds_s1.records], p=s1_pk_p, k=s1_pk_k, seed=args.seed)
            train_loader_s1 = DataLoader(ds_s1, batch_sampler=pk, num_workers=args.num_workers, collate_fn=collator, pin_memory=pin)
        else:
            train_loader_s1 = DataLoader(
                ds_s1,
                batch_size=s1_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collator,
                pin_memory=pin,
                drop_last=len(ds_s1) >= s1_batch_size,
            )

        ds_s2 = RVLCDIPCachedDataset(
            data_dir=args.data_dir,
            split="train",
            transform=aug_tf if s2_use_aug else None,
            is_contrastive=False,
            max_samples=args.max_samples,
        )
        ds_val = RVLCDIPCachedDataset(
            data_dir=args.data_dir,
            split="val",
            transform=None,
            is_contrastive=False,
            max_samples=args.max_samples,
        )
        train_loader_s2 = DataLoader(
            ds_s2,
            batch_size=s2_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=s2_eval_bs,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=pin,
        )
    else:
        train_loader_s1 = build_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=s1_batch_size,
            num_workers=args.num_workers,
            is_contrastive=True,
            transform=aug_tf,
            max_samples=args.max_samples,
        )
        train_loader_s2 = build_dataloader(
            data_dir=args.data_dir,
            split="train",
            batch_size=s2_batch_size,
            num_workers=args.num_workers,
            is_contrastive=False,
            transform=aug_tf if s2_use_aug else None,
            max_samples=args.max_samples,
        )
        val_loader = build_dataloader(
            data_dir=args.data_dir,
            split="val",
            batch_size=s2_eval_bs,
            num_workers=args.num_workers,
            is_contrastive=False,
            transform=None,
            max_samples=args.max_samples,
        )

    trainer.train_stage1(
        train_loader=train_loader_s1,
        epochs=s1_epochs,
        lr=s1_lr,
        weight_decay=s1_wd,
        warmup_epochs=max(1, int(s1_epochs * s1_warmup)) if s1_warmup > 0 else 0,
        use_lars=s1_use_lars,
        lars_eta=s1_lars_eta,
        exclude_bias_norm=s1_excl_bn,
        max_grad_norm=s1_max_gn,
        save_dir=args.save_dir,
    )
    trainer.train_stage2(
        train_loader=train_loader_s2,
        val_loader=val_loader,
        epochs=s2_epochs,
        lr=s2_lr,
        weight_decay=s2_wd,
        gradient_accumulation_steps=s2_grad_accum,
        save_dir=args.save_dir,
    )


def _resolve_sub_dir(args: argparse.Namespace, attr: str, default_sub: str) -> str:
    val = getattr(args, attr, None)
    default_path = f"outputs/{default_sub}"
    out_root = getattr(args, "output_dir", "outputs")
    if val is None or (val == default_path and out_root != "outputs"):
        return str(Path(out_root) / default_sub)
    return str(val)


def _pick_checkpoint(exp_dir: str) -> str:
    root = Path(exp_dir)
    best = root / "model_best.pt"
    if best.exists():
        return str(best)
    return str(root / "model_final.pt")


def _run_eval_for_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: str,
    eval_out_dir: str,
    model_type: str = "auto",
) -> dict[str, Any]:
    import json
    from train_test.evaluate import (
        EarlyExitEvaluator,
        build_cached_eval_loader,
        load_model_for_evaluation,
    )

    out_dir_path = Path(eval_out_dir)
    report_file = out_dir_path / "eval_report.json"
    if _arg(args, "resume", None, True) and report_file.exists():
        print(f"[Eval:{model_type}] Already completed at {report_file}, loading saved report.")
        return json.loads(report_file.read_text(encoding="utf-8"))

    s2_def = Stage2ClassifierConfig()
    eval_bs = _arg(args, "s2_eval_batch_size", "batch_size", s2_def.eval_batch_size)
    calibrate = _arg(args, "calibrate", None, s2_def.calibrate_temperature)
    epsilon = _arg(args, "epsilon", None, 0.1)
    benchmark_latency = _arg(args, "benchmark_latency", None, False)

    model, resolved_type = load_model_for_evaluation(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        num_classes=args.num_classes,
        exit_layers=args.exit_layers,
        device=args.device,
    )

    evaluator = EarlyExitEvaluator(
        model=model,
        exit_layers=args.exit_layers,
        device=args.device,
        use_amp=args.use_amp,
    )
    if args.dry_run:
        dummy_loader = [
            {
                "pixel_values": torch.randn(4, 3, 224, 224),
                "input_ids": torch.randint(0, 1000, (4, 16), dtype=torch.long),
                "bbox": torch.zeros((4, 16, 4), dtype=torch.long),
                "attention_mask": torch.ones((4, 16), dtype=torch.long),
                "labels": torch.tensor([0, 5, 10, 15]),
            }
        ]
        val_loader = dummy_loader if calibrate else None
        test_loader = dummy_loader
    else:
        val_loader = (
            build_cached_eval_loader(
                data_dir=args.data_dir,
                split="val",
                batch_size=eval_bs,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
            )
            if calibrate
            else None
        )
        test_loader = build_cached_eval_loader(
            data_dir=args.data_dir,
            split="test",
            batch_size=eval_bs,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

    report = evaluator.run_full_evaluation(
        test_loader=test_loader,
        val_loader=val_loader,
        criterion=args.criterion,
        threshold=args.threshold,
        epsilon=epsilon,
        calibrate=calibrate,
        output_dir=eval_out_dir,
        benchmark_latency=benchmark_latency,
    )

    print(f"[Model={resolved_type}] Evaluated {report['num_test_samples']} samples.")
    for l_str, m in report["uncalibrated"]["static"].items():
        print(
            f"Exit L{int(l_str):2d} [Uncal]: Acc={m['accuracy']:.2f}% | F1={m['f1_macro']:.4f} "
            f"| ECE={m['ece']:.4f} | NLL={m['nll']:.4f} | AURC={m['aurc']:.4f}"
        )
    if args.sweep:
        sweep_src = report.get("calibrated", report["uncalibrated"])["global_sweep"]
        tag = "Cal" if "calibrated" in report else "Uncal"
        for s in sweep_src:
            print(
                f"[{tag}] t={s['threshold']:.2f} | Acc={s['adaptive_accuracy']:.2f}% "
                f"| ECE={s['ece']:.4f} | AvgL={s['avg_exit_layer']:.2f} "
                f"| ParamsSaved={s['params_reduction']:.2f}% | RelLatSaved={s['relative_latency_reduction']:.2f}%"
            )
    else:
        single = report.get("calibrated", report["uncalibrated"])["single_threshold"]
        print(
            f"Adaptive Acc={single['adaptive_accuracy']:.2f}% | ECE={single['ece']:.4f} "
            f"| AvgL={single['avg_exit_layer']:.2f} | ParamsSaved={single['params_reduction']:.2f}%"
        )
    if "calibrated" in report:
        h = report["calibrated"]["heuristic"]
        print(
            f"[Cal Heuristic eps={h['epsilon']:.2f}] Acc={h['adaptive_accuracy']:.2f}% "
            f"| ECE={h['ece']:.4f} | ParamsSaved={h['params_reduction']:.2f}% "
            f"| RelLatSaved={h['relative_latency_reduction']:.2f}%"
        )
    if "latency_benchmark" in report:
        lb = report["latency_benchmark"]
        print(
            f"[True Latency] Full={lb['full_ms_per_sample']:.2f}ms -> EE={lb['ee_ms_per_sample']:.2f}ms "
            f"(Speedup={lb['wall_clock_speedup_percent']:.2f}%)"
        )
    if "artifacts" in report:
        print(f"Saved evaluation report & Pareto plots to: {report['artifacts']['report_json']}")
    return report


def _run_standard_eval(args: argparse.Namespace) -> None:
    _run_eval_for_checkpoint(
        args=args,
        checkpoint_path=args.checkpoint_path,
        eval_out_dir=_resolve_sub_dir(args, "eval_output_dir", "eval"),
        model_type=_arg(args, "model_type", None, "auto"),
    )


def _run_exp1_from_args(args: argparse.Namespace) -> None:
    from train_test.exp_runner import run_exp1_hamed_ce

    ee_def = EEBaselineConfig()
    run_exp1_hamed_ce(
        data_dir=args.data_dir,
        output_dir=_resolve_sub_dir(args, "exp1_dir", "exp1_hamed_ce"),
        exit_layers=args.exit_layers,
        num_classes=args.num_classes,
        weighting_mode=_arg(args, "ee_weighting_mode", "weighting_mode", ee_def.weighting_mode),
        gamma=_arg(args, "ee_gamma", "gamma", ee_def.gamma),
        batch_size=_arg(args, "ee_batch_size", "batch_size", ee_def.batch_size),
        eval_batch_size=_arg(args, "ee_eval_batch_size", None, ee_def.eval_batch_size),
        grad_accum_steps=_arg(args, "ee_grad_accum_steps", "grad_accum_steps", ee_def.grad_accum_steps),
        epochs=_arg(args, "ee_epochs", "epochs", ee_def.epochs),
        lr=_arg(args, "ee_lr", "lr", ee_def.lr),
        weight_decay=_arg(args, "ee_weight_decay", "weight_decay", ee_def.weight_decay),
        warmup_ratio=_arg(args, "ee_warmup_ratio", "warmup_ratio", ee_def.warmup_ratio),
        scheduler_type=_arg(args, "ee_scheduler", None, ee_def.scheduler),
        max_grad_norm=_arg(args, "ee_max_grad_norm", None, ee_def.max_grad_norm),
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
        resume=_arg(args, "resume", None, True),
    )


def _run_exp2_from_args(args: argparse.Namespace) -> None:
    from train_test.exp_runner import run_exp2_supcon

    s1_def = Stage1SupConConfig()
    s2_def = Stage2ClassifierConfig()
    run_exp2_supcon(
        data_dir=args.data_dir,
        output_dir=_resolve_sub_dir(args, "exp2_dir", "exp2_supcon"),
        exit_layers=args.exit_layers,
        num_classes=args.num_classes,
        weighting_mode=_arg(args, "s1_exit_weighting", "weighting_mode", s1_def.exit_weighting),
        gamma=_arg(args, "s1_gamma", "gamma", s1_def.gamma),
        batch_size=_arg(args, "s1_batch_size", "batch_size", s1_def.batch_size),
        grad_accum_steps=_arg(args, "s1_grad_accum_steps", "grad_accum_steps", s1_def.grad_accum_steps),
        sampler=_arg(args, "s1_sampler", None, s1_def.sampler),
        pk_p=_arg(args, "s1_pk_p", None, s1_def.pk_p),
        pk_k=_arg(args, "s1_pk_k", None, s1_def.pk_k),
        epochs=_arg(args, "s1_epochs", "epochs", s1_def.epochs),
        lr=_arg(args, "s1_lr", "lr", s1_def.lr),
        weight_decay=_arg(args, "s1_weight_decay", "weight_decay", s1_def.weight_decay),
        exclude_bias_norm=_arg(args, "s1_exclude_bias_norm", None, s1_def.exclude_bias_norm),
        warmup_ratio=_arg(args, "s1_warmup_ratio", "warmup_ratio", s1_def.warmup_ratio),
        warmup_start_lr_ratio=_arg(args, "s1_warmup_start_lr_ratio", None, s1_def.warmup_start_lr_ratio),
        eta_min_ratio=_arg(args, "s1_eta_min_ratio", None, s1_def.eta_min),
        temperature=_arg(args, "s1_temperature", "temperature", s1_def.temperature),
        base_temperature=_arg(args, "s1_base_temperature", None, s1_def.base_temperature),
        smoothing_coeff=_arg(args, "s1_smoothing_coeff", "smoothing_coeff", s1_def.smoothing_coeff),
        use_lars=_arg(args, "s1_use_lars", "use_lars", s1_def.use_lars),
        lars_eta=_arg(args, "s1_lars_eta", None, s1_def.lars_eta),
        max_grad_norm=_arg(args, "s1_max_grad_norm", None, s1_def.max_grad_norm),
        max_shear=_arg(args, "s1_max_shear", None, s1_def.max_shear),
        brightness_range=s1_def.brightness_range,
        p_blur=_arg(args, "s1_p_blur", None, s1_def.p_blur),
        blur_sigma=s1_def.blur_sigma,
        noise_std=_arg(args, "s1_noise_std", None, s1_def.noise_std),
        p_erase=_arg(args, "s1_p_erase", None, s1_def.p_erase),
        epochs_stage2=_arg(args, "s2_epochs", "epochs_stage2", s2_def.epochs),
        batch_size_stage2=_arg(args, "s2_batch_size", "batch_size", s2_def.batch_size),
        eval_batch_size=_arg(args, "s2_eval_batch_size", None, s2_def.eval_batch_size),
        grad_accum_steps_stage2=_arg(args, "s2_grad_accum_steps", "grad_accum_steps", s2_def.grad_accum_steps),
        lr_stage2=_arg(args, "s2_lr", "lr_stage2", s2_def.lr),
        weight_decay_stage2=_arg(args, "s2_weight_decay", "weight_decay", s2_def.weight_decay),
        label_smoothing_stage2=_arg(args, "s2_label_smoothing", None, s2_def.label_smoothing),
        use_aug_stage2=_arg(args, "s2_use_aug", None, s2_def.use_augmentation),
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
        resume=_arg(args, "resume", None, True),
    )


def _run_exp3_from_args(args: argparse.Namespace) -> None:
    from train_test.exp_runner import run_exp3_groupwise

    s1_def = Stage1SupConConfig()
    s2_def = Stage2ClassifierConfig()
    run_exp3_groupwise(
        data_dir=args.data_dir,
        output_dir=_resolve_sub_dir(args, "exp3_dir", "exp3_groupwise"),
        exit_layers=args.exit_layers,
        num_classes=args.num_classes,
        weighting_mode=_arg(args, "s1_exit_weighting", "weighting_mode", s1_def.exit_weighting),
        gamma=_arg(args, "s1_gamma", "gamma", s1_def.gamma),
        batch_size=_arg(args, "s1_batch_size", "batch_size", s1_def.batch_size),
        grad_accum_steps=_arg(args, "s1_grad_accum_steps", "grad_accum_steps", s1_def.grad_accum_steps),
        sampler=_arg(args, "s1_sampler", None, s1_def.sampler),
        pk_p=_arg(args, "s1_pk_p", None, s1_def.pk_p),
        pk_k=_arg(args, "s1_pk_k", None, s1_def.pk_k),
        epochs=_arg(args, "s1_epochs", "epochs", s1_def.epochs),
        lr=_arg(args, "s1_lr", "lr", s1_def.lr),
        weight_decay=_arg(args, "s1_weight_decay", "weight_decay", s1_def.weight_decay),
        exclude_bias_norm=_arg(args, "s1_exclude_bias_norm", None, s1_def.exclude_bias_norm),
        warmup_ratio=_arg(args, "s1_warmup_ratio", "warmup_ratio", s1_def.warmup_ratio),
        warmup_start_lr_ratio=_arg(args, "s1_warmup_start_lr_ratio", None, s1_def.warmup_start_lr_ratio),
        eta_min_ratio=_arg(args, "s1_eta_min_ratio", None, s1_def.eta_min),
        temperature=_arg(args, "s1_temperature", "temperature", s1_def.temperature),
        base_temperature=_arg(args, "s1_base_temperature", None, s1_def.base_temperature),
        smoothing_coeff=_arg(args, "s1_smoothing_coeff", "smoothing_coeff", s1_def.smoothing_coeff),
        use_lars=_arg(args, "s1_use_lars", "use_lars", s1_def.use_lars),
        lars_eta=_arg(args, "s1_lars_eta", None, s1_def.lars_eta),
        max_grad_norm=_arg(args, "s1_max_grad_norm", None, s1_def.max_grad_norm),
        max_shear=_arg(args, "s1_max_shear", None, s1_def.max_shear),
        brightness_range=s1_def.brightness_range,
        p_blur=_arg(args, "s1_p_blur", None, s1_def.p_blur),
        blur_sigma=s1_def.blur_sigma,
        noise_std=_arg(args, "s1_noise_std", None, s1_def.noise_std),
        p_erase=_arg(args, "s1_p_erase", None, s1_def.p_erase),
        epochs_stage2=_arg(args, "s2_epochs", "epochs_stage2", s2_def.epochs),
        batch_size_stage2=_arg(args, "s2_batch_size", "batch_size", s2_def.batch_size),
        eval_batch_size=_arg(args, "s2_eval_batch_size", None, s2_def.eval_batch_size),
        grad_accum_steps_stage2=_arg(args, "s2_grad_accum_steps", "grad_accum_steps", s2_def.grad_accum_steps),
        lr_stage2=_arg(args, "s2_lr", "lr_stage2", s2_def.lr),
        weight_decay_stage2=_arg(args, "s2_weight_decay", "weight_decay", s2_def.weight_decay),
        label_smoothing_stage2=_arg(args, "s2_label_smoothing", None, s2_def.label_smoothing),
        use_aug_stage2=_arg(args, "s2_use_aug", None, s2_def.use_augmentation),
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
        resume=_arg(args, "resume", None, True),
    )


def _run_all_from_args(args: argparse.Namespace) -> None:
    import json
    from train_test.plot_gradient_conflict import generate_all_visualizations

    if not args.dry_run and not (Path(args.data_dir) / "ocr_cache" / "train.jsonl").exists():
        from data_processing.prepare_rvl_subset import prepare_rvl_cdip_subset

        counts = prepare_rvl_cdip_subset(
            output_dir=args.data_dir,
            dataset_name=args.dataset_name,
            ocr_engine=args.ocr_engine,
            max_samples_per_split=args.max_samples,
            seed=args.seed,
        )
        print(f"Prepared dataset at {args.data_dir}: {counts}")

    exp1_dir = _resolve_sub_dir(args, "exp1_dir", "exp1_hamed_ce")
    exp2_dir = _resolve_sub_dir(args, "exp2_dir", "exp2_supcon")
    exp3_dir = _resolve_sub_dir(args, "exp3_dir", "exp3_groupwise")
    figures_dir = _resolve_sub_dir(args, "figures_dir", "figures")
    eval_root = Path(_resolve_sub_dir(args, "eval_output_dir", "eval"))

    _run_exp1_from_args(args)
    _run_exp2_from_args(args)
    _run_exp3_from_args(args)

    generate_all_visualizations(
        exp1_log_path=exp1_dir,
        exp2_log_path=exp2_dir,
        output_dir=figures_dir,
    )

    eval_summary: dict[str, Any] = {}
    for name, ed, mtype in (
        ("exp1_hamed_ce", exp1_dir, "ee_hamed"),
        ("exp2_supcon", exp2_dir, "dic"),
        ("exp3_groupwise", exp3_dir, "dic_groupwise"),
    ):
        ckpt = _pick_checkpoint(ed)
        if Path(ckpt).exists():
            rep = _run_eval_for_checkpoint(
                args=args,
                checkpoint_path=ckpt,
                eval_out_dir=str(eval_root / name),
                model_type=mtype,
            )
            eval_summary[name] = {
                "checkpoint": ckpt,
                "uncalibrated_l12_acc": rep["uncalibrated"]["static"][str(max(args.exit_layers))]["accuracy"],
                "calibrated_heuristic": rep.get("calibrated", {}).get("heuristic", {}),
            }
    eval_root.mkdir(parents=True, exist_ok=True)
    summary_path = eval_root / "all_models_summary.json"
    summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(f"[run_all] Full thesis pipeline finished. Consolidated eval summary: {summary_path}")


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
        _run_exp1_from_args(args)
    elif args.mode == "exp2_supcon":
        _run_exp2_from_args(args)
    elif args.mode == "exp3_groupwise":
        _run_exp3_from_args(args)
    elif args.mode == "run_both":
        from train_test.plot_gradient_conflict import generate_all_visualizations

        _run_exp1_from_args(args)
        _run_exp2_from_args(args)
        generate_all_visualizations(
            exp1_log_path=_resolve_sub_dir(args, "exp1_dir", "exp1_hamed_ce"),
            exp2_log_path=_resolve_sub_dir(args, "exp2_dir", "exp2_supcon"),
            output_dir=_resolve_sub_dir(args, "figures_dir", "figures"),
        )
    elif args.mode == "run_all":
        _run_all_from_args(args)
    elif args.mode == "visualize_all":
        from train_test.plot_gradient_conflict import generate_all_visualizations

        res = generate_all_visualizations(
            exp1_log_path=_resolve_sub_dir(args, "exp1_dir", "exp1_hamed_ce"),
            exp2_log_path=_resolve_sub_dir(args, "exp2_dir", "exp2_supcon"),
            output_dir=_resolve_sub_dir(args, "figures_dir", "figures"),
        )
        print(f"Saved figures and summary tables to: {res['output_dir']}")
    elif args.mode == "train_standard":
        _run_standard_train(args)
    elif args.mode == "evaluate_standard":
        _run_standard_eval(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    ee_def = EEBaselineConfig()
    s1_def = Stage1SupConConfig()
    s2_def = Stage2ClassifierConfig()

    parser = argparse.ArgumentParser(description="Adaptive Contrastive Early-Exiting DIC Runner")

    parser.add_argument(
        "--mode",
        type=str,
        default="run_all",
        choices=[
            "run_all",            # Chạy trọn bộ bài: auto prepare_data -> exp1 -> exp2 -> exp3 -> visualize -> eval 3 model (có --resume)
            "prepare_data",       # Tải RVL-CDIP subset + chạy OCR & lưu cache vào data_dir
            "exp1_hamed_ce",      # Chạy baseline EE gốc (Hamed et al., CE) + đo gradient conflict
            "exp2_supcon",        # Chạy without-groupwise SupCon-ST + Stage 2 probe + đo gradient conflict
            "exp3_groupwise",     # Chạy 6-Group Sequential Recompute SupCon-ST ([0],[1],[4],[8],[10],[12]) + Stage 2 probe
            "run_both",           # Chạy tuần tự exp1_hamed_ce -> exp2_supcon -> visualize_all
            "visualize_all",      # Vẽ 6 biểu đồ & bảng summary từ log JSON có sẵn (không train lại)
            "train_standard",     # Chạy DICTrainer chuẩn 2 stage theo --train_type (lưu vào --save_dir)
            "evaluate_standard",  # Đánh giá EarlyExitEvaluator trên test theo --criterion / --sweep
        ],
    )
    parser.add_argument("--data_dir", type=str, default="data/rvl_cdip_100")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--exp1_dir", type=str, default="outputs/exp1_hamed_ce")
    parser.add_argument("--exp2_dir", type=str, default="outputs/exp2_supcon")
    parser.add_argument("--exp3_dir", type=str, default="outputs/exp3_groupwise")
    parser.add_argument("--figures_dir", type=str, default="outputs/figures")

    parser.add_argument("--dataset_name", type=str, default="jordyvl/rvl_cdip_100_examples_per_class")
    parser.add_argument("--ocr_engine", type=str, default="tesseract", choices=["tesseract", "easyocr", "pp_ocrv6"])

    # Source: Hamed et al. (ICDAR 2024, arXiv:2405.12705), https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/train_full_dataset.sh
    parser.add_argument("--exit_layers", type=int, nargs="+", default=[0, 1, 4, 8, 10, 12])
    parser.add_argument("--num_classes", type=int, default=16)

    # Source: EE baseline defaults in https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/configs.py & EE/models/EE_modules.py
    ee_grp = parser.add_argument_group("EE Baseline (Hamed et al., ICDAR 2024)")
    ee_grp.add_argument("--ee_epochs", type=int, default=ee_def.epochs)
    ee_grp.add_argument("--ee_batch_size", type=int, default=ee_def.batch_size)
    ee_grp.add_argument("--ee_eval_batch_size", type=int, default=ee_def.eval_batch_size)
    ee_grp.add_argument("--ee_grad_accum_steps", type=int, default=ee_def.grad_accum_steps)
    ee_grp.add_argument("--ee_lr", type=float, default=ee_def.lr)
    ee_grp.add_argument("--ee_weight_decay", type=float, default=ee_def.weight_decay)
    ee_grp.add_argument("--ee_warmup_ratio", type=float, default=ee_def.warmup_ratio)
    ee_grp.add_argument("--ee_scheduler", type=str, default=ee_def.scheduler, choices=["linear", "cosine", "warmup_cosine"])
    ee_grp.add_argument("--ee_max_grad_norm", type=float, default=ee_def.max_grad_norm)
    ee_grp.add_argument("--ee_weighting_mode", type=str, default=ee_def.weighting_mode, choices=["uniform", "subgraph_paper", "subgraph_repo"])
    ee_grp.add_argument("--ee_gamma", type=float, default=ee_def.gamma)

    # Source: Stage 1 SupCon-ST (SimCLR arXiv:2002.05709 + ConOSR AAAI 2023 without Mixup + https://github.com/nguhcv/cl-tad)
    s1_grp = parser.add_argument_group("Stage 1 Contrastive Representation Learning (SimCLR + ConOSR SupCon-ST)")
    s1_grp.add_argument("--s1_epochs", type=int, default=s1_def.epochs)
    s1_grp.add_argument("--s1_batch_size", type=int, default=s1_def.batch_size)
    s1_grp.add_argument("--s1_grad_accum_steps", type=int, default=s1_def.grad_accum_steps)
    s1_grp.add_argument("--s1_sampler", type=str, default=s1_def.sampler, choices=["pk", "random"])
    s1_grp.add_argument("--s1_pk_p", type=int, default=s1_def.pk_p)
    s1_grp.add_argument("--s1_pk_k", type=int, default=s1_def.pk_k)
    s1_grp.add_argument("--s1_lr", type=float, default=s1_def.lr)
    s1_grp.add_argument("--s1_weight_decay", type=float, default=s1_def.weight_decay)
    s1_grp.add_argument("--s1_exclude_bias_norm", action=argparse.BooleanOptionalAction, default=s1_def.exclude_bias_norm)
    s1_grp.add_argument("--s1_warmup_ratio", type=float, default=s1_def.warmup_ratio)
    s1_grp.add_argument("--s1_warmup_start_lr_ratio", type=float, default=s1_def.warmup_start_lr_ratio)
    s1_grp.add_argument("--s1_eta_min_ratio", type=float, default=s1_def.eta_min)
    s1_grp.add_argument("--s1_temperature", type=float, default=s1_def.temperature)
    s1_grp.add_argument("--s1_base_temperature", type=float, default=s1_def.base_temperature)
    s1_grp.add_argument("--s1_smoothing_coeff", type=float, default=s1_def.smoothing_coeff)
    s1_grp.add_argument("--s1_use_lars", action=argparse.BooleanOptionalAction, default=s1_def.use_lars)
    s1_grp.add_argument("--s1_lars_eta", type=float, default=s1_def.lars_eta)
    s1_grp.add_argument("--s1_max_grad_norm", type=float, default=s1_def.max_grad_norm)
    s1_grp.add_argument("--s1_max_shear", type=float, default=s1_def.max_shear)
    s1_grp.add_argument("--s1_p_blur", type=float, default=s1_def.p_blur)
    s1_grp.add_argument("--s1_noise_std", type=float, default=s1_def.noise_std)
    s1_grp.add_argument("--s1_p_erase", type=float, default=s1_def.p_erase)
    s1_grp.add_argument("--s1_exit_weighting", type=str, default=s1_def.exit_weighting, choices=["uniform", "subgraph_paper", "subgraph_repo"])
    s1_grp.add_argument("--s1_gamma", type=float, default=s1_def.gamma)

    # Source: Stage 2 Classifier (ConOSR Step 2 https://github.com/NJU-RINC/ConOSR + SimCLR Linear Eval)
    s2_grp = parser.add_argument_group("Stage 2 Frozen-Backbone Classifier Probe")
    s2_grp.add_argument("--s2_epochs", type=int, default=s2_def.epochs)
    s2_grp.add_argument("--s2_batch_size", type=int, default=s2_def.batch_size)
    s2_grp.add_argument("--s2_eval_batch_size", type=int, default=s2_def.eval_batch_size)
    s2_grp.add_argument("--s2_grad_accum_steps", type=int, default=s2_def.grad_accum_steps)
    s2_grp.add_argument("--s2_lr", type=float, default=s2_def.lr)
    s2_grp.add_argument("--s2_weight_decay", type=float, default=s2_def.weight_decay)
    s2_grp.add_argument("--s2_label_smoothing", type=float, default=s2_def.label_smoothing)
    s2_grp.add_argument("--s2_use_aug", action=argparse.BooleanOptionalAction, default=s2_def.use_augmentation)

    parser.add_argument("--probe_every_n_steps", type=int, default=10)
    parser.add_argument("--landscape_grid_size", type=int, default=11)
    parser.add_argument("--landscape_span", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_stage2_probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--train_type", type=str, default="without_groupwise", choices=["without_groupwise", "groupwise"])
    parser.add_argument("--save_dir", type=str, default="checkpoints/run_01")
    parser.add_argument("--checkpoint_path", type=str, default="outputs/exp2_supcon/model_final.pt")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "dic", "dic_groupwise", "ee_hamed"])
    parser.add_argument("--eval_output_dir", type=str, default="outputs/eval")
    parser.add_argument("--criterion", type=str, default="max_confidence", choices=["max_confidence", "entropy", "top12_margin"])
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=s2_def.calibrate_temperature)
    parser.add_argument("--benchmark_latency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=True)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main_app(args)
