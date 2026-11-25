from __future__ import annotations

import ast
from argparse import Namespace
import io
import json
import math
from pathlib import Path
import tokenize

from PIL import Image
import pytest
import torch
import torch.nn as nn

from data_processing.data_processing import (
    MultimodalCollator,
    RVLCDIPCachedDataset,
    get_transforms,
)
from model.DIC_without_groupwise import DICWithoutGroupwise
from train_test.trainer import DICTrainer
from utils.gradient_conflict import (
    GradientConflictProber,
    compute_2d_loss_landscape,
    enable_layoutlmv3_checkpointing,
)


class _ToyTwoExitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4, bias=False)
        self.head1 = nn.Linear(4, 2, bias=False)
        self.head2 = nn.Linear(4, 2, bias=False)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        h = torch.tanh(self.shared(x))
        return {1: self.head1(h), 2: self.head2(h)}


def test_prober_opposing_equal_gradients() -> None:
    prober = GradientConflictProber(probe_param_groups={})
    g1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    metrics = prober.compute_metrics_from_vectors(
        grad_vectors={1: g1, 2: g2},
        weights={1: 0.5, 2: 0.5},
    )
    assert metrics["mean_pairwise_cosine"] == pytest.approx(-1.0, abs=1e-5)
    assert metrics["conflict_rate"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["cancellation_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["dominance_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["condition_number"] > 1e3


def test_prober_unequal_opposing_gradients_harmful_step() -> None:
    prober = GradientConflictProber(probe_param_groups={})
    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([-3.0, 0.0, 0.0])
    metrics = prober.compute_metrics_from_vectors(
        grad_vectors={1: g1, 2: g2},
        weights={1: 1.0, 2: 1.0},
    )
    assert metrics["mean_pairwise_cosine"] == pytest.approx(-1.0, abs=1e-5)
    assert metrics["descent_alignment"]["1"] == pytest.approx(-1.0, abs=1e-5)
    assert metrics["descent_alignment"]["2"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["harmful_step"]["1"] is True
    assert metrics["harmful_step"]["2"] is False
    assert metrics["dominance_ratio"] == pytest.approx(3.0, abs=1e-5)
    assert metrics["cancellation_ratio"] == pytest.approx(0.5, abs=1e-5)


def test_prober_orthogonal_gradients() -> None:
    prober = GradientConflictProber(probe_param_groups={})
    g1 = torch.tensor([1.0, 0.0, 0.0])
    g2 = torch.tensor([0.0, 1.0, 0.0])
    g3 = torch.tensor([0.0, 0.0, 1.0])
    metrics = prober.compute_metrics_from_vectors(
        grad_vectors={1: g1, 2: g2, 3: g3},
        weights={1: 1.0, 2: 1.0, 3: 1.0},
    )
    assert metrics["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["conflict_rate"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["condition_number"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["dominance_ratio"] == pytest.approx(1.0, abs=1e-5)
    expected_cancel = 1.0 - (3.0 ** 0.5) / 3.0
    assert metrics["cancellation_ratio"] == pytest.approx(expected_cancel, abs=1e-5)


def test_prober_aligned_gradients_and_summary() -> None:
    prober = GradientConflictProber(probe_param_groups={})
    g1 = torch.tensor([2.0, 0.0])
    g2 = torch.tensor([2.0, 0.0])
    step1 = {"layer_1": prober.compute_metrics_from_vectors({1: g1, 2: g2}, {1: 0.5, 2: 0.5})}
    step2 = {"layer_1": prober.compute_metrics_from_vectors({1: g1, 2: -3.0 * g1}, {1: 0.5, 2: 0.5})}
    prober.history.append({"step": 1, "layers": step1})
    prober.history.append({"step": 2, "layers": step2})

    assert step1["layer_1"]["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert step1["layer_1"]["cancellation_ratio"] == pytest.approx(0.0, abs=1e-5)

    summary = prober.summarize()
    layer_sum = summary["layer_1"]
    assert layer_sum["mean_pairwise_cosine"] == pytest.approx(0.0, abs=1e-5)
    assert layer_sum["conflict_rate"] == pytest.approx(0.5, abs=1e-5)
    assert layer_sum["harmful_step_ratio"]["1"] == pytest.approx(0.5, abs=1e-5)
    assert layer_sum["harmful_step_ratio"]["2"] == pytest.approx(0.0, abs=1e-5)


def test_prober_autograd_and_2d_loss_landscape_restores_weights() -> None:
    torch.manual_seed(0)
    net = _ToyTwoExitNet()
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    ce = nn.CrossEntropyLoss()

    prober = GradientConflictProber(probe_param_groups={"layer_1": list(net.shared.parameters())})
    logits = net(x)
    losses = {1: ce(logits[1], y), 2: ce(logits[2], y)}
    record = prober.probe(step=1, exit_losses=losses, weights={1: 0.5, 2: 0.5})
    assert "layer_1" in record["layers"]
    assert -1.0 <= record["layers"]["layer_1"]["mean_pairwise_cosine"] <= 1.0

    before = [p.detach().clone() for p in net.parameters()]

    def eval_fn() -> dict[int, torch.Tensor]:
        out = net(x)
        return {1: ce(out[1], y), 2: ce(out[2], y)}

    landscape = compute_2d_loss_landscape(
        model=net,
        eval_losses_fn=eval_fn,
        shallow_key=1,
        deep_key=2,
        weights={1: 0.5, 2: 0.5},
        grid_size=5,
        span=0.5,
        target_params=list(net.shared.parameters()),
    )
    after = [p.detach().clone() for p in net.parameters()]
    for b, a in zip(before, after):
        assert torch.allclose(b, a, atol=1e-6)

    assert len(landscape["alphas"]) == 5
    assert len(landscape["betas"]) == 5
    assert len(landscape["shallow_loss"]) == 5
    assert len(landscape["shallow_loss"][0]) == 5
    assert landscape["min_distance"] >= 0.0


def test_prober_numeric_exit_and_layer_ordering() -> None:
    prober = GradientConflictProber(probe_param_groups={})
    grad_vecs = {
        12: torch.tensor([1.0, 0.0]),
        4: torch.tensor([0.0, 1.0]),
        0: torch.tensor([1.0, 1.0]),
        8: torch.tensor([-1.0, 0.0]),
        1: torch.tensor([0.5, 0.5]),
    }
    metrics = prober.compute_metrics_from_vectors(grad_vecs)
    assert metrics["active_exits"] == ["0", "1", "4", "8", "12"]

    prober.history.append({"step": 1, "layers": {"layer_10": metrics, "layer_4": metrics, "layer_0": metrics}})
    summary = prober.summarize()
    assert list(summary.keys()) == ["layer_0", "layer_4", "layer_10"]


def test_landscape_restricts_to_shared_params_when_target_params_none() -> None:
    class _SplitDepthNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4, bias=False)
            self.deep_only = nn.Linear(4, 4, bias=False)
            self.h_shallow = nn.Linear(4, 2, bias=False)
            self.h_deep = nn.Linear(4, 2, bias=False)

        def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
            s = torch.tanh(self.shared(x))
            d = torch.tanh(self.deep_only(s))
            return {0: self.h_shallow(s), 12: self.h_deep(d)}

    torch.manual_seed(7)
    net = _SplitDepthNet()
    x = torch.randn(4, 4)
    y = torch.tensor([0, 1, 0, 1])
    ce = nn.CrossEntropyLoss()
    deep_orig = net.deep_only.weight.detach().clone()
    observed_deep_changed = False

    def eval_fn() -> dict[int, torch.Tensor]:
        nonlocal observed_deep_changed
        if not torch.equal(net.deep_only.weight.detach(), deep_orig):
            observed_deep_changed = True
        out = net(x)
        return {0: ce(out[0], y), 12: ce(out[12], y)}

    ls = compute_2d_loss_landscape(
        model=net,
        eval_losses_fn=eval_fn,
        shallow_key=0,
        deep_key=12,
        grid_size=3,
        span=0.4,
        target_params=None,
    )
    assert observed_deep_changed is False
    assert len(ls["shallow_loss"]) == 3


def test_cached_dataset_and_prepare_stratified_max_samples(tmp_path: Path) -> None:
    from data_processing.prepare_rvl_subset import prepare_rvl_cdip_subset

    src_dir = tmp_path / "sorted_src"
    (src_dir / "images" / "train").mkdir(parents=True)
    (src_dir / "labels").mkdir(parents=True)
    lines = []
    idx = 0
    for cls_id in (0, 1, 2, 3):
        for _ in range(3):
            Image.new("RGB", (64, 64), color=(250, 250, 250)).save(src_dir / "images" / "train" / f"{idx}.png")
            lines.append(f"train/{idx}.png {cls_id}")
            idx += 1
    (src_dir / "labels" / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_dir = tmp_path / "prepared_balanced"
    prepare_rvl_cdip_subset(
        output_dir=out_dir,
        ocr_engine="tesseract",
        max_samples_per_split=4,
        source_local_dir=src_dir,
    )
    ds_sub = RVLCDIPCachedDataset(data_dir=out_dir, split="train", max_samples=4)
    labels_prepared = sorted(r["label"] for r in ds_sub.records)
    assert labels_prepared == [0, 1, 2, 3]

    unshuffled_dir = tmp_path / "unshuffled_cache"
    (unshuffled_dir / "ocr_cache").mkdir(parents=True)
    with open(unshuffled_dir / "ocr_cache" / "train.jsonl", "w", encoding="utf-8") as f:
        for i in range(8):
            rec = {"id": i, "image_path": f"images/train/{i}.png", "label": 0 if i < 4 else 1, "words": ["a"], "boxes": [[0, 0, 10, 10]]}
            f.write(json.dumps(rec) + "\n")
    ds_bal = RVLCDIPCachedDataset(data_dir=unshuffled_dir, split="train", max_samples=4)
    assert sorted(r["label"] for r in ds_bal.records) == [0, 0, 1, 1]


def test_exp1_rejects_invalid_exit_layers(tmp_path: Path) -> None:
    from train_test.exp_runner import run_exp1_hamed_ce

    with pytest.raises(ValueError):
        run_exp1_hamed_ce(
            data_dir=tmp_path,
            output_dir=tmp_path / "bad_final",
            exit_layers=(0, 1, 4, 8),
            device="cpu",
            dry_run=True,
        )
    with pytest.raises(ValueError):
        run_exp1_hamed_ce(
            data_dir=tmp_path,
            output_dir=tmp_path / "bad_range",
            exit_layers=(0, 1, 13),
            device="cpu",
            dry_run=True,
        )


def test_dic_without_groupwise_layer0_pooling_and_compute_logits_false() -> None:
    model = DICWithoutGroupwise(num_classes=16, exit_layers=(0, 1, 4, 8, 12))
    wrapped_count = enable_layoutlmv3_checkpointing(model)
    assert wrapped_count == 12
    assert enable_layoutlmv3_checkpointing(model) == 12

    model.train()
    input_ids = torch.randint(0, 100, (2, 16))
    bbox = torch.zeros((2, 16, 4), dtype=torch.long)
    pixel_values = torch.randn(2, 3, 224, 224)
    attention_mask = torch.ones((2, 16), dtype=torch.long)

    out = model(
        input_ids=input_ids,
        bbox=bbox,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        compute_projections=True,
        compute_logits=False,
    )
    assert out.logits_dict == {}
    assert out.final_logits is None
    assert set(out.projections_dict.keys()) == {0, 1, 4, 8, 12}
    assert out.cls_tokens_dict[0].shape == (2, 768)

    layer0_loss = out.projections_dict[0].pow(2).mean()
    layer12_loss = out.projections_dict[12].pow(2).mean()
    layer0_params = list(model.backbone.model.encoder.layer[0].parameters())
    grads = torch.autograd.grad(layer12_loss, layer0_params, retain_graph=True, allow_unused=True)
    assert any(g is not None and g.abs().sum().item() > 0 for g in grads)
    assert layer0_loss.requires_grad


def test_trainer_keeps_frozen_backbone_in_eval_mode() -> None:
    model = DICWithoutGroupwise(num_classes=16, exit_layers=(1, 12))
    trainer = DICTrainer(
        model=model,
        train_type="without_groupwise",
        exit_layers=(1, 12),
        device="cpu",
        gradient_accumulation_steps=1,
        use_amp=True,
    )
    assert trainer.use_amp is False
    dummy_loader = [
        {
            "pixel_values": torch.randn(2, 3, 224, 224),
            "input_ids": torch.zeros((2, 8), dtype=torch.long),
            "bbox": torch.zeros((2, 8, 4), dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
    ]
    trainer.train_stage2(dummy_loader, val_loader=dummy_loader, epochs=1)
    assert model.is_encoder_frozen is True
    assert model.backbone.training is False


def test_cached_dataset_and_multimodal_collator(tmp_path: Path) -> None:
    img_dir = tmp_path / "images" / "train"
    ocr_dir = tmp_path / "ocr_cache"
    img_dir.mkdir(parents=True)
    ocr_dir.mkdir(parents=True)

    records = []
    for idx in range(2):
        img_rel = f"images/train/{idx:05d}.png"
        Image.new("RGB", (64, 64), color=(240, 240, 240)).save(tmp_path / img_rel)
        records.append(
            {
                "id": idx,
                "image_path": img_rel,
                "label": idx,
                "words": ["invoice", "total", "100"],
                "boxes": [[10, 10, 100, 50], [110, 10, 200, 50], [210, 10, 300, 50]],
            }
        )

    with open(ocr_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    collator = MultimodalCollator(max_length=32)

    ds_single = RVLCDIPCachedDataset(data_dir=tmp_path, split="train", is_contrastive=False)
    batch_single = collator([ds_single[0], ds_single[1]])
    assert batch_single["pixel_values"].shape == (2, 3, 224, 224)
    assert batch_single["input_ids"].shape[0] == 2
    assert batch_single["bbox"].shape[-1] == 4
    assert batch_single["labels"].tolist() == [0, 1]

    ds_con = RVLCDIPCachedDataset(
        data_dir=tmp_path,
        split="train",
        transform=get_transforms(),
        is_contrastive=True,
    )
    batch_con = collator([ds_con[0], ds_con[1]])
    assert "view1" in batch_con and "view2" in batch_con
    assert batch_con["view1"]["pixel_values"].shape == (2, 3, 224, 224)
    assert batch_con["view2"]["pixel_values"].shape == (2, 3, 224, 224)
    assert torch.equal(batch_con["view1"]["input_ids"], batch_con["view2"]["input_ids"])


def test_ocr_ppdoclayout_root_dir_resolution() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "model" / "modules" / "ocr_ppdoclayout_ppocrv6small.py").read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parents[2]" in src


def test_end_to_end_main_dry_run_and_visualization(tmp_path: Path) -> None:
    import main as root_main

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = tmp_path / "exp_outputs"

    for weighting_mode in ("uniform", "subgraph_paper", "subgraph_repo"):
        exp1_dir = out_root / f"exp1_{weighting_mode}"
        args1 = Namespace(
            mode="exp1_hamed_ce",
            data_dir=str(tmp_path / "unused"),
            output_dir=str(out_root),
            exp1_dir=str(exp1_dir),
            exp2_dir=str(out_root / "exp2"),
            figures_dir=str(out_root / "figures"),
            dataset_name="jordyvl/rvl_cdip_100_examples_per_class",
            ocr_engine="tesseract",
            exit_layers=[0, 1, 4, 8, 12],
            num_classes=16,
            weighting_mode=weighting_mode,
            gamma=0.5,
            batch_size=2,
            grad_accum_steps=1,
            epochs=1,
            epochs_stage2=1,
            lr=5e-5,
            lr_stage2=3e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            temperature=0.1,
            smoothing_coeff=0.1,
            use_lars=False,
            probe_every_n_steps=1,
            landscape_grid_size=3,
            landscape_span=0.3,
            max_samples=4,
            num_workers=0,
            seed=42,
            device=device,
            use_amp=True,
            use_checkpointing=True,
            dry_run=True,
            run_stage2_probe=True,
            train_type="without_groupwise",
            save_dir=str(tmp_path / "ckpt"),
            checkpoint_path=str(tmp_path / "ckpt" / "best_model.pt"),
            criterion="max_confidence",
            threshold=0.8,
            sweep=False,
        )
        root_main.main_app(args1)
        log1 = exp1_dir / "gradient_conflict_log.json"
        assert log1.exists()
        payload1 = json.loads(log1.read_text(encoding="utf-8"))
        assert payload1["experiment"] == "exp1_hamed_ce"
        assert payload1["config"]["weighting_mode"] == weighting_mode
        assert payload1["summary"]["layer_0"]["active_exits"] == ["0", "1", "4", "8", "12"]
        assert len(payload1["step_logs"]) >= 1
        assert "loss_landscape" in payload1

    exp2_dir = out_root / "exp2"
    args2 = Namespace(**{**vars(args1), "mode": "exp2_supcon", "exp1_dir": str(out_root / "exp1_subgraph_paper"), "exp2_dir": str(exp2_dir)})
    root_main.main_app(args2)
    log2 = exp2_dir / "gradient_conflict_log.json"
    assert log2.exists()
    payload2 = json.loads(log2.read_text(encoding="utf-8"))
    assert payload2["experiment"] == "exp2_supcon"
    assert payload2["summary"]["layer_0"]["active_exits"] == ["0", "1", "4", "8", "12"]
    assert len(payload2["step_logs"]) >= 1
    assert "loss_landscape" in payload2

    args_vis = Namespace(**{**vars(args2), "mode": "visualize_all"})
    root_main.main_app(args_vis)
    fig_dir = out_root / "figures"
    expected_files = [
        "fig1_cosine_similarity_heatmaps.png",
        "fig1_cosine_similarity_heatmaps.pdf",
        "fig2_conflict_trajectories.png",
        "fig2_conflict_trajectories.pdf",
        "fig3_descent_alignment_hsr.png",
        "fig3_descent_alignment_hsr.pdf",
        "fig4_norm_dominance.png",
        "fig4_norm_dominance.pdf",
        "fig5_loss_landscapes_2d.png",
        "fig5_loss_landscapes_2d.pdf",
        "fig6_exit_performance.png",
        "fig6_exit_performance.pdf",
        "summary_table.csv",
        "summary_table.md",
    ]
    for fname in expected_files:
        fpath = fig_dir / fname
        assert fpath.exists() and fpath.stat().st_size > 0, f"Missing or empty: {fpath}"

    single_fig_dir = out_root / "figures_single"
    args_single_vis = Namespace(**{**vars(args2), "mode": "visualize_all", "exp2_dir": str(out_root / "missing_exp2"), "figures_dir": str(single_fig_dir)})
    root_main.main_app(args_single_vis)
    assert (single_fig_dir / "fig1_cosine_similarity_heatmaps.png").exists()
    assert (single_fig_dir / "summary_table.md").exists()


def test_prepare_rvl_subset_local_with_tesseract_and_resume(tmp_path: Path) -> None:
    from data_processing.prepare_rvl_subset import prepare_rvl_cdip_subset

    src_dir = tmp_path / "raw_src"
    (src_dir / "images" / "train").mkdir(parents=True)
    (src_dir / "labels").mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (128, 64), color=(255, 255, 255)).save(src_dir / "images" / "train" / f"{i}.png")
    (src_dir / "labels" / "train.txt").write_text("train/0.png 0\ntrain/1.png 1\n", encoding="utf-8")

    out_dir = tmp_path / "prepared"
    counts1 = prepare_rvl_cdip_subset(
        output_dir=out_dir,
        ocr_engine="tesseract",
        source_local_dir=src_dir,
    )
    assert counts1["train"] == 2
    cache_lines_1 = (out_dir / "ocr_cache" / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(cache_lines_1) == 2

    counts2 = prepare_rvl_cdip_subset(
        output_dir=out_dir,
        ocr_engine="tesseract",
        source_local_dir=src_dir,
    )
    assert counts2["train"] == 2
    cache_lines_2 = (out_dir / "ocr_cache" / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(cache_lines_2) == 2


def test_exp2_lars_and_non_zero_exit_layers(tmp_path: Path) -> None:
    from train_test.exp_runner import run_exp1_hamed_ce, run_exp2_supcon

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p1 = run_exp1_hamed_ce(
        data_dir=tmp_path,
        output_dir=tmp_path / "e1",
        exit_layers=(1, 4, 8, 10, 12),
        batch_size=2,
        grad_accum_steps=1,
        epochs=1,
        probe_every_n_steps=1,
        landscape_grid_size=3,
        max_samples=4,
        device=device,
        dry_run=True,
    )
    assert "layer_1" in p1["summary"]
    assert "layer_0" not in p1["summary"]

    p2 = run_exp2_supcon(
        data_dir=tmp_path,
        output_dir=tmp_path / "e2",
        exit_layers=(1, 4, 8, 10, 12),
        batch_size=2,
        grad_accum_steps=1,
        epochs=1,
        epochs_stage2=0,
        use_lars=True,
        probe_every_n_steps=1,
        landscape_grid_size=3,
        max_samples=4,
        device=device,
        dry_run=True,
        run_stage2_probe=False,
    )
    assert "layer_1" in p2["summary"]
    assert p2["stage2_logs"] == []


def test_separated_configs_and_cli_isolation() -> None:
    import main as root_main
    from utils.config import (
        EEBaselineConfig,
        ExperimentConfig,
        Stage1SupConConfig,
        Stage2ClassifierConfig,
        build_adamw_param_groups,
    )

    cfg = ExperimentConfig()
    assert cfg.exit_layers == (0, 1, 4, 8, 10, 12)
    assert cfg.ee_baseline == EEBaselineConfig()
    assert cfg.stage1 == Stage1SupConConfig()
    assert cfg.stage2 == Stage2ClassifierConfig()
    assert cfg.ee_baseline.warmup_ratio == 0.0
    assert cfg.ee_baseline.scheduler == "linear"
    assert cfg.ee_baseline.weighting_mode == "subgraph_repo"
    assert cfg.stage1.temperature == cfg.stage1.base_temperature == 0.1
    assert cfg.stage1.sampler == "pk"
    assert cfg.stage2.label_smoothing == 0.1

    parser = root_main.build_parser()
    parsed = parser.parse_args(["--mode", "run_both", "--s1_lr", "1e-5", "--s1_warmup_ratio", "0.2", "--s2_label_smoothing", "0.0"])
    assert parsed.ee_lr == 1e-4
    assert parsed.ee_warmup_ratio == 0.0
    assert parsed.s1_lr == 1e-5
    assert parsed.s1_warmup_ratio == 0.2
    assert parsed.s2_label_smoothing == 0.0

    mod = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    groups = build_adamw_param_groups(mod.named_parameters(), weight_decay=1e-4, exclude_bias_norm=True)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-4 and groups[0]["exclude_from_lars"] is False
    assert groups[1]["weight_decay"] == 0.0 and groups[1]["exclude_from_lars"] is True


def test_exit_weights_supcon_scale_lars_exclusion_and_pk_sampler() -> None:
    from data_processing.data_processing import PKBatchSampler, RandomGaussianBlur
    from train_test.exp_runner import compute_exit_weights
    from utils.lars_wrapper import LARSWrapper
    from utils.losses import MultiExitCELoss, SupConLoss

    class _DummyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Linear(4, 4)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

    m = _DummyLM()
    exits = (0, 1, 4, 8, 10, 12)
    w_paper = compute_exit_weights(m, exits, weighting_mode="subgraph_paper", gamma=0.5)
    assert math.isclose(w_paper[12], 0.5, rel_tol=1e-6)
    assert math.isclose(sum(w_paper[l] for l in (0, 1, 4, 8, 10)), 0.5, rel_tol=1e-6)
    assert w_paper[0] > w_paper[1] > w_paper[4] > w_paper[8] > w_paper[10]

    w_repo = compute_exit_weights(m, exits, weighting_mode="subgraph_repo", gamma=0.5)
    assert math.isclose(w_repo[12], 0.5, rel_tol=1e-6)
    assert math.isclose(sum(w_repo[l] for l in (0, 1, 4, 8, 10)), 0.1, rel_tol=1e-6)

    sc = SupConLoss(temperature=0.1)
    assert sc.base_temperature == 0.1
    ce = MultiExitCELoss(exit_layers=(0, 12), label_smoothing=0.1)
    assert math.isclose(ce.ce_fn.label_smoothing, 0.1, rel_tol=1e-6)

    p_decay = nn.Parameter(torch.ones(4, 4))
    p_bias = nn.Parameter(torch.ones(4) * 1e-3)
    opt = torch.optim.SGD(
        [
            {"params": [p_decay], "weight_decay": 0.0, "exclude_from_lars": False},
            {"params": [p_bias], "weight_decay": 0.0, "exclude_from_lars": True},
        ],
        lr=1.0,
    )
    lars = LARSWrapper(opt, eta=0.02, clip=True)
    p_decay.grad = torch.ones_like(p_decay) * 10.0
    p_bias.grad = torch.ones_like(p_bias) * 10.0
    lars.step()
    assert torch.allclose(p_bias.detach(), torch.ones(4) * 1e-3 - 10.0, atol=1e-4)
    assert p_decay.detach().mean().item() > -5.0

    labels = [c for c in range(16) for _ in range(5)]
    sampler = PKBatchSampler(labels, p=4, k=2, seed=42)
    batches = list(sampler)
    assert len(batches) == len(labels) // 8
    for b in batches:
        assert len(b) == 8
        bl = [labels[i] for i in b]
        assert len(set(bl)) == 4
        assert all(bl.count(c) == 2 for c in set(bl))

    blur = RandomGaussianBlur(p=1.0, sigma=(0.5, 1.0))
    img = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    out_img = blur(img)
    assert out_img.shape == img.shape and out_img.dtype == img.dtype


def test_no_comments_or_docstrings_in_new_files() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    new_files = [
        repo_root / "main.py",
        repo_root / "utils" / "gradient_conflict.py",
        repo_root / "data_processing" / "prepare_rvl_subset.py",
        repo_root / "train_test" / "exp_runner.py",
        repo_root / "train_test" / "plot_gradient_conflict.py",
        repo_root / "tests" / "test_gradient_conflict.py",
    ]
    for fpath in new_files:
        text = fpath.read_text(encoding="utf-8")
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        comments = [tok for tok in tokens if tok.type == tokenize.COMMENT]
        if fpath.name == "main.py":
            comments = [
                tok
                for tok in comments
                if not tok.line.lstrip().startswith('"') and not tok.string.lstrip("# ").startswith("Source:")
            ]
        elif fpath.name == "exp_runner.py":
            comments = [tok for tok in comments if not tok.string.lstrip("# ").startswith("Source:")]
        assert comments == [], f"Found comments in {fpath}: {comments}"

        tree = ast.parse(text)
        assert ast.get_docstring(tree) is None, f"Module docstring found in {fpath}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node) is None, f"Docstring found on {node.name} in {fpath}"
