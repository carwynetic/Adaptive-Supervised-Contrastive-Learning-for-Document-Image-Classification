"""
Anytime Early-Exiting Evaluation Pipeline for Document Image Classification (DIC).
Đánh giá hiệu năng suy luận thích ứng (Adaptive Inference):
    1. Per-Exit Static Accuracy: Đo Top-1 Accuracy độc lập tại từng tầng [1, 4, 8, 10, 12].
    2. Adaptive Dynamic Inference: Cho phép mẫu thoát sớm dựa trên các tiêu chí:
        - Max-Confidence (MSP)
        - Shannon Entropy
        - Top 1-2 Margin
    3. Trade-off Curve Generator: Quét ngưỡng (Threshold sweep) để đo mức độ tiết kiệm FLOPs/Độ trễ.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Đảm bảo đường dẫn gốc của project luôn nằm trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.DIC_without_groupwise import DICWithoutGroupwise
from model.modules.ee_modules import entropy, max_confidence, top12_margin
from utils.dataloader import build_dataloader


class EarlyExitEvaluator:
    def __init__(
        self,
        model: nn.Module,
        exit_layers: Sequence[int] = (0, 1, 4, 8, 10, 12),
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()
        self.exit_layers = tuple(sorted(exit_layers))

    # ==========================================================================
    # 1. TRÍCH XUẤT TOÀN BỘ LOGITS TRÊN TẬP DỮ LIỆU
    # ==========================================================================

    @torch.no_grad()
    def collect_all_logits(self, dataloader: DataLoader) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Duyệt qua tập dữ liệu và thu thập toàn bộ logits tại 5 tầng exit cùng ground truth.

        Returns:
            all_logits: Dict[layer -> Tensor (N, num_classes)]
            all_labels: Tensor (N,)
        """
        logits_collector: Dict[int, List[torch.Tensor]] = {l: [] for l in self.exit_layers}
        labels_collector: List[torch.Tensor] = []

        for batch in dataloader:
            labels = batch["labels"].to(self.device)
            labels_collector.append(labels)

            pixel_values = batch["pixel_values"].to(self.device)
            b = pixel_values.shape[0]

            if "input_ids" in batch:
                input_ids = batch["input_ids"].to(self.device)
                bbox = batch["bbox"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            else:
                input_ids = torch.zeros((b, 1), dtype=torch.long, device=self.device)
                bbox = torch.zeros((b, 1, 4), dtype=torch.long, device=self.device)
                attention_mask = torch.ones((b, 1), dtype=torch.long, device=self.device)

            out = self.model(
                input_ids=input_ids,
                bbox=bbox,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                compute_projections=False,
            )

            for l in self.exit_layers:
                logits_collector[l].append(out.logits_dict[l].detach())

        all_logits = {l: torch.cat(logits_collector[l], dim=0) for l in self.exit_layers}
        all_labels = torch.cat(labels_collector, dim=0)

        return all_logits, all_labels

    # ==========================================================================
    # 2. ĐÁNH GIÁ TĨNH TỪNG TẦNG (PER-EXIT STATIC EVALUATION)
    # ==========================================================================

    def evaluate_static_exits(
        self, all_logits: Dict[int, torch.Tensor], labels: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """
        Đo đạc độ chính xác Top-1, độ tin cậy trung bình và entropy tại từng tầng.
        """
        results = {}
        total_samples = labels.size(0)

        for l in self.exit_layers:
            logits = all_logits[l]
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            acc = (correct / total_samples) * 100.0

            conf = max_confidence(logits).mean().item()
            ent = entropy(logits).mean().item()

            results[l] = {
                "accuracy": acc,
                "avg_confidence": conf,
                "avg_entropy": ent,
            }

        return results

    # ==========================================================================
    # 3. SUY LUẬN THÍCH ỨNG ĐỘNG (ADAPTIVE EARLY EXITING)
    # ==========================================================================

    def evaluate_adaptive_inference(
        self,
        all_logits: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        criterion: str = "max_confidence",
        threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Cho phép mẫu thoát sớm tại tầng đầu tiên thỏa mãn tiêu chuẩn ngưỡng.

        Args:
            all_logits: Dict[layer -> Tensor (N, num_classes)]
            labels: Tensor (N,)
            criterion: 'max_confidence', 'entropy', hoặc 'top12_margin'
            threshold: Ngưỡng tự tin để quyết định thoát sớm.

        Returns:
            Dict chứa accuracy tổng thể, phân bố mẫu thoát tại các tầng, và tầng thoát trung bình.
        """
        num_samples = labels.size(0)
        final_preds = torch.zeros(num_samples, dtype=torch.long, device=labels.device)
        exit_at_layer = torch.zeros(num_samples, dtype=torch.long, device=labels.device)
        is_exited = torch.zeros(num_samples, dtype=torch.bool, device=labels.device)

        # Duyệt tuần tự qua các tầng từ nông đến sâu
        for l in self.exit_layers:
            # Những mẫu chưa thoát
            active_mask = ~is_exited
            if not active_mask.any():
                break

            logits = all_logits[l][active_mask]

            if l == self.exit_layers[-1]:
                # Tầng cuối cùng: Bắt buộc tất cả các mẫu còn lại phải thoát tại đây
                should_exit = torch.ones(logits.size(0), dtype=torch.bool, device=labels.device)
            else:
                # Tính giá trị đo độ tự tin tại tầng trung gian
                if criterion == "max_confidence":
                    score = max_confidence(logits)
                    should_exit = score >= threshold
                elif criterion == "entropy":
                    score = entropy(logits)
                    should_exit = score <= threshold
                elif criterion == "top12_margin":
                    score = top12_margin(logits)
                    should_exit = score >= threshold
                else:
                    raise ValueError(f"Không hỗ trợ criterion: {criterion}")

            # Lưu lại dự đoán của những mẫu được phép thoát
            active_indices = torch.nonzero(active_mask).squeeze(-1)
            exited_indices = active_indices[should_exit]

            if exited_indices.numel() > 0:
                preds = torch.argmax(logits[should_exit], dim=-1)
                final_preds[exited_indices] = preds
                exit_at_layer[exited_indices] = l
                is_exited[exited_indices] = True

        # Tính toán các chỉ số
        correct = (final_preds == labels).sum().item()
        adaptive_acc = (correct / num_samples) * 100.0
        avg_exit_layer = exit_at_layer.float().mean().item()

        # Tính tỷ lệ phần trăm mẫu thoát tại từng tầng
        exit_distribution = {}
        for l in self.exit_layers:
            count = (exit_at_layer == l).sum().item()
            exit_distribution[l] = (count / num_samples) * 100.0

        # Ước lượng mức tiết kiệm tính toán (so với việc luôn chạy đủ 12 tầng)
        deepest_layer = self.exit_layers[-1]
        compute_saved = (1.0 - (avg_exit_layer / deepest_layer)) * 100.0

        return {
            "adaptive_accuracy": adaptive_acc,
            "avg_exit_layer": avg_exit_layer,
            "compute_saved_percent": compute_saved,
            "exit_distribution": exit_distribution,
        }

    # ==========================================================================
    # 4. QUÉT DẢI NGƯỠNG (THRESHOLD SWEEP)
    # ==========================================================================

    def sweep_thresholds(
        self,
        all_logits: Dict[int, torch.Tensor],
        labels: torch.Tensor,
        criterion: str = "max_confidence",
        thresholds: Optional[Sequence[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Quét nhiều giá trị ngưỡng để thiết lập đường cong Trade-off giữa Tốc độ và Accuracy.
        """
        if thresholds is None:
            if criterion == "entropy":
                # Entropy giảm dần từ cao về thấp
                thresholds = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
            else:
                thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]

        sweep_results = []
        for t in thresholds:
            res = self.evaluate_adaptive_inference(all_logits, labels, criterion=criterion, threshold=t)
            res["threshold"] = t
            sweep_results.append(res)

        return sweep_results


# ==============================================================================
# HÀM CHÍNH VÀ CLI ENTRYPOINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá Anytime Early-Exiting Inference cho DIC")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/run_01/best_model.pt", help="Đường dẫn file .pt")
    parser.add_argument("--data_dir", type=str, default="data/rvl-cdip", help="Đường dẫn thư mục tập dữ liệu")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size khi đánh giá")
    parser.add_argument("--exit_layers", type=int, nargs="+", default=[0, 1, 4, 8, 10, 12])
    parser.add_argument("--criterion", type=str, default="max_confidence", choices=["max_confidence", "entropy", "top12_margin"])
    parser.add_argument("--threshold", type=float, default=0.8, help="Ngưỡng tự tin cố định")
    parser.add_argument("--sweep", action="store_true", help="Bật chế độ quét dải ngưỡng để vẽ Trade-off curve")
    parser.add_argument("--dry_run", action="store_true", help="Chạy thử với dữ liệu giả lập")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("ĐÁNH GIÁ SUY LUẬN THÍCH ỨNG (ANYTIME EARLY EXITING INFERENCE)")
    print("=" * 75)
    print(f"Tiêu chí quyết định thoát : {args.criterion.upper()}")
    print(f"Danh sách tầng Exits      : {args.exit_layers}")
    print(f"Thiết bị                  : {args.device}")
    print("=" * 75)

    # 1. Khởi tạo mô hình
    model = DICWithoutGroupwise(num_classes=16, exit_layers=args.exit_layers)

    # Tải trọng số từ checkpoint nếu có
    if Path(args.checkpoint_path).exists():
        ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[+] Đã tải trọng số từ: {args.checkpoint_path}")
    else:
        print(f"[!] Cảnh báo: Không tìm thấy checkpoint tại {args.checkpoint_path}, sử dụng trọng số hiện tại.")

    evaluator = EarlyExitEvaluator(model=model, exit_layers=args.exit_layers, device=args.device)

    # 2. Chuẩn bị dữ liệu kiểm thử
    if args.dry_run:
        print("[!] Chế độ DRY-RUN: Sử dụng dữ liệu giả lập để kiểm tra luồng.")
        dummy_loader = [
            {
                "pixel_values": torch.randn(4, 3, 224, 224),
                "input_ids": torch.zeros((4, 16), dtype=torch.long),
                "bbox": torch.zeros((4, 16, 4), dtype=torch.long),
                "labels": torch.tensor([0, 5, 10, 15]),
            }
            for _ in range(3)
        ]
        dataloader = dummy_loader
    else:
        print(f"[+] Tải tập kiểm thử test từ: {args.data_dir}")
        dataloader = build_dataloader(
            data_dir=args.data_dir,
            split="test",
            batch_size=args.batch_size,
            is_contrastive=False,
            shuffle=False,
        )

    # 3. Trích xuất toàn bộ Logits
    print("\n[1/3] Đang trích xuất Logits từ các tầng Exits...")
    all_logits, all_labels = evaluator.collect_all_logits(dataloader)
    print(f"[+] Thu thập thành công {all_labels.size(0)} mẫu.")

    # 4. Đánh giá tĩnh từng tầng (Static evaluation)
    print("\n[2/3] HIỆU NĂNG ĐỘC LẬP TỪNG TẦNG EXIT (STATIC EVALUATION):")
    print("-" * 75)
    static_results = evaluator.evaluate_static_exits(all_logits, all_labels)
    for l, metrics in static_results.items():
        print(
            f"  Exit L{l:2d}: Top-1 Acc = {metrics['accuracy']:6.2f}% | "
            f"Avg Confidence = {metrics['avg_confidence']:.4f} | "
            f"Avg Entropy = {metrics['avg_entropy']:.4f}"
        )

    # 5. Đánh giá suy luận động thích ứng (Adaptive inference)
    print(f"\n[3/3] HIỆU NĂNG SUY LUẬN THÍCH ỨNG (ADAPTIVE EARLY-EXITING):")
    print("-" * 75)

    if args.sweep:
        print(f"[+] Quét dải ngưỡng với tiêu chí '{args.criterion}':\n")
        sweep_data = evaluator.sweep_thresholds(all_logits, all_labels, criterion=args.criterion)
        print(f"{'Ngưỡng (t)':^12} | {'Adaptive Acc':^14} | {'Avg Exit Layer':^16} | {'Compute Saved':^15}")
        print("-" * 65)
        for s in sweep_data:
            print(
                f"{s['threshold']:^12.2f} | {s['adaptive_accuracy']:^12.2f}% | "
                f"L{s['avg_exit_layer']:^14.2f} | {s['compute_saved_percent']:^13.2f}%"
            )
    else:
        res = evaluator.evaluate_adaptive_inference(
            all_logits, all_labels, criterion=args.criterion, threshold=args.threshold
        )
        print(f"  Ngưỡng áp dụng ({args.criterion}) : {args.threshold:.2f}")
        print(f"  Độ chính xác thích ứng (Adaptive Acc): {res['adaptive_accuracy']:.2f}%")
        print(f"  Tầng thoát trung bình (Avg Layer)   : L{res['avg_exit_layer']:.2f} / L12")
        print(f"  Mức độ tiết kiệm tính toán (Saved)  : {res['compute_saved_percent']:.2f}%")
        print("\n  Phân bố mẫu thoát tại từng tầng:")
        for l, pct in res["exit_distribution"].items():
            print(f"    - Tầng L{l:2d}: {pct:6.2f}%")

    print("\n" + "=" * 75)
    print("HOÀN TẤT ĐÁNH GIÁ SƠ BỘ THÀNH CÔNG!")
    print("=" * 75)


if __name__ == "__main__":
    main()