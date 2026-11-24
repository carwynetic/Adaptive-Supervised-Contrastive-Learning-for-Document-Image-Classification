from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def enable_layoutlmv3_checkpointing(model: nn.Module) -> int:
    wrapped = 0
    for module in model.modules():
        if module.__class__.__name__ == "LayoutLMv3Layer":
            if getattr(module, "_manual_ckpt_wrapped", False):
                wrapped += 1
                continue
            orig_forward = module.forward

            def _make_forward(mod: nn.Module, fn: Callable[..., Any]) -> Callable[..., Any]:
                def _wrapped_forward(
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    output_attentions: bool = False,
                    rel_pos: Optional[torch.Tensor] = None,
                    rel_2d_pos: Optional[torch.Tensor] = None,
                    **kwargs: Any,
                ) -> Any:
                    if mod.training and torch.is_grad_enabled() and hidden_states.requires_grad:
                        def _custom_fn(
                            hs: torch.Tensor,
                            attn_mask: Optional[torch.Tensor],
                            r_pos: Optional[torch.Tensor],
                            r_2d_pos: Optional[torch.Tensor],
                        ) -> Any:
                            return fn(
                                hs,
                                attn_mask,
                                output_attentions=output_attentions,
                                rel_pos=r_pos,
                                rel_2d_pos=r_2d_pos,
                                **kwargs,
                            )

                        return checkpoint(
                            _custom_fn,
                            hidden_states,
                            attention_mask,
                            rel_pos,
                            rel_2d_pos,
                            use_reentrant=False,
                        )
                    return fn(
                        hidden_states,
                        attention_mask,
                        output_attentions=output_attentions,
                        rel_pos=rel_pos,
                        rel_2d_pos=rel_2d_pos,
                        **kwargs,
                    )

                return _wrapped_forward

            module.forward = _make_forward(module, orig_forward)
            setattr(module, "_manual_ckpt_wrapped", True)
            wrapped += 1
    return wrapped


def _resolve_layoutlmv3_base(model: nn.Module) -> nn.Module:
    if hasattr(model, "layoutlmv3"):
        return model.layoutlmv3
    if hasattr(model, "backbone") and hasattr(model.backbone, "model"):
        return model.backbone.model
    return model


def extract_shared_layer_param_groups(
    model: nn.Module,
    probe_layers: Sequence[int] = (0, 1, 4, 8),
) -> Dict[str, List[nn.Parameter]]:
    base = _resolve_layoutlmv3_base(model)
    groups: Dict[str, List[nn.Parameter]] = {}

    for layer_idx in probe_layers:
        if layer_idx == 0:
            params: List[nn.Parameter] = []
            for name, p in base.named_parameters():
                if not p.requires_grad:
                    continue
                if (
                    name.startswith("patch_embed.")
                    or name.startswith("LayerNorm.")
                    or name.startswith("embeddings.LayerNorm.")
                    or name.startswith("embeddings.x_position_embeddings.")
                    or name.startswith("embeddings.y_position_embeddings.")
                    or name.startswith("embeddings.h_position_embeddings.")
                    or name.startswith("embeddings.w_position_embeddings.")
                ):
                    params.append(p)
            if not params and hasattr(base, "embeddings"):
                params = [p for p in base.embeddings.parameters() if p.requires_grad]
            if params:
                groups["layer_0"] = params
        else:
            encoder = getattr(base, "encoder", None)
            if encoder is not None and hasattr(encoder, "layer"):
                idx0 = layer_idx - 1
                if 0 <= idx0 < len(encoder.layer):
                    params = [p for p in encoder.layer[idx0].parameters() if p.requires_grad]
                    if params:
                        groups[f"layer_{layer_idx}"] = params
    return groups


def extract_shared_backbone_params(model: nn.Module) -> List[nn.Parameter]:
    base = _resolve_layoutlmv3_base(model)
    params: List[nn.Parameter] = []
    for name, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if "early_exits" in name or "exit_embeddings" in name or "lte_" in name:
            continue
        if "word_embeddings" in name:
            continue
        params.append(p)
    return params


def _exit_sort_key(item: Any) -> Tuple[int, Any]:
    s = str(item)
    if s.lstrip("-").isdigit():
        return (0, int(s))
    return (1, s)


def _layer_sort_key(name: str) -> Tuple[int, str]:
    tail = name.split("_")[-1]
    if tail.lstrip("-").isdigit():
        return (int(tail), name)
    return (999999, name)


class GradientConflictProber:
    def __init__(
        self,
        probe_param_groups: Dict[str, List[nn.Parameter]],
        eps: float = 1e-12,
        max_condition_number: float = 1e6,
    ) -> None:
        self.probe_param_groups = probe_param_groups
        self.eps = eps
        self.max_condition_number = max_condition_number
        self.history: List[Dict[str, Any]] = []

    def compute_metrics_from_vectors(
        self,
        grad_vectors: Dict[Any, torch.Tensor],
        weights: Optional[Dict[Any, float]] = None,
    ) -> Dict[str, Any]:
        active_keys: List[Any] = []
        raw_vecs: List[torch.Tensor] = []
        weighted_vecs: List[torch.Tensor] = []
        unweighted_norms: Dict[str, float] = {}
        weighted_norms: Dict[str, float] = {}

        for k in sorted(grad_vectors.keys(), key=_exit_sort_key):
            vec = grad_vectors[k].detach().to(dtype=torch.float64, device="cpu").reshape(-1)
            norm_val = float(torch.linalg.vector_norm(vec).item())
            if not math.isfinite(norm_val) or norm_val <= self.eps:
                continue
            w = float(weights.get(k, weights.get(str(k), 1.0))) if weights is not None else 1.0
            w_vec = vec * w
            w_norm = float(torch.linalg.vector_norm(w_vec).item())
            active_keys.append(k)
            raw_vecs.append(vec)
            weighted_vecs.append(w_vec)
            unweighted_norms[str(k)] = norm_val
            weighted_norms[str(k)] = w_norm

        k_count = len(active_keys)
        key_strs = [str(k) for k in active_keys]
        if k_count < 2:
            return {
                "active_exits": key_strs,
                "pairwise_cosine": [[1.0]] if k_count == 1 else [],
                "mean_pairwise_cosine": 1.0 if k_count == 1 else 0.0,
                "conflict_rate": 0.0,
                "descent_alignment": {key_strs[0]: 1.0} if k_count == 1 else {},
                "harmful_step": {key_strs[0]: False} if k_count == 1 else {},
                "cancellation_ratio": 0.0,
                "dominance_ratio": 1.0,
                "unweighted_norms": unweighted_norms,
                "weighted_norms": weighted_norms,
                "condition_number": 1.0,
                "singular_values": [weighted_norms[key_strs[0]]] if k_count == 1 else [],
            }

        raw_mat = torch.stack(raw_vecs, dim=0)
        weighted_mat = torch.stack(weighted_vecs, dim=0)

        raw_norms_t = torch.linalg.vector_norm(raw_mat, dim=1, keepdim=True).clamp_min(self.eps)
        unit_mat = raw_mat / raw_norms_t
        cos_mat = torch.clamp(unit_mat @ unit_mat.T, -1.0, 1.0)

        off_diag_vals: List[float] = []
        conflict_count = 0
        for i in range(k_count):
            for j in range(i + 1, k_count):
                val = float(cos_mat[i, j].item())
                off_diag_vals.append(val)
                if val < 0.0:
                    conflict_count += 1

        mean_pairwise_cosine = float(sum(off_diag_vals) / len(off_diag_vals))
        conflict_rate = float(conflict_count / len(off_diag_vals))

        g_total = weighted_mat.sum(dim=0)
        g_total_norm = float(torch.linalg.vector_norm(g_total).item())
        sum_w_norms = float(sum(weighted_norms[ks] for ks in key_strs))
        cancellation_ratio = float(max(0.0, min(1.0, 1.0 - g_total_norm / (sum_w_norms + self.eps))))

        descent_alignment: Dict[str, float] = {}
        harmful_step: Dict[str, bool] = {}
        for idx, ks in enumerate(key_strs):
            if g_total_norm <= self.eps:
                align = 0.0
            else:
                dot_val = float(torch.dot(raw_mat[idx], g_total).item())
                align = float(max(-1.0, min(1.0, dot_val / (unweighted_norms[ks] * g_total_norm + self.eps))))
            descent_alignment[ks] = align
            harmful_step[ks] = bool(align < 0.0)

        w_norm_list = [weighted_norms[ks] for ks in key_strs]
        dominance_ratio = float(max(w_norm_list) / (min(w_norm_list) + self.eps))

        gram = weighted_mat @ weighted_mat.T
        gram = 0.5 * (gram + gram.T)
        eigvals = torch.linalg.eigvalsh(gram)
        eigvals = torch.clamp(eigvals, min=0.0)
        singular_values = torch.sqrt(eigvals).flip(0)
        s_max = float(singular_values[0].item())
        s_min = float(singular_values[-1].item())
        if s_min <= self.eps * max(1.0, s_max):
            cond_num = float(self.max_condition_number)
        else:
            cond_num = float(min(self.max_condition_number, s_max / s_min))

        return {
            "active_exits": key_strs,
            "pairwise_cosine": [[float(cos_mat[i, j].item()) for j in range(k_count)] for i in range(k_count)],
            "mean_pairwise_cosine": mean_pairwise_cosine,
            "conflict_rate": conflict_rate,
            "descent_alignment": descent_alignment,
            "harmful_step": harmful_step,
            "cancellation_ratio": cancellation_ratio,
            "dominance_ratio": dominance_ratio,
            "unweighted_norms": unweighted_norms,
            "weighted_norms": weighted_norms,
            "condition_number": cond_num,
            "singular_values": [float(v.item()) for v in singular_values],
        }

    def probe(
        self,
        step: int,
        exit_losses: Dict[Any, torch.Tensor],
        weights: Optional[Dict[Any, float]] = None,
    ) -> Dict[str, Any]:
        ordered_layers = list(self.probe_param_groups.keys())
        unique_params: List[nn.Parameter] = []
        param_to_idx: Dict[int, int] = {}
        layer_param_indices: Dict[str, List[int]] = {}

        for layer_name in ordered_layers:
            indices: List[int] = []
            for p in self.probe_param_groups[layer_name]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid not in param_to_idx:
                    param_to_idx[pid] = len(unique_params)
                    unique_params.append(p)
                indices.append(param_to_idx[pid])
            if indices:
                layer_param_indices[layer_name] = indices

        per_layer_vectors: Dict[str, Dict[Any, torch.Tensor]] = {
            layer_name: {} for layer_name in layer_param_indices
        }

        if unique_params:
            for exit_key, loss_tensor in exit_losses.items():
                if not loss_tensor.requires_grad:
                    continue
                grads = torch.autograd.grad(
                    loss_tensor,
                    unique_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                for layer_name, indices in layer_param_indices.items():
                    chunks: List[torch.Tensor] = []
                    has_nonzero = False
                    for idx in indices:
                        g = grads[idx]
                        if g is not None:
                            has_nonzero = True
                            chunks.append(g.detach().to(dtype=torch.float32, device="cpu").reshape(-1))
                        else:
                            chunks.append(torch.zeros(unique_params[idx].numel(), dtype=torch.float32))
                    if has_nonzero and chunks:
                        per_layer_vectors[layer_name][exit_key] = torch.cat(chunks, dim=0)

        layer_metrics: Dict[str, Any] = {}
        for layer_name, vec_dict in per_layer_vectors.items():
            layer_metrics[layer_name] = self.compute_metrics_from_vectors(
                grad_vectors=vec_dict,
                weights=weights,
            )

        record = {"step": int(step), "layers": layer_metrics}
        self.history.append(record)
        return record

    def summarize(self) -> Dict[str, Any]:
        if not self.history:
            return {}

        layer_names = sorted(
            {ln for rec in self.history for ln in rec["layers"].keys()},
            key=_layer_sort_key,
        )
        summary: Dict[str, Any] = {}

        for ln in layer_names:
            entries = [rec["layers"][ln] for rec in self.history if ln in rec["layers"]]
            if not entries:
                continue

            ref_exits = entries[0]["active_exits"]
            k_count = len(ref_exits)

            cos_vals = [e["mean_pairwise_cosine"] for e in entries]
            cr_vals = [e["conflict_rate"] for e in entries]
            cancel_vals = [e["cancellation_ratio"] for e in entries]
            dom_vals = [e["dominance_ratio"] for e in entries]
            cond_vals = [e["condition_number"] for e in entries]

            aligned_entries = [e for e in entries if e["active_exits"] == ref_exits]
            if aligned_entries and k_count > 0:
                cos_stack = np.array([e["pairwise_cosine"] for e in aligned_entries], dtype=np.float64)
                mean_cos_mat = cos_stack.mean(axis=0).tolist()
                pair_cr_mat = (cos_stack < 0.0).mean(axis=0).tolist()
            else:
                mean_cos_mat = []
                pair_cr_mat = []

            mean_align: Dict[str, float] = {}
            hsr: Dict[str, float] = {}
            mean_w_norms: Dict[str, float] = {}
            mean_u_norms: Dict[str, float] = {}
            for ex in ref_exits:
                a_list = [e["descent_alignment"][ex] for e in entries if ex in e["descent_alignment"]]
                h_list = [1.0 if e["harmful_step"][ex] else 0.0 for e in entries if ex in e["harmful_step"]]
                wn_list = [e["weighted_norms"][ex] for e in entries if ex in e["weighted_norms"]]
                un_list = [e["unweighted_norms"][ex] for e in entries if ex in e["unweighted_norms"]]
                if a_list:
                    mean_align[ex] = float(np.mean(a_list))
                if h_list:
                    hsr[ex] = float(np.mean(h_list))
                if wn_list:
                    mean_w_norms[ex] = float(np.mean(wn_list))
                if un_list:
                    mean_u_norms[ex] = float(np.mean(un_list))

            summary[ln] = {
                "num_steps": len(entries),
                "active_exits": ref_exits,
                "mean_pairwise_cosine": float(np.mean(cos_vals)),
                "std_pairwise_cosine": float(np.std(cos_vals)),
                "conflict_rate": float(np.mean(cr_vals)),
                "pairwise_cosine_matrix": mean_cos_mat,
                "pairwise_conflict_rate_matrix": pair_cr_mat,
                "mean_descent_alignment": mean_align,
                "harmful_step_ratio": hsr,
                "mean_cancellation_ratio": float(np.mean(cancel_vals)),
                "max_cancellation_ratio": float(np.max(cancel_vals)),
                "mean_dominance_ratio": float(np.mean(dom_vals)),
                "max_dominance_ratio": float(np.max(dom_vals)),
                "mean_weighted_norms": mean_w_norms,
                "mean_unweighted_norms": mean_u_norms,
                "mean_condition_number": float(np.mean(cond_vals)),
                "median_condition_number": float(np.median(cond_vals)),
                "max_condition_number": float(np.max(cond_vals)),
            }

        return summary


def _filter_normalize_direction(
    direction: List[torch.Tensor],
    params: List[torch.Tensor],
    eps: float = 1e-12,
) -> List[torch.Tensor]:
    normalized: List[torch.Tensor] = []
    for d, p in zip(direction, params):
        if p.ndim >= 2:
            p_flat = p.detach().float().view(p.shape[0], -1)
            d_flat = d.detach().float().view(d.shape[0], -1)
            p_norms = torch.linalg.vector_norm(p_flat, dim=1, keepdim=True)
            d_norms = torch.linalg.vector_norm(d_flat, dim=1, keepdim=True)
            scale = torch.where(d_norms > eps, p_norms / (d_norms + eps), torch.zeros_like(d_norms))
            d_scaled = (d_flat * scale).view_as(p)
            normalized.append(d_scaled)
        else:
            normalized.append(torch.zeros_like(p, dtype=torch.float32))
    return normalized


def compute_2d_loss_landscape(
    model: nn.Module,
    eval_losses_fn: Callable[[], Dict[Any, torch.Tensor]],
    shallow_key: Any,
    deep_key: Any,
    weights: Optional[Dict[Any, float]] = None,
    grid_size: int = 11,
    span: float = 0.5,
    target_params: Optional[List[nn.Parameter]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    params = target_params if target_params is not None else extract_shared_backbone_params(model)
    if not params:
        params = [p for p in model.parameters() if p.requires_grad]

    active_params: List[nn.Parameter] = []
    orig_weights_cpu: List[torch.Tensor] = []
    eps = 1e-12

    try:
        with torch.enable_grad():
            losses_init = eval_losses_fn()
            loss_shallow = losses_init[shallow_key]
            loss_deep = losses_init[deep_key]

            g_s_raw = torch.autograd.grad(loss_shallow, params, retain_graph=True, allow_unused=True)
            g_d_raw = torch.autograd.grad(loss_deep, params, retain_graph=False, allow_unused=True)
            del losses_init, loss_shallow, loss_deep

        shared_indices = [
            idx
            for idx, (gs, gd) in enumerate(zip(g_s_raw, g_d_raw))
            if gs is not None and gd is not None
        ]
        if target_params is None and shared_indices:
            active_params = [params[i] for i in shared_indices]
            g_s_sel = [g_s_raw[i] for i in shared_indices]
            g_d_sel = [g_d_raw[i] for i in shared_indices]
        else:
            active_params = list(params)
            g_s_sel = list(g_s_raw)
            g_d_sel = list(g_d_raw)
        del g_s_raw, g_d_raw

        orig_weights_cpu = [p.detach().float().cpu().clone() for p in active_params]

        g_s = [
            g.detach().float().cpu() if g is not None else torch.zeros_like(w0)
            for g, w0 in zip(g_s_sel, orig_weights_cpu)
        ]
        g_d = [
            g.detach().float().cpu() if g is not None else torch.zeros_like(w0)
            for g, w0 in zip(g_d_sel, orig_weights_cpu)
        ]
        del g_s_sel, g_d_sel

        dot_sd = sum(float((gs * gd).sum().item()) for gs, gd in zip(g_s, g_d))
        norm_s_sq = sum(float((gs * gs).sum().item()) for gs in g_s)

        if norm_s_sq <= eps:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            g_s = [torch.randn(w0.shape, generator=gen, dtype=torch.float32) for w0 in orig_weights_cpu]
            norm_s_sq = sum(float((gs * gs).sum().item()) for gs in g_s)

        proj = dot_sd / (norm_s_sq + eps)
        d2_raw = [gd - proj * gs for gd, gs in zip(g_d, g_s)]
        del g_d
        norm_d2_sq = sum(float((v * v).sum().item()) for v in d2_raw)

        if norm_d2_sq <= eps:
            gen = torch.Generator(device="cpu").manual_seed(seed + 1)
            rand_d = [torch.randn(w0.shape, generator=gen, dtype=torch.float32) for w0 in orig_weights_cpu]
            dot_rs = sum(float((r * gs).sum().item()) for r, gs in zip(rand_d, g_s))
            d2_raw = [r - (dot_rs / (norm_s_sq + eps)) * gs for r, gs in zip(rand_d, g_s)]
            del rand_d

        d1_cpu = _filter_normalize_direction(g_s, orig_weights_cpu, eps=eps)
        del g_s
        d2_cpu = _filter_normalize_direction(d2_raw, orig_weights_cpu, eps=eps)
        del d2_raw

        alphas = np.linspace(-span, span, grid_size).tolist()
        betas = np.linspace(-span, span, grid_size).tolist()

        shallow_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        deep_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        total_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

        with torch.no_grad():
            for i, beta in enumerate(betas):
                for j, alpha in enumerate(alphas):
                    for p, w0, u1, u2 in zip(active_params, orig_weights_cpu, d1_cpu, d2_cpu):
                        updated = w0 + alpha * u1 + beta * u2
                        p.copy_(updated.to(device=p.device, dtype=p.dtype))
                    step_losses = eval_losses_fn()
                    l_s = float(step_losses[shallow_key].detach().float().item())
                    l_d = float(step_losses[deep_key].detach().float().item())
                    if weights is not None:
                        l_tot = sum(
                            float(weights.get(k, weights.get(str(k), 1.0 / len(step_losses))))
                            * float(v.detach().float().item())
                            for k, v in step_losses.items()
                        )
                    else:
                        l_tot = float(
                            sum(float(v.detach().float().item()) for v in step_losses.values())
                            / max(1, len(step_losses))
                        )
                    shallow_grid[i, j] = l_s
                    deep_grid[i, j] = l_d
                    total_grid[i, j] = l_tot
    finally:
        if active_params and orig_weights_cpu:
            with torch.no_grad():
                for p, w0 in zip(active_params, orig_weights_cpu):
                    p.copy_(w0.to(device=p.device, dtype=p.dtype))
        if was_training:
            model.train()

    s_idx = np.unravel_index(int(np.argmin(shallow_grid)), shallow_grid.shape)
    d_idx = np.unravel_index(int(np.argmin(deep_grid)), deep_grid.shape)
    t_idx = np.unravel_index(int(np.argmin(total_grid)), total_grid.shape)

    shallow_min_coord = [float(alphas[s_idx[1]]), float(betas[s_idx[0]])]
    deep_min_coord = [float(alphas[d_idx[1]]), float(betas[d_idx[0]])]
    total_min_coord = [float(alphas[t_idx[1]]), float(betas[t_idx[0]])]
    min_distance = float(
        math.hypot(
            shallow_min_coord[0] - deep_min_coord[0],
            shallow_min_coord[1] - deep_min_coord[1],
        )
    )

    return {
        "shallow_exit": str(shallow_key),
        "deep_exit": str(deep_key),
        "alphas": [float(a) for a in alphas],
        "betas": [float(b) for b in betas],
        "shallow_loss": shallow_grid.tolist(),
        "deep_loss": deep_grid.tolist(),
        "total_loss": total_grid.tolist(),
        "shallow_min_coord": shallow_min_coord,
        "deep_min_coord": deep_min_coord,
        "total_min_coord": total_min_coord,
        "min_distance": min_distance,
    }
