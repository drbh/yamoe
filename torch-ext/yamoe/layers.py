import math
import torch
import time
from ._ops import ops


class _ExpertsFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_flat: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights_dense: torch.Tensor,
        gate_up_proj: torch.Tensor,
        gate_up_proj_bias: torch.Tensor,
        down_proj: torch.Tensor,
        down_proj_bias: torch.Tensor,
        expert_capacity: int,
        num_experts: int,
        top_k: int,
        enable_router_grads: bool,
    ):
        out = ops.experts(
            hidden_flat,
            router_indices,
            routing_weights_dense,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            expert_capacity,
            num_experts,
            top_k,
        )
        ctx.expert_capacity = expert_capacity
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.enable_router_grads = bool(enable_router_grads)
        ctx.save_for_backward(
            hidden_flat,
            router_indices,
            routing_weights_dense,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            hidden_flat,
            router_indices,
            routing_weights_dense,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
        ) = ctx.saved_tensors

        (
            grad_hidden_flat,
            grad_routing_weights,
            grad_gate_up_proj,
            grad_gate_up_proj_bias,
            grad_down_proj,
            grad_down_proj_bias,
        ) = ops.experts_backward(
            grad_output,
            hidden_flat,
            router_indices,
            routing_weights_dense,
            gate_up_proj,
            gate_up_proj_bias,
            down_proj,
            down_proj_bias,
            ctx.expert_capacity,
            ctx.num_experts,
            ctx.top_k,
        )

        # Return grad for dense routing; autograd handles scatter->softmax->linear
        grad_routing_weights_dense = (
            grad_routing_weights if ctx.enable_router_grads else None
        )

        # Return gradients per input (None for non-differentiable)
        return (
            grad_hidden_flat,  # hidden_flat
            None,  # router_indices
            grad_routing_weights_dense,  # routing_weights_dense
            grad_gate_up_proj,
            grad_gate_up_proj_bias,
            grad_down_proj,
            grad_down_proj_bias,
            None,  # expert_capacity
            None,  # num_experts
            None,  # top_k
            None,  # enable_router_grads
        )


def ceil_div(a, b):
    return (a + b - 1) // b

class Yamoe(torch.nn.Module):
    can_torch_compile: bool = False

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Default to the inference path, but honor a caller that opts into the
        # autograd-aware path (_ExpertsFn) by setting `enable_router_grads = True`
        # before calling forward. Previously this was hardcoded to False, which
        # silently disabled backward.
        self.enable_router_grads = getattr(self, "enable_router_grads", False)
        self._timing_stats = {}

        batch_size, seq_len, hidden_dim = hidden_states.shape
        batch_seq = batch_size * seq_len

        num_experts = getattr(self, "num_experts", 32)
        top_k = getattr(self, "top_k", 4)
        timing = getattr(self, "_timing_enabled", False)

        if timing:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Route tokens to experts
        x_flat = hidden_states.view(-1, hidden_dim)
        logits = torch.nn.functional.linear(
            x_flat, self.router.weight, self.router.bias
        )

        if timing:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self._timing_stats["router"] = (t1 - t0) * 1000

        # Compute top-k
        if top_k == 1:
            routing_logits_topk, router_indices = logits.max(dim=-1, keepdim=True)
        else:
            routing_logits_topk, router_indices = torch.topk(logits, top_k, dim=-1)

        # Match reference path exactly: use F.softmax with explicit dtype
        routing_weights_topk = torch.nn.functional.softmax(
            routing_logits_topk, dim=-1, dtype=routing_logits_topk.dtype
        )

        # Create router scores
        router_scores = torch.zeros_like(logits).scatter_(
            1, router_indices, routing_weights_topk
        )

        if timing:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            self._timing_stats["topk_softmax"] = (t2 - t1) * 1000

        dense_routing = router_scores  # [B*S, E]

        # Kernel expects shapes: [E, H, 2H] and [E, H, H]
        gate_up = self.experts.gate_up_proj[:, :, : 2 * hidden_dim].contiguous()
        gate_up_bias = self.experts.gate_up_proj_bias[:, : 2 * hidden_dim].contiguous()
        down_proj = self.experts.down_proj[:, :hidden_dim, :].contiguous()

        # Per-expert token capacity. The kernel processes at most `expert_capacity`
        # tokens per expert and silently drops the rest, so a value that is too
        # small degrades quality on imbalanced routing. `capacity_factor`
        # (GShard/Switch convention) scales the mean load: 1.0 = exactly the mean
        # (drops on any imbalance), higher = more headroom at the cost of memory
        # and compute, which both grow linearly with capacity. Default 2.0 is the
        # usual inference setting; raise it toward `batch_seq` for drop-free, lower
        # it to save memory. An expert can never receive more than `batch_seq`
        # tokens, so we clamp there (a large factor is drop-free without waste).
        capacity_factor = float(getattr(self, "capacity_factor", 2.0))
        expert_capacity = math.ceil(capacity_factor * batch_seq * top_k / num_experts)
        expert_capacity = max(1, min(expert_capacity, batch_seq))

        # Optionally surface how many token slots were dropped this call (opt-in:
        # the bincount + .item() sync is skipped unless requested).
        if getattr(self, "report_capacity_drops", False):
            counts = torch.bincount(router_indices.reshape(-1), minlength=num_experts)
            self._last_capacity_dropped = int(
                (counts - expert_capacity).clamp(min=0).sum().item()
            )

        if timing:
            torch.cuda.synchronize()
            t3 = time.perf_counter()

        # Compute expert output with custom backward
        if self.enable_router_grads:
            output = _ExpertsFn.apply(
                hidden_states.view(-1, hidden_dim),
                router_indices,
                dense_routing,
                gate_up,
                gate_up_bias,
                down_proj,
                self.experts.down_proj_bias,
                expert_capacity,
                num_experts,
                top_k,
                self.enable_router_grads,
            )
        else:
            with torch.no_grad():
                routing_weights_flat = dense_routing.view(-1, num_experts)

                # Pick the expert kernel. By default we auto-dispatch on the token
                # count, because the three kernels win in different regimes:
                #   * experts_gather (fused gather-GEMM): reads only the *active*
                #     experts' weights via device pointer arrays (no sort, no host
                #     sync, capturable). Wins in the decode / tiny-batch regime,
                #     where few experts are active — empirically while
                #     batch_seq*top_k < num_experts. For larger token counts it
                #     re-reads shared experts and loses.
                #   * experts_static (fixed-capacity batched GEMM): CUDA-graph
                #     capturable; used for prefill when `capture_safe` is requested.
                #   * experts (grouped GEMM): the default for prefill / large batch.
                # Explicit attributes override the auto choice:
                #   decode_gather = True/False  -> force the gather path on/off
                #   capture_safe  = True        -> use the capturable static path
                # The threshold is tunable via `gather_max_pairs` (default E).
                gather_threshold = getattr(self, "gather_max_pairs", num_experts)
                decode_gather = getattr(self, "decode_gather", None)
                if decode_gather is None:
                    decode_gather = batch_seq * top_k < gather_threshold

                # `set_device` is a host call that is fine eagerly but is skipped on
                # the capturable paths (the device context is already set).
                if decode_gather:
                    output = ops.experts_gather(
                        x_flat,
                        router_indices,
                        routing_weights_flat,
                        gate_up,
                        gate_up_bias,
                        down_proj,
                        self.experts.down_proj_bias,
                        num_experts,
                        top_k,
                    )
                elif getattr(self, "capture_safe", False):
                    output = ops.experts_static(
                        x_flat,
                        router_indices,
                        routing_weights_flat,
                        gate_up,
                        gate_up_bias,
                        down_proj,
                        self.experts.down_proj_bias,
                        expert_capacity,
                        num_experts,
                        top_k,
                    )
                else:
                    torch.cuda.set_device(hidden_states.device)
                    output = ops.experts(
                        x_flat,
                        router_indices,
                        routing_weights_flat,
                        gate_up,
                        gate_up_bias,
                        down_proj,
                        self.experts.down_proj_bias,
                        expert_capacity,
                        num_experts,
                        top_k,
                    )

        if timing:
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            self._timing_stats["experts_kernel"] = (t4 - t3) * 1000

        # Reshape output back to [B, S, H]
        output = output.view(batch_size, seq_len, hidden_dim)

        if timing:
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            self._timing_stats["total"] = (t5 - t0) * 1000
            print(f"\n[Yamoe.forward timing in ms]")
            print(f"  Router linear: {self._timing_stats['router']:.3f}")
            print(f"  TopK + Softmax: {self._timing_stats['topk_softmax']:.3f}")
            print(f"  Experts kernel: {self._timing_stats['experts_kernel']:.3f}")
            print(f"  Total: {self._timing_stats['total']:.3f}")

        return output, router_scores
