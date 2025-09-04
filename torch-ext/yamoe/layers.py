import torch
from ._ops import ops


class Yamoe(torch.nn.Module):
    """Yamoe MoE layer with routing and expert computation"""

    can_torch_compile: bool = True

    def __init__(self):
        super().__init__()
        # Pre-allocate buffers to avoid repeated allocations
        self._routing_weights_buffer = None
        self._batch_indices_buffer = None
        self._last_batch_seq = None
        self._last_num_experts = None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        batch_seq = batch_size * seq_len

        num_experts = getattr(self, "num_experts", 128)
        top_k = getattr(self, "top_k", 4)

        # Route tokens to experts
        x_flat = hidden_states.view(-1, hidden_dim)
        logits = torch.nn.functional.linear(
            x_flat, self.router.weight, self.router.bias
        )

        # Compute top-k
        if top_k == 1:
            routing_weights, router_indices = logits.max(dim=-1, keepdim=True)
        else:
            routing_weights, router_indices = torch.topk(logits, top_k, dim=-1)

        routing_weights = routing_weights.softmax(dim=-1)

        # Create router scores
        router_scores = (
            torch.zeros_like(logits)
            .scatter_(1, router_indices, routing_weights)
            .transpose(0, 1)
        )

        # Convert routing_weights to sparse format [batch_seq, num_experts]
        # Reuse buffer if possible to reduce allocations
        if (
            self._routing_weights_buffer is None
            or self._last_batch_seq != batch_seq
            or self._last_num_experts != num_experts
            or self._routing_weights_buffer.device != routing_weights.device
        ):
            self._routing_weights_buffer = torch.zeros(
                batch_seq,
                num_experts,
                device=routing_weights.device,
                dtype=routing_weights.dtype,
            )
            self._batch_indices_buffer = (
                torch.arange(batch_seq, device=routing_weights.device)
                .unsqueeze(1)
                .expand(-1, top_k)
            )
            self._last_batch_seq = batch_seq
            self._last_num_experts = num_experts
        else:
            self._routing_weights_buffer.zero_()

        # Fill sparse routing weights
        flat_indices = router_indices.view(batch_seq, top_k)
        flat_weights = routing_weights.view(batch_seq, top_k)
        self._routing_weights_buffer[self._batch_indices_buffer, flat_indices] = (
            flat_weights
        )

        # FIX: Use the correct expert projections
        gate_up = self.experts.gate_up_proj[:, :, : hidden_dim * top_k].contiguous()
        gate_up_bias = self.experts.gate_up_proj_bias[
            :, : hidden_dim * top_k
        ].contiguous()

        down_proj = self.experts.down_proj[:, :hidden_dim, :].contiguous()

        expert_capacity = batch_seq * top_k // num_experts * 2

        with torch.no_grad():
            # Compute expert output
            output = ops.experts(
                hidden_states.view(-1, hidden_dim),
                router_indices,
                self._routing_weights_buffer,
                gate_up,
                gate_up_bias,
                down_proj,
                self.experts.down_proj_bias,
                expert_capacity,
                num_experts,
                top_k,
            )

        # Reshape output back to [B, S, H]
        output = output.view(batch_size, seq_len, hidden_dim)
        return output, router_scores
