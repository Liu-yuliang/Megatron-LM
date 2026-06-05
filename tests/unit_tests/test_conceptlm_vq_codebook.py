# Copyright (c) 2026, ConceptLM contributors.

import torch
import torch.nn.functional as F

from megatron.core.models.gpt.conceptlm_v1 import SimVQProductQuantizer
from megatron.core.optimizer.emerging_optimizers import _is_nonlinear_or_embedding


def _legacy_quantizer_forward(quantizer, codebook, concept_hidden):
    concept_seq, batch_size, hidden_size = concept_hidden.shape
    inputs = concept_hidden.reshape(
        concept_seq, batch_size, quantizer.num_codebooks, quantizer.head_dim
    )
    distances = (
        inputs.float().square().sum(dim=-1, keepdim=True)
        - 2.0 * torch.einsum("cbhd,hnd->cbhn", inputs.float(), codebook.float())
        + codebook.float()
        .square()
        .sum(dim=-1)
        .view(1, 1, quantizer.num_codebooks, quantizer.codebook_size)
    )
    code_ids = torch.argmin(distances, dim=-1)

    gather_index = code_ids.permute(2, 0, 1).reshape(quantizer.num_codebooks, -1)
    quantized_heads = []
    for head_idx in range(quantizer.num_codebooks):
        quantized_heads.append(codebook[head_idx].index_select(0, gather_index[head_idx]))
    quantized = torch.stack(quantized_heads, dim=0)
    quantized = quantized.reshape(
        quantizer.num_codebooks, concept_seq, batch_size, quantizer.head_dim
    ).permute(1, 2, 0, 3)
    quantized = quantized.reshape(concept_seq, batch_size, hidden_size).to(concept_hidden.dtype)

    codebook_loss = F.mse_loss(quantized, concept_hidden.detach())
    commitment_loss = F.mse_loss(quantized.detach(), concept_hidden)
    vq_loss = codebook_loss + quantizer.commitment_cost * commitment_loss
    return quantized, code_ids, vq_loss


def test_sim_vq_codebook_is_stored_as_2d_parameters_for_muon():
    quantizer = SimVQProductQuantizer(hidden_size=12, num_codebooks=3, codebook_size=5)

    codebook_params = list(quantizer.codebook)
    assert len(codebook_params) == 3
    assert all(tuple(param.shape) == (5, 4) for param in codebook_params)
    assert all(not _is_nonlinear_or_embedding(param) for param in codebook_params)

    state_dict = quantizer.state_dict()
    assert "codebook" not in state_dict
    assert set(state_dict) == {"codebook.0", "codebook.1", "codebook.2"}
    assert tuple(quantizer.transformed_codebook().shape) == (3, 5, 4)


def test_sim_vq_codebook_loads_legacy_3d_state_dict_strictly():
    quantizer = SimVQProductQuantizer(hidden_size=12, num_codebooks=3, codebook_size=5)
    legacy_codebook = torch.randn(3, 5, 4)

    quantizer.load_state_dict({"codebook": legacy_codebook}, strict=True)

    torch.testing.assert_close(quantizer.transformed_codebook(), legacy_codebook)


def test_sim_vq_2d_codebook_forward_matches_legacy_3d_codebook_math():
    torch.manual_seed(1234)
    quantizer = SimVQProductQuantizer(
        hidden_size=12, num_codebooks=3, codebook_size=5, commitment_cost=0.25
    )
    legacy_codebook = torch.randn(3, 5, 4)
    concept_hidden = torch.randn(2, 4, 12)
    with torch.no_grad():
        for codebook_idx, codebook in enumerate(quantizer.codebook):
            codebook.copy_(legacy_codebook[codebook_idx])

    actual_quantized, actual_code_ids, actual_vq_loss = quantizer(concept_hidden)
    expected_quantized, expected_code_ids, expected_vq_loss = _legacy_quantizer_forward(
        quantizer, legacy_codebook, concept_hidden
    )

    torch.testing.assert_close(actual_quantized, expected_quantized)
    torch.testing.assert_close(actual_vq_loss, expected_vq_loss)
    assert torch.equal(actual_code_ids, expected_code_ids)
