from __future__ import annotations

from unittest.mock import Mock

from cs336_alignment.backend_lifecycle import (
    BackendLifecycleManager,
    InferenceBackendConfig,
    PhaseManagementConfig,
    TrainingBackendConfig,
)


def make_manager(
    *,
    synchronize_before_weight_sync: bool = True,
) -> BackendLifecycleManager:
    return BackendLifecycleManager(
        training_config=TrainingBackendConfig(
            model_id="dummy-model",
            device="cuda:0",
        ),
        inference_config=InferenceBackendConfig(
            model_id="dummy-model",
            device="cuda:1",
            seed=0,
            gpu_memory_utilization=0.85,
        ),
        phase_config=PhaseManagementConfig(
            clear_cuda_cache_on_phase_change=False,
            synchronize_before_weight_sync=synchronize_before_weight_sync,
        ),
    )


def test_enter_training_phase_initializes_policy_once(monkeypatch):
    init_policy = Mock()
    policy = Mock()
    tokenizer = object()
    init_policy.return_value = (policy, tokenizer)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)

    manager = make_manager()

    first_policy, first_tokenizer = manager.enter_training_phase()
    second_policy, second_tokenizer = manager.enter_training_phase()

    init_policy.assert_called_once()
    assert first_policy is policy
    assert second_policy is policy
    assert first_tokenizer is tokenizer
    assert second_tokenizer is tokenizer
    assert policy.train.call_count == 2


def test_enter_inference_phase_initializes_once_and_syncs_each_time(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = object()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager()

    first_llm = manager.enter_inference_phase(sync_weights=True)
    second_llm = manager.enter_inference_phase(sync_weights=True)

    init_policy.assert_called_once()
    init_vllm.assert_called_once()
    assert first_llm is llm
    assert second_llm is llm
    assert sync_weights.call_count == 2
    for call in sync_weights.call_args_list:
        assert call.kwargs["policy"] is policy
        assert call.kwargs["llm"] is llm
        assert call.kwargs["synchronize_devices"] is True


def test_enter_inference_phase_can_skip_weight_sync(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = object()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager()

    result = manager.enter_inference_phase(sync_weights=False)

    assert result is llm
    init_policy.assert_not_called()
    init_vllm.assert_called_once()
    sync_weights.assert_not_called()


def test_enter_inference_phase_respects_phase_sync_setting(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = object()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager(synchronize_before_weight_sync=False)

    manager.enter_inference_phase(sync_weights=True)

    sync_weights.assert_called_once()
    assert sync_weights.call_args.kwargs["synchronize_devices"] is False
