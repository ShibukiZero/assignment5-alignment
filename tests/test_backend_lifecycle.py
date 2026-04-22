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
    enable_sleep_mode: bool = False,
    sleep_level: int = 1,
    reset_prefix_cache_after_weight_sync: bool = True,
    keep_policy_resident_on_device: bool = True,
    keep_optimizer_state_resident_on_device: bool = True,
) -> BackendLifecycleManager:
    return BackendLifecycleManager(
        training_config=TrainingBackendConfig(
            model_id="dummy-model",
            device="cuda:0",
            keep_policy_resident_on_device=keep_policy_resident_on_device,
            keep_optimizer_state_resident_on_device=keep_optimizer_state_resident_on_device,
        ),
        inference_config=InferenceBackendConfig(
            model_id="dummy-model",
            device="cuda:1",
            seed=0,
            gpu_memory_utilization=0.85,
            enable_sleep_mode=enable_sleep_mode,
            sleep_level=sleep_level,
            reset_prefix_cache_after_weight_sync=reset_prefix_cache_after_weight_sync,
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
    llm = Mock()
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
    llm = Mock()
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


def test_init_vllm_receives_sleep_mode(monkeypatch):
    init_vllm = Mock()
    llm = object()
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)

    manager = make_manager(enable_sleep_mode=True, sleep_level=2)

    result = manager.initialize_inference_backend()

    assert result is llm
    init_vllm.assert_called_once()
    assert init_vllm.call_args.kwargs["enable_sleep_mode"] is True


def test_enter_inference_phase_wakes_sleeping_engine_and_resets_prefix_cache(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager(enable_sleep_mode=True)

    manager.enter_training_phase()
    manager.initialize_inference_backend()
    manager.enter_inference_phase(sync_weights=True)

    llm.wake_up.assert_called_once()
    llm.reset_prefix_cache.assert_called_once()
    sync_weights.assert_called_once()


def test_enter_training_phase_sleeps_engine_after_inference(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager(enable_sleep_mode=True, sleep_level=2)

    manager.enter_training_phase()
    manager.initialize_inference_backend()
    manager.enter_inference_phase(sync_weights=True)
    manager.enter_training_phase()

    assert llm.sleep.call_count == 2
    assert llm.sleep.call_args_list[0].kwargs == {"level": 2}
    assert llm.sleep.call_args_list[1].kwargs == {"level": 2}
    assert policy.train.call_count == 2


def test_enter_inference_phase_can_skip_prefix_cache_reset(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager(
        enable_sleep_mode=True,
        reset_prefix_cache_after_weight_sync=False,
    )

    manager.enter_training_phase()
    manager.initialize_inference_backend()
    manager.enter_inference_phase(sync_weights=True)

    llm.reset_prefix_cache.assert_not_called()


def test_initialize_inference_backend_sleeps_immediately_during_training(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)

    manager = make_manager(enable_sleep_mode=True, sleep_level=1)

    manager.enter_training_phase()
    result = manager.initialize_inference_backend()

    assert result is llm
    llm.sleep.assert_called_once_with(level=1)


def test_initialize_rl_runtime_initializes_sleeping_inference_before_training(monkeypatch):
    init_policy = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    call_order: list[str] = []

    def record_init_vllm(*args, **kwargs):
        call_order.append("init_vllm")
        return llm

    def record_init_policy(*args, **kwargs):
        call_order.append("init_policy")
        return policy, tokenizer

    init_policy.side_effect = record_init_policy
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", record_init_vllm)

    manager = make_manager(enable_sleep_mode=True, sleep_level=2)

    returned_policy, returned_tokenizer, optimizer = manager.initialize_rl_runtime()

    assert call_order == ["init_vllm", "init_policy"]
    assert returned_policy is policy
    assert returned_tokenizer is tokenizer
    assert optimizer is None
    llm.sleep.assert_called_once_with(level=2)
    state = manager.debug_state()
    assert state["current_phase"] is None
    assert state["inference_backend_awake"] is False


def test_initialize_rl_runtime_can_offload_policy_and_optimizer_after_init(monkeypatch):
    init_policy = Mock()
    offload_training = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    optimizer = Mock()
    init_policy.return_value = (policy, tokenizer)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", Mock(return_value=llm))
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        offload_training,
    )

    manager = make_manager(
        enable_sleep_mode=True,
        keep_policy_resident_on_device=False,
        keep_optimizer_state_resident_on_device=False,
    )

    manager.initialize_rl_runtime()
    manager.attach_training_optimizer(optimizer)

    assert offload_training.call_count == 2
    assert offload_training.call_args_list[0].args == (policy, None)
    assert offload_training.call_args_list[0].kwargs == {
        "offload_policy": True,
        "offload_optimizer_state": False,
    }
    assert offload_training.call_args_list[1].args == (policy, optimizer)
    assert offload_training.call_args_list[1].kwargs == {
        "offload_policy": False,
        "offload_optimizer_state": True,
    }


def test_attach_training_optimizer_can_apply_optimizer_only_residency_after_init(monkeypatch):
    init_policy = Mock()
    offload_training = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    optimizer = Mock()
    init_policy.return_value = (policy, tokenizer)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", Mock(return_value=llm))
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        offload_training,
    )

    manager = make_manager(
        enable_sleep_mode=True,
        keep_policy_resident_on_device=True,
        keep_optimizer_state_resident_on_device=False,
    )

    manager.initialize_rl_runtime()
    manager.attach_training_optimizer(optimizer)

    offload_training.assert_called_once_with(
        policy,
        optimizer,
        offload_policy=False,
        offload_optimizer_state=True,
    )


def test_enter_inference_phase_offloads_training_backend_when_enabled(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    offload_training = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    optimizer = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        offload_training,
    )

    manager = make_manager(
        keep_policy_resident_on_device=False,
        keep_optimizer_state_resident_on_device=False,
    )
    manager.attach_training_optimizer(optimizer)

    manager.enter_training_phase()
    manager.enter_inference_phase(sync_weights=True)

    offload_training.assert_called_once_with(
        policy,
        optimizer,
        offload_policy=True,
        offload_optimizer_state=True,
    )


def test_enter_inference_phase_offloads_before_initializing_inference_backend(monkeypatch):
    init_policy = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    optimizer = Mock()
    call_order: list[str] = []

    def record_offload(*args, **kwargs):
        call_order.append("offload")

    def record_init_vllm(*args, **kwargs):
        call_order.append("init_vllm")
        return llm

    init_policy.return_value = (policy, tokenizer)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", record_init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        record_offload,
    )

    manager = make_manager(
        keep_policy_resident_on_device=False,
        keep_optimizer_state_resident_on_device=False,
    )
    manager.attach_training_optimizer(optimizer)

    manager.enter_training_phase()
    manager.enter_inference_phase(sync_weights=True)

    assert call_order == ["offload", "init_vllm"]


def test_enter_training_phase_reloads_training_backend_when_offloaded(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    offload_training = Mock()
    load_training = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    optimizer = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        offload_training,
    )
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_training_backend_to_device",
        load_training,
    )

    manager = make_manager(
        keep_policy_resident_on_device=False,
        keep_optimizer_state_resident_on_device=False,
    )
    manager.attach_training_optimizer(optimizer)

    manager.enter_training_phase()
    manager.enter_inference_phase(sync_weights=True)
    manager.enter_training_phase()

    load_training.assert_called_once_with(
        policy,
        "cuda:0",
        optimizer,
        load_policy=True,
        offload_optimizer_state=True,
    )


def test_enter_inference_phase_offloads_only_policy_when_no_optimizer_attached(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    offload_training = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        offload_training,
    )

    manager = make_manager(
        keep_policy_resident_on_device=False,
        keep_optimizer_state_resident_on_device=False,
    )

    manager.enter_training_phase()
    manager.enter_inference_phase(sync_weights=True)

    offload_training.assert_called_once_with(
        policy,
        None,
        offload_policy=True,
        offload_optimizer_state=False,
    )


def test_enter_inference_phase_can_offload_optimizer_only(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    offload_training = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    optimizer = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_training_backend_to_cpu",
        offload_training,
    )

    manager = make_manager(
        keep_policy_resident_on_device=True,
        keep_optimizer_state_resident_on_device=False,
    )
    manager.attach_training_optimizer(optimizer)

    manager.enter_training_phase()
    manager.enter_inference_phase(sync_weights=True)

    offload_training.assert_called_once_with(
        policy,
        optimizer,
        offload_policy=False,
        offload_optimizer_state=True,
    )


def test_debug_state_reflects_phase_transitions(monkeypatch):
    init_policy = Mock()
    init_vllm = Mock()
    sync_weights = Mock()
    policy = Mock()
    tokenizer = object()
    llm = Mock()
    init_policy.return_value = (policy, tokenizer)
    init_vllm.return_value = llm
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_policy", init_policy)
    monkeypatch.setattr("cs336_alignment.backend_lifecycle.init_vllm", init_vllm)
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.load_policy_into_vllm_instance",
        sync_weights,
    )

    manager = make_manager(
        enable_sleep_mode=True,
        keep_policy_resident_on_device=False,
        keep_optimizer_state_resident_on_device=False,
    )

    state = manager.debug_state()
    assert state["current_phase"] is None
    assert state["inference_backend_awake"] is False
    assert state["training_backend_offloaded"] is False

    manager.enter_training_phase()
    manager.enter_inference_phase(sync_weights=True)

    state = manager.debug_state()
    assert state["current_phase"] == "inference"
    assert state["training_backend_offloaded"] is True


def test_offload_training_backend_to_cpu_clears_gradients_and_skips_optimizer_when_disabled(monkeypatch):
    from cs336_alignment.backend_lifecycle import offload_training_backend_to_cpu

    policy = Mock()
    optimizer = Mock()
    offload_optimizer_state = Mock()
    monkeypatch.setattr(
        "cs336_alignment.backend_lifecycle.offload_optimizer_state_to_cpu",
        offload_optimizer_state,
    )

    offload_training_backend_to_cpu(
        policy,
        optimizer,
        offload_policy=True,
        offload_optimizer_state=False,
    )

    policy.zero_grad.assert_called_once_with(set_to_none=True)
    policy.to.assert_called_once_with("cpu")
    offload_optimizer_state.assert_not_called()
