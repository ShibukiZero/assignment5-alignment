from __future__ import annotations

import gc
import logging
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed


logger = logging.getLogger(__name__)

WARMUP_NUM_PROMPTS = 64
WARMUP_RANDOM_CHARS = 64
WARMUP_RANDOM_SEED = 0
WARMUP_MAX_NEW_TOKENS = 1
WARMUP_N = 8
WARMUP_TEMPERATURE = 1.0
WARMUP_TOP_P = 1.0


@dataclass(frozen=True)
class TrainingBackendConfig:
    model_id: str
    device: str
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "flash_attention_2"
    keep_policy_resident_on_device: bool = True
    keep_optimizer_state_resident_on_device: bool = False


@dataclass(frozen=True)
class InferenceBackendConfig:
    model_id: str
    device: str
    seed: int
    gpu_memory_utilization: float
    dtype: torch.dtype = torch.float16
    enable_prefix_caching: bool = True
    enable_sleep_mode: bool = False
    sleep_level: int = 1
    reset_prefix_cache_after_weight_sync: bool = True


@dataclass(frozen=True)
class PhaseManagementConfig:
    clear_cuda_cache_on_phase_change: bool = False
    synchronize_before_weight_sync: bool = True


def init_policy(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    model.train()
    return model, tokenizer


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float,
    dtype: torch.dtype = torch.float16,
    enable_prefix_caching: bool = True,
    enable_sleep_mode: bool = False,
) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_sleep_mode=enable_sleep_mode,
        )


def sync_policy_cuda_devices(policy: PreTrainedModel) -> None:
    devices = {
        parameter.device
        for parameter in policy.parameters()
        if parameter.device.type == "cuda"
    }
    for device in devices:
        torch.cuda.synchronize(device)


def load_policy_into_vllm_instance(
    policy: PreTrainedModel,
    llm: LLM,
    *,
    synchronize_devices: bool = True,
) -> None:
    if synchronize_devices:
        sync_policy_cuda_devices(policy)
    state_dict = policy.state_dict()
    llm_engine = getattr(llm, "llm_engine", getattr(llm, "engine", None))
    if llm_engine is None:
        raise AttributeError("Could not find vLLM engine on the LLM object.")
    vllm_model = llm_engine.model_executor.driver_worker.model_runner.model
    vllm_model.load_weights(state_dict.items())


def offload_optimizer_state_to_cpu(optimizer: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to("cpu", non_blocking=True)


def load_optimizer_state_to_device(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device, non_blocking=True)


def offload_training_backend_to_cpu(
    policy: PreTrainedModel,
    optimizer: Any | None,
    *,
    offload_policy: bool,
    offload_optimizer_state: bool,
) -> None:
    policy.zero_grad(set_to_none=True)
    if offload_policy:
        policy.to("cpu")
    if optimizer is not None and offload_optimizer_state:
        offload_optimizer_state_to_cpu(optimizer)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_training_backend_to_device(
    policy: PreTrainedModel,
    device: str,
    optimizer: Any | None,
    *,
    load_policy: bool,
    offload_optimizer_state: bool,
) -> None:
    if load_policy:
        policy.to(device)
    if optimizer is not None and offload_optimizer_state:
        load_optimizer_state_to_device(optimizer, device)


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def make_random_warmup_payload(
    rng: random.Random,
    *,
    num_chars: int,
) -> str:
    alphabet = string.ascii_letters + string.digits + " +-*/=()[]{}"
    return "".join(rng.choice(alphabet) for _ in range(num_chars))


def build_synthetic_warmup_prompts(
    *,
    num_prompts: int,
    num_chars: int,
    seed: int,
) -> tuple[str, ...]:
    rng = random.Random(seed)
    prompts: list[str] = []
    for prompt_index in range(num_prompts):
        payload = make_random_warmup_payload(rng, num_chars=num_chars)
        prompts.append(
            (
                "A conversation between User and Assistant. "
                f"Warmup sample {prompt_index}. "
                f"User: {payload}\n"
                "Assistant: <think>"
            )
        )
    return tuple(prompts)


class BackendLifecycleManager:
    def __init__(
        self,
        training_config: TrainingBackendConfig,
        inference_config: InferenceBackendConfig,
        phase_config: PhaseManagementConfig | None = None,
    ) -> None:
        self.training_config = training_config
        self.inference_config = inference_config
        self.phase_config = phase_config or PhaseManagementConfig()
        self._policy: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._llm: LLM | None = None
        self._optimizer: Any | None = None
        self._current_phase: str | None = None
        self._inference_backend_awake = False
        self._policy_offloaded = False
        self._optimizer_state_offloaded = False
        self._fresh_vllm_warmup_completed = False

    @classmethod
    def from_defaults(
        cls,
        *,
        model_id: str,
        policy_device: str,
        vllm_device: str,
        seed: int,
        vllm_gpu_memory_utilization: float,
        enable_sleep_mode: bool = False,
        sleep_level: int = 1,
        reset_prefix_cache_after_weight_sync: bool = True,
        keep_policy_resident_on_device: bool = True,
        keep_optimizer_state_resident_on_device: bool = False,
    ) -> "BackendLifecycleManager":
        return cls(
            training_config=TrainingBackendConfig(
                model_id=model_id,
                device=policy_device,
                keep_policy_resident_on_device=keep_policy_resident_on_device,
                keep_optimizer_state_resident_on_device=keep_optimizer_state_resident_on_device,
            ),
            inference_config=InferenceBackendConfig(
                model_id=model_id,
                device=vllm_device,
                seed=seed,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                enable_sleep_mode=enable_sleep_mode,
                sleep_level=sleep_level,
                reset_prefix_cache_after_weight_sync=reset_prefix_cache_after_weight_sync,
            ),
        )

    def initialize_training_backend(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._policy is None or self._tokenizer is None:
            self._policy, self._tokenizer = init_policy(
                model_id=self.training_config.model_id,
                device=self.training_config.device,
                torch_dtype=self.training_config.torch_dtype,
                attn_implementation=self.training_config.attn_implementation,
            )
            self._policy_offloaded = False
            self._optimizer_state_offloaded = False
        return self._policy, self._tokenizer

    def initialize_rl_runtime(
        self,
        *,
        optimizer_factory: Any | None = None,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, Any | None]:
        self._run_synthetic_vllm_warmup_if_needed()
        llm = self.initialize_inference_backend()
        if self.inference_config.enable_sleep_mode and self._inference_backend_awake:
            llm.sleep(level=self.inference_config.sleep_level)
            self._inference_backend_awake = False

        policy, tokenizer = self.initialize_training_backend()
        if optimizer_factory is not None and self._optimizer is None:
            self._optimizer = optimizer_factory(policy.parameters())

        self._apply_inactive_training_residency()
        self._current_phase = None
        return policy, tokenizer, self._optimizer

    def initialize_inference_backend(self) -> LLM:
        if self._llm is None:
            self._llm = init_vllm(
                model_id=self.inference_config.model_id,
                device=self.inference_config.device,
                seed=self.inference_config.seed,
                gpu_memory_utilization=self.inference_config.gpu_memory_utilization,
                dtype=self.inference_config.dtype,
                enable_prefix_caching=self.inference_config.enable_prefix_caching,
                enable_sleep_mode=self.inference_config.enable_sleep_mode,
            )
            self._inference_backend_awake = True
            if self._current_phase == "training":
                self._sleep_inference_backend_if_needed()
        return self._llm

    def enter_training_phase(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        policy, tokenizer = self.initialize_training_backend()
        self._sleep_inference_backend_if_needed()
        self._load_training_backend_if_needed()
        self._maybe_cleanup_after_phase_change()
        policy.train()
        self._current_phase = "training"
        return policy, tokenizer

    def enter_inference_phase(self, *, sync_weights: bool = True) -> LLM:
        if sync_weights:
            policy, _ = self.initialize_training_backend()
        else:
            policy = None

        self._apply_inactive_training_residency()
        llm = self.initialize_inference_backend()
        self._wake_inference_backend_if_needed()
        self._maybe_cleanup_after_phase_change()
        if sync_weights:
            load_policy_into_vllm_instance(
                policy=policy,
                llm=llm,
                synchronize_devices=self.phase_config.synchronize_before_weight_sync,
            )
            self._reset_inference_prefix_cache_if_needed()
        self._current_phase = "inference"
        return llm

    def prepare_for_training(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        return self.enter_training_phase()

    def prepare_for_inference(self, *, sync_weights: bool = True) -> LLM:
        return self.enter_inference_phase(sync_weights=sync_weights)

    def after_inference(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        return self.enter_training_phase()

    def training_model(self) -> PreTrainedModel:
        policy, _ = self.enter_training_phase()
        return policy

    def tokenizer(self) -> PreTrainedTokenizerBase:
        _, tokenizer = self.enter_training_phase()
        return tokenizer

    def inference_engine(self) -> LLM:
        return self.initialize_inference_backend()

    def attach_training_optimizer(self, optimizer: Any) -> None:
        self._optimizer = optimizer
        if self._current_phase != "training":
            self._apply_inactive_training_residency()

    def training_optimizer(self) -> Any | None:
        return self._optimizer

    def debug_state(self) -> dict[str, Any]:
        return {
            "current_phase": self._current_phase,
            "inference_backend_awake": self._inference_backend_awake,
            "training_backend_offloaded": self._policy_offloaded or self._optimizer_state_offloaded,
            "policy_offloaded": self._policy_offloaded,
            "optimizer_state_offloaded": self._optimizer_state_offloaded,
            "has_optimizer": self._optimizer is not None,
        }

    def _sleep_inference_backend_if_needed(self) -> None:
        if (
            self._llm is None
            or not self.inference_config.enable_sleep_mode
            or not self._inference_backend_awake
        ):
            return
        self._llm.sleep(level=self.inference_config.sleep_level)
        self._inference_backend_awake = False

    def _apply_inactive_training_residency(self) -> None:
        if self._policy is None:
            return

        offload_policy = (
            not self.training_config.keep_policy_resident_on_device and not self._policy_offloaded
        )
        offload_optimizer_state = (
            self._optimizer is not None
            and not self.training_config.keep_optimizer_state_resident_on_device
            and not self._optimizer_state_offloaded
        )
        if not offload_policy and not offload_optimizer_state:
            self._policy.zero_grad(set_to_none=True)
            return

        offload_training_backend_to_cpu(
            self._policy,
            self._optimizer,
            offload_policy=offload_policy,
            offload_optimizer_state=offload_optimizer_state,
        )
        self._policy_offloaded = self._policy_offloaded or offload_policy
        self._optimizer_state_offloaded = self._optimizer_state_offloaded or offload_optimizer_state

    def _load_training_backend_if_needed(self) -> None:
        if self._policy is None or (not self._policy_offloaded and not self._optimizer_state_offloaded):
            return
        load_training_backend_to_device(
            self._policy,
            self.training_config.device,
            self._optimizer,
            load_policy=self._policy_offloaded,
            offload_optimizer_state=self._optimizer_state_offloaded,
        )
        self._policy_offloaded = False
        self._optimizer_state_offloaded = False

    def _wake_inference_backend_if_needed(self) -> None:
        if (
            self._llm is None
            or not self.inference_config.enable_sleep_mode
            or self._inference_backend_awake
        ):
            return
        self._llm.wake_up()
        self._inference_backend_awake = True

    def _reset_inference_prefix_cache_if_needed(self) -> None:
        if (
            self._llm is None
            or not self.inference_config.reset_prefix_cache_after_weight_sync
        ):
            return
        self._llm.reset_prefix_cache()

    def _maybe_cleanup_after_phase_change(self) -> None:
        if not self.phase_config.clear_cuda_cache_on_phase_change:
            return
        for device in {self.training_config.device, self.inference_config.device}:
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
        torch.cuda.empty_cache()

    def _run_synthetic_vllm_warmup_if_needed(self) -> None:
        if self._fresh_vllm_warmup_completed or not env_flag("CS336_FRESH_VLLM_WARMUP"):
            return

        prompts = build_synthetic_warmup_prompts(
            num_prompts=WARMUP_NUM_PROMPTS,
            num_chars=WARMUP_RANDOM_CHARS,
            seed=WARMUP_RANDOM_SEED,
        )
        sampling_params = SamplingParams(
            temperature=WARMUP_TEMPERATURE,
            top_p=WARMUP_TOP_P,
            max_tokens=WARMUP_MAX_NEW_TOKENS,
            n=WARMUP_N,
            include_stop_str_in_output=False,
        )

        logger.info(
            "Starting fresh vLLM warmup over %d prompts",
            len(prompts),
        )
        started_at = time.time()
        llm = init_vllm(
            model_id=self.inference_config.model_id,
            device=self.inference_config.device,
            seed=self.inference_config.seed,
            gpu_memory_utilization=self.inference_config.gpu_memory_utilization,
            dtype=self.inference_config.dtype,
            enable_prefix_caching=self.inference_config.enable_prefix_caching,
            enable_sleep_mode=False,
        )
        outputs = llm.generate(list(prompts), sampling_params)
        elapsed = time.time() - started_at

        del outputs
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._fresh_vllm_warmup_completed = True
        logger.info(
            "Completed fresh vLLM warmup in %.2f seconds",
            elapsed,
        )
