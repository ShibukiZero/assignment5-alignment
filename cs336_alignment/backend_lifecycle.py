from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


@dataclass(frozen=True)
class TrainingBackendConfig:
    model_id: str
    device: str
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str = "flash_attention_2"


@dataclass(frozen=True)
class InferenceBackendConfig:
    model_id: str
    device: str
    seed: int
    gpu_memory_utilization: float
    dtype: torch.dtype = torch.float16
    enable_prefix_caching: bool = True


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

    @classmethod
    def from_defaults(
        cls,
        *,
        model_id: str,
        policy_device: str,
        vllm_device: str,
        seed: int,
        vllm_gpu_memory_utilization: float,
    ) -> "BackendLifecycleManager":
        return cls(
            training_config=TrainingBackendConfig(
                model_id=model_id,
                device=policy_device,
            ),
            inference_config=InferenceBackendConfig(
                model_id=model_id,
                device=vllm_device,
                seed=seed,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
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
        return self._policy, self._tokenizer

    def initialize_inference_backend(self) -> LLM:
        if self._llm is None:
            self._llm = init_vllm(
                model_id=self.inference_config.model_id,
                device=self.inference_config.device,
                seed=self.inference_config.seed,
                gpu_memory_utilization=self.inference_config.gpu_memory_utilization,
                dtype=self.inference_config.dtype,
                enable_prefix_caching=self.inference_config.enable_prefix_caching,
            )
        return self._llm

    def enter_training_phase(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        policy, tokenizer = self.initialize_training_backend()
        self._maybe_cleanup_after_phase_change()
        policy.train()
        return policy, tokenizer

    def enter_inference_phase(self, *, sync_weights: bool = True) -> LLM:
        llm = self.initialize_inference_backend()
        self._maybe_cleanup_after_phase_change()
        if sync_weights:
            policy, _ = self.initialize_training_backend()
            load_policy_into_vllm_instance(
                policy=policy,
                llm=llm,
                synchronize_devices=self.phase_config.synchronize_before_weight_sync,
            )
        return llm

    def training_model(self) -> PreTrainedModel:
        policy, _ = self.initialize_training_backend()
        return policy

    def tokenizer(self) -> PreTrainedTokenizerBase:
        _, tokenizer = self.initialize_training_backend()
        return tokenizer

    def inference_engine(self) -> LLM:
        return self.initialize_inference_backend()

    def _maybe_cleanup_after_phase_change(self) -> None:
        if not self.phase_config.clear_cuda_cache_on_phase_change:
            return
        for device in {self.training_config.device, self.inference_config.device}:
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
        torch.cuda.empty_cache()
