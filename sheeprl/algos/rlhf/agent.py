from pathlib import Path
from typing import Optional

from lightning import Fabric
from omegaconf import OmegaConf

from sheeprl.algos.rlhf.config_store.model import FINETUNE_MODE, ModelConfig
from sheeprl.algos.rlhf.lora_utils import add_lora, add_multiple_lora, disable_lora, enable_lora, select_lora
from sheeprl.algos.rlhf.models import ActorModel, CriticModel, RewardModel
from sheeprl.algos.rlhf.utils import get_last_checkpoint_path, setup_finetuning, trainable_parameter_summary


class PPOAgent:
    _reference: ActorModel
    _reward: RewardModel
    _finetune_mode: FINETUNE_MODE
    _actor: Optional[ActorModel] = None
    _critic: Optional[CriticModel] = None
    _share_actor_critic: bool = False
    _share_critic_reward: bool = False

    def __init__(
        self,
        fabric: Fabric,
        model_cfg: ModelConfig,
        init_critic_with_rm: bool,
        sft_experiment_dir: str,
        rm_experiment_dir: str,
    ) -> None:
        self.model_cfg = model_cfg
        self.sft_experiment_dir = sft_experiment_dir
        self.rm_experiment_dir = rm_experiment_dir
        self.fabric = fabric

        sft_exp_cfg = OmegaConf.load(Path(sft_experiment_dir) / ".hydra/config.yaml")
        sft_ckpt_model_cfg = ModelConfig(**sft_exp_cfg.model)
        sft_checkpoint_path = get_last_checkpoint_path(sft_experiment_dir)
        sft_model_name = sft_ckpt_model_cfg.name

        rm_exp_cfg = OmegaConf.load(Path(rm_experiment_dir) / ".hydra/config.yaml")
        rm_ckpt_model_cfg = ModelConfig(**rm_exp_cfg.model)
        rm_checkpoint_path = get_last_checkpoint_path(rm_experiment_dir)
        rm_model_name = rm_ckpt_model_cfg.name

        fabric.print("Loading reference model")
        self._reference = ActorModel.from_checkpoint(
            device=fabric.device,
            model_cfg=sft_ckpt_model_cfg,
            path=sft_checkpoint_path,
            freeze=True,
        )
        # Reward model
        fabric.print("Loading reward model")
        self._reward = RewardModel.from_checkpoint(
            device=fabric.device,
            model_cfg=rm_ckpt_model_cfg,
            path=rm_checkpoint_path,
            freeze=True,
        )

        lora_enabled = model_cfg.finetune_mode == FINETUNE_MODE.LORA
        lora_cfg = model_cfg.lora_cfg
        if not init_critic_with_rm:
            same_actor_critic = sft_model_name == rm_model_name
            if lora_enabled and same_actor_critic:
                # here we can share reference model between Actor and Critic
                self._share_actor_critic = True
                fabric.print("Adding LORA parameters to both actor and critic")
                add_multiple_lora(self._reference, lora_cfg=lora_cfg, device=fabric.device, num=2)
                trainable_parameter_summary(self._reference, show_names=False, fabric=fabric)
                self._reference = fabric.setup_module(self._reference)
            else:
                # Actor and critic cannot be shared, we fallback to the default behavior
                fabric.print("Loading actor model")
                actor_model = ActorModel.from_checkpoint(
                    device=fabric.device, model_cfg=sft_ckpt_model_cfg, path=sft_checkpoint_path, freeze=True
                )
                setup_finetuning(fabric, actor_model, model_cfg)
                trainable_parameter_summary(model=actor_model, show_names=False, fabric=fabric)
                self._actor = fabric.setup_module(actor_model)

                fabric.print("Loading critic model from sft model")
                critic_model = CriticModel.from_checkpoint(
                    device=fabric.device, model_cfg=sft_ckpt_model_cfg, path=sft_ckpt_model_cfg, freeze=True
                )
                setup_finetuning(fabric, critic_model, model_cfg)
                trainable_parameter_summary(model=critic_model, show_names=False, fabric=fabric)
                self._critic = fabric.setup_module(critic_model)
        else:
            # here we have critic model initialized with reward model so we need separete actor model
            fabric.print("Loading actor model")
            actor_model = ActorModel.from_checkpoint(
                device=fabric.device, model_cfg=sft_ckpt_model_cfg, path=sft_checkpoint_path, freeze=True
            )
            setup_finetuning(fabric, actor_model, model_cfg)
            trainable_parameter_summary(model=actor_model, show_names=False, fabric=fabric)
            self._actor = fabric.setup_module(actor_model)
            if lora_enabled:
                self._share_critic_reward = True
                fabric.print("Adding LORA parameters to reward model for critic model")
                add_lora(self._reward, lora_cfg=lora_cfg, device=fabric.device)
                trainable_parameter_summary(self._reward, show_names=False, fabric=fabric)
                self._reward = fabric.setup_module(self._reward)
            else:
                fabric.print("Loading critic model from reward model")
                critic_model = CriticModel.from_checkpoint(
                    device=fabric.device, model_cfg=rm_ckpt_model_cfg, path=rm_checkpoint_path, freeze=True
                )
                setup_finetuning(fabric, critic_model, model_cfg)
                trainable_parameter_summary(model=critic_model, show_names=False, fabric=fabric)
                self._critic = fabric.setup_module(critic_model)

    @property
    def actor(self) -> ActorModel:
        if self._share_actor_critic:
            enable_lora(self._reference)
            return select_lora(self._reference, index=0)
        else:
            return self._actor

    @property
    def critic(self) -> CriticModel:
        if self._share_actor_critic:
            enable_lora(self._reference)
            return select_lora(self._reference, index=1)
        elif self._share_critic_reward:
            enable_lora(self._reward)
            self._reward.disable_bias_gain()
            return self._reward
        else:
            return self._critic

    @property
    def reference(self) -> ActorModel:
        if self._share_actor_critic:
            disable_lora(self._reference)

        return self._reference

    @property
    def reward(self) -> RewardModel:
        if self._share_critic_reward:
            disable_lora(self._reward)
            self._reward.enable_bias_gain()
        return self._reward


class DPOAgent:
    _reference: ActorModel
    _finetune_mode: FINETUNE_MODE
    _actor: Optional[ActorModel] = None

    def __init__(
        self,
        fabric: Fabric,
        model_cfg: ModelConfig,
        sft_experiment_dir: str,
    ) -> None:
        self.model_cfg = model_cfg
        self.sft_experiment_dir = sft_experiment_dir
        self.fabric = fabric
        # Currently we only support same architecture for reference and actor models
        sft_exp_cfg = OmegaConf.load(Path(sft_experiment_dir) / ".hydra/config.yaml")
        sft_ckpt_model_cfg = ModelConfig(**sft_exp_cfg.model)
        sft_checkpoint_path = get_last_checkpoint_path(sft_experiment_dir)

        fabric.print("Loading reference model")
        self._reference = ActorModel.from_checkpoint(
            device=fabric.device,
            model_cfg=sft_ckpt_model_cfg,
            path=sft_checkpoint_path,
            freeze=True,
        )
        self._finetune_mode = model_cfg.finetune_mode
        lora_enabled = model_cfg.finetune_mode == FINETUNE_MODE.LORA
        lora_cfg = model_cfg.lora_cfg
        if lora_enabled:
            fabric.print("Adding LORA parameters to reference model")
            add_lora(self._reference, lora_cfg=lora_cfg, device=fabric.device)
            trainable_parameter_summary(self._reference, show_names=False, fabric=fabric)
            self._reference = fabric.setup_module(self._reference)
        else:
            fabric.print("Loading actor model")
            actor_model = ActorModel.from_checkpoint(
                device=fabric.device, model_cfg=sft_ckpt_model_cfg, path=sft_checkpoint_path, freeze=True
            )
            setup_finetuning(fabric, actor_model, model_cfg)
            trainable_parameter_summary(model=actor_model, show_names=False, fabric=fabric)
            self._actor = fabric.setup_module(actor_model)

    @property
    def actor(self) -> ActorModel:
        if self._finetune_mode == FINETUNE_MODE.LORA:
            enable_lora(self._reference)
            return self._reference
        else:
            return self._actor

    @property
    def reference(self) -> ActorModel:
        if self._finetune_mode == FINETUNE_MODE.LORA:
            disable_lora(self._reference)
        return self._reference
