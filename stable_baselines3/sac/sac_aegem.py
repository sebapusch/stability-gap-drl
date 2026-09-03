from collections.abc import Callable, Iterable

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from stable_baselines3.common.buffers import ExpertBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import GymEnv, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.sac_bc import _gaussian_kl, _l2
from stable_baselines3.sac.sac_fine_tune import SAC_FineTune

ActorDistLoss = Callable[[th.Tensor, th.Tensor, th.Tensor, th.Tensor], th.Tensor]


class SAC_AEGEM(SAC_FineTune):
    """SAC with actor-only A-GEM transformation from expert-buffer gradients."""

    def __init__(
        self,
        expert_buffer_size: int,
        n_tasks: int,
        expert_buffer_batch_size: int,
        loss_fn: str = "kl",
        **kwargs,
    ):
        super().__init__(**kwargs)

        action_dim = self.action_space.shape[0]
        self.expert_buffer = ExpertBuffer(
            buffer_size=expert_buffer_size,
            n_tasks=n_tasks,
            observation_space=self.observation_space,
            output_size=2 * action_dim,
            device=self.device,
        )
        self.expert_buffer_batch_size = expert_buffer_batch_size
        self.loss_fn: ActorDistLoss = _l2 if loss_fn == "l2" else _gaussian_kl
        self.task_ix: int = 0

    def on_task_change(
        self,
        task_ix: int,
        env: GymEnv,
        logger: Logger,
    ) -> None:
        self.task_ix = task_ix

        if task_ix > 0:
            assert self.replay_buffer is not None

            def get_dist(obs: th.Tensor) -> th.Tensor:
                mu, log_std, _ = self.actor.get_action_dist_params(obs)
                return th.cat([mu, log_std], dim=1)

            self.expert_buffer.populate(get_dist, self.replay_buffer)

        super().on_task_change(task_ix, env, logger)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.task_ix == 0:
            return super().train(gradient_steps, batch_size)

        self.policy.set_training_mode(True)

        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        expert_losses: list[float] = []
        actor_projection_rates: list[float] = []

        actor_params = list(self.actor.parameters())

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )

            ent_coef, ent_coef_loss = self._update_ent_coef(replay_data)
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None:
                ent_coef_losses.append(ent_coef_loss.item())

            critic_loss = self._critic_loss(replay_data, ent_coef)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            actor_loss, new_actor_grads = self._actor_loss_and_gradients(
                replay_data,
                ent_coef,
                actor_params,
            )
            actor_losses.append(actor_loss.item())

            expert_loss = self._expert_actor_loss()
            expert_losses.append(expert_loss.item())
            old_actor_grads = self._gradients_for_loss(
                expert_loss,
                actor_params,
                self.actor.optimizer,
            )

            actor_projection_rates.append(
                self._project_gradients(
                    old_actor_grads,
                    new_actor_grads,
                    actor_params,
                )
            )
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/expert_actor_loss", np.mean(expert_losses))
        self.logger.record("train/num_projected_actor", np.mean(actor_projection_rates))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _critic_loss(self, replay_data: ReplayBufferSamples, ent_coef: th.Tensor) -> th.Tensor:
        discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

        with th.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

        current_q_values = self.critic(replay_data.observations, replay_data.actions)
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_loss += self.get_critic_auxiliary_loss()
        assert isinstance(critic_loss, th.Tensor)
        return critic_loss

    def _actor_loss(self, replay_data: ReplayBufferSamples, ent_coef: th.Tensor) -> th.Tensor:
        actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        assert isinstance(actor_loss, th.Tensor)
        return actor_loss

    def _expert_actor_loss(self) -> th.Tensor:
        action_dim = self.action_space.shape[0]
        expert_samples = self.expert_buffer.sample(self.expert_buffer_batch_size)

        mu, log_std, _ = self.actor.get_action_dist_params(expert_samples.observations)
        expert_mu = expert_samples.outputs[:, :action_dim]
        expert_log_std = expert_samples.outputs[:, action_dim:]

        return self.loss_fn(mu, log_std, expert_mu, expert_log_std)

    def _gradients_for_loss(
        self,
        loss: th.Tensor,
        params: Iterable[Parameter],
        optimizer: th.optim.Optimizer,
    ) -> list[th.Tensor]:
        param_list = list(params)
        optimizer.zero_grad()
        loss.backward()
        return [
            param.grad.detach().clone() if param.grad is not None else th.zeros_like(param)
            for param in param_list
        ]

    def _actor_loss_and_gradients(
        self,
        replay_data: ReplayBufferSamples,
        ent_coef: th.Tensor,
        params: Iterable[Parameter],
    ) -> tuple[th.Tensor, list[th.Tensor]]:
        critic_requires_grad = [param.requires_grad for param in self.critic.parameters()]
        for param in self.critic.parameters():
            param.requires_grad_(False)
        try:
            actor_loss = self._actor_loss(replay_data, ent_coef)
            gradients = self._gradients_for_loss(actor_loss, params, self.actor.optimizer)
            return actor_loss, gradients
        finally:
            for param, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
                param.requires_grad_(requires_grad)

    def _project_gradients(
        self,
        old_grads: list[th.Tensor],
        new_grads: list[th.Tensor],
        params: Iterable[Parameter],
    ) -> float:
        total = 0
        projected = 0

        for old_grad, new_grad, param in zip(old_grads, new_grads, params):
            dot = th.sum(new_grad * old_grad)
            old_norm_sq = th.sum(old_grad * old_grad)
            projected_grad = new_grad

            if dot.item() < 0 and old_norm_sq.item() > 0:
                projected_grad = new_grad - dot / old_norm_sq * old_grad
                projected += 1

            param.grad = projected_grad.detach().clone()
            total += 1

        return projected / total if total > 0 else 0.0

    def _update_ent_coef(
        self,
        replay_data: ReplayBufferSamples,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        _, log_prob = self.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            ent_coef = th.exp(self.log_ent_coef.detach())
            assert isinstance(self.target_entropy, float)
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
        else:
            ent_coef = self.ent_coef_tensor

        return ent_coef, ent_coef_loss
