# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script for MISA."""

import absl.app
import absl.flags
import gym
import numpy as np
import tqdm

from algos.misa import MISA
from algos.model import (
  FullyConnectedQFunction,
  SamplerPolicy,
  TanhGaussianPolicy,
)
from data import Dataset, RandSampler
from utilities.jax_utils import batch_to_jax
from utilities.replay_buffer import get_d4rl_dataset
from utilities.sampler import TrajSampler
from utilities.utils import (
  Timer,
  WandBLogger,
  define_flags_with_default,
  get_user_flags,
  norm_obs,
  prefix_metrics,
  set_random_seed,
)
from viskit.logging import logger, setup_logger

FLAGS_DEF = define_flags_with_default(
  env="walker2d-medium-v2",
  max_traj_length=1000,
  save_model=False,
  seed=42,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch="64-64",
  policy_arch="256-256",
  qf_arch="256-256",
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  algo_cfg=MISA.get_default_config(),
  n_epochs=1200,
  bc_epochs=0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=10,
  # configs for trining scheme
  logging=WandBLogger.get_default_config(),
  use_layer_norm=False,
  activation="elu",
  obs_norm=False,
)


def main(argv):
  FLAGS = absl.flags.FLAGS
  algo_cfg = FLAGS.algo_cfg

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  for k, v in algo_cfg.items():
    variant[f"algo.{k}"] = v

  logging_configs = FLAGS.logging

  is_adroit = any(
    [w in FLAGS.env for w in ["pen", "hammer", "door", "relocate"]]
  )
  is_kitchen = "kitchen" in FLAGS.env
  is_mujoco = any([w in FLAGS.env for w in ["hopper", "walker", "cheetah"]])
  is_antmaze = "antmaze" in FLAGS.env

  env_log_name = ""
  if is_adroit:
    env_log_name = "Adroit"
  elif is_kitchen:
    env_log_name = "Kitchen"
  elif is_mujoco:
    env_log_name = "Mujoco"
  elif is_antmaze:
    env_log_name = "Antmaze"
  else:
    raise NotImplementedError

  logging_configs["project"] = f"MISA-{env_log_name}"
  wandb_logger = WandBLogger(
    config=logging_configs, variant=variant, env_name=FLAGS.env
  )

  setup_logger(
    variant=variant,
    exp_id=wandb_logger.experiment_id,
    seed=FLAGS.seed,
    base_log_dir=FLAGS.logging.output_dir,
    include_exp_prefix_sub_dir=False,
  )

  set_random_seed(FLAGS.seed)
  obs_mean = 0
  obs_std = 1
  obs_clip = np.inf

  eval_sampler = TrajSampler(gym.make(FLAGS.env), FLAGS.max_traj_length)
  dataset = get_d4rl_dataset(
    eval_sampler.env,
    algo_cfg.nstep,
    algo_cfg.discount,
  )

  dataset["rewards"
         ] = dataset["rewards"] * FLAGS.reward_scale + FLAGS.reward_bias
  dataset["actions"] = np.clip(
    dataset["actions"], -FLAGS.clip_action, FLAGS.clip_action
  )

  if is_kitchen or is_adroit or is_antmaze:
    if FLAGS.obs_norm:
      obs_mean = dataset["observations"].mean()
      obs_std = dataset["observations"].std()
      obs_clip = 10
      norm_obs(dataset, obs_mean, obs_std, obs_clip)

    if is_antmaze:
      dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
    else:
      min_r, max_r = np.min(dataset["rewards"]), np.max(dataset["rewards"])
      dataset["rewards"] = (dataset["rewards"] - min_r) / (max_r - min_r)
      dataset["rewards"] = (dataset["rewards"] - 0.5) * 2

  dataset = Dataset(dataset)
  sampler = RandSampler(dataset.size(), FLAGS.batch_size)
  dataset.set_sampler(sampler)

  observation_dim = eval_sampler.env.observation_space.shape[0]
  action_dim = eval_sampler.env.action_space.shape[0]
  policy = TanhGaussianPolicy(
    observation_dim,
    action_dim,
    FLAGS.policy_arch,
    FLAGS.orthogonal_init,
    FLAGS.policy_log_std_multiplier,
    FLAGS.policy_log_std_offset,
    use_layer_norm=FLAGS.use_layer_norm,
  )
  qf = FullyConnectedQFunction(
    observation_dim,
    action_dim,
    FLAGS.qf_arch,
    FLAGS.orthogonal_init,
    FLAGS.use_layer_norm,
    FLAGS.activation,
  )

  if algo_cfg.target_entropy >= 0.0:
    action_space = eval_sampler.env.action_space
    algo_cfg.target_entropy = -np.prod(action_space.shape).item()

  agent = MISA(algo_cfg, policy, qf)
  sampler_policy = SamplerPolicy(agent.policy, agent.train_params["policy"])

  viskit_metrics = {}
  for epoch in range(FLAGS.n_epochs):
    metrics = {"epoch": epoch}

    with Timer() as train_timer:
      for _ in tqdm.tqdm(range(FLAGS.n_train_step_per_epoch)):
        batch = batch_to_jax(dataset.sample())
        metrics.update(
          prefix_metrics(
            agent.train(batch, bc=epoch < FLAGS.bc_epochs), "agent"
          )
        )

    with Timer() as eval_timer:
      if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
        trajs = eval_sampler.sample(
          sampler_policy.update_params(agent.train_params["policy"]),
          FLAGS.eval_n_trajs,
          deterministic=True,
          obs_statistics=(obs_mean, obs_std, obs_clip),
        )

        metrics["average_return"] = np.mean(
          [np.sum(t["rewards"]) for t in trajs]
        )
        metrics["average_traj_length"] = np.mean(
          [len(t["rewards"]) for t in trajs]
        )
        metrics["average_normalizd_return"] = np.mean(
          [
            eval_sampler.env.get_normalized_score(np.sum(t["rewards"]))
            for t in trajs
          ]
        )
        metrics["done"] = np.mean([np.sum(t["dones"]) for t in trajs])

        if FLAGS.save_model:
          save_data = {"agent": agent, "variant": variant, "epoch": epoch}
          wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

    metrics["train_time"] = train_timer()
    metrics["eval_time"] = eval_timer()
    metrics["epoch_time"] = train_timer() + eval_timer()
    wandb_logger.log(metrics)
    viskit_metrics.update(metrics)
    logger.record_dict(viskit_metrics)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

  if FLAGS.save_model:
    save_data = {"agent": agent, "variant": variant, "epoch": epoch}
    wandb_logger.save_pickle(save_data, "model_final.pkl")


if __name__ == "__main__":
  absl.app.run(main)
