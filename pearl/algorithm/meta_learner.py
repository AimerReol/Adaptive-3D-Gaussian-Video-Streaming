import datetime
import os
import time
import warnings
from collections import deque
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from meta_rl.pearl.algorithm.buffers import MultiTaskReplayBuffer
from meta_rl.pearl.algorithm.sac import SAC
from meta_rl.pearl.algorithm.sampler import Sampler


class MetaLearner:
    def __init__(
        self,
        env: HalfCheetahEnv,
        env_name: str,
        agent: SAC,
        observ_dim: int,
        action_dim: int,
        train_tasks: List[int],
        test_tasks: List[int],
        save_exp_name: str,
        save_file_name: str,
        load_exp_name: str,
        load_file_name: str,
        load_ckpt_num: int,
        device: torch.device,
        **config,
    ) -> None:
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.device = device

        self.num_iterations: int = config["num_iterations"]
        self.num_sample_tasks: int = config["num_sample_tasks"]

        self.num_init_samples: int = config["num_init_samples"]
        self.num_prior_samples: int = config["num_prior_samples"]
        self.num_posterior_samples: int = config["num_posterior_samples"]

        self.num_meta_grads: int = config["num_meta_grads"]
        self.meta_batch_size: int = config["meta_batch_size"]
        self.batch_size: int = config["batch_size"]
        self.max_step: int = config["max_step"]

        self.sampler = Sampler(env=env, agent=agent, max_step=config["max_step"], device=device)

        # 分离并初始化回放缓冲区
        # -用于RL更新的缓冲区
        # -用于更新编码器的缓冲区
        self.rl_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )
        self.encoder_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )

        if not save_file_name:
            save_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_path = os.path.join("results", save_exp_name, save_file_name)

        self.writer = SummaryWriter(log_dir=self.result_path)

        if load_exp_name and load_file_name:
            ckpt_path = os.path.join(
                "results",
                load_exp_name,
                load_file_name,
                "checkpoint_" + str(load_ckpt_num) + ".pt",
            )
            ckpt = torch.load(ckpt_path)

            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.encoder.load_state_dict(ckpt["encoder"])
            self.agent.qf1.load_state_dict(ckpt["qf1"])
            self.agent.qf2.load_state_dict(ckpt["qf2"])
            self.agent.target_qf1.load_state_dict(ckpt["target_qf1"])
            self.agent.target_qf2.load_state_dict(ckpt["target_qf2"])
            self.agent.log_alpha = ckpt["log_alpha"]
            self.agent.alpha = ckpt["alpha"]
            self.rl_replay_buffer = ckpt["rl_replay_buffer"]
            self.encoder_replay_buffer = ckpt["encoder_replay_buffer"]

        # 设置早期停止学习的条件
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])
        self.num_stop_conditions: int = config["num_stop_conditions"]
        self.stop_goal: int = config["stop_goal"]
        self.is_early_stopping = False

    def collect_train_data(
        self,
        task_index: int,
        max_samples: int,
        update_posterior: bool,
        add_to_enc_buffer: bool,
    ) -> None:
        # 收集给定索引任务的路径数据
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = False

        cur_samples = 0
        while cur_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - cur_samples,
                update_posterior=update_posterior,
                accum_context=False,
            )
            cur_samples += num_samples

            # 将收集的数据保存到RL回放缓冲区
            self.rl_replay_buffer.add_trajs(task_index, trajs)
            if add_to_enc_buffer:
                # 将收集的数据保存到编码器回放缓冲区
                self.encoder_replay_buffer.add_trajs(task_index, trajs)

            if update_posterior:
                # 根据示例context更新posterior
                context_batch = self.sample_context(np.array([task_index]))
                self.agent.encoder.infer_posterior(context_batch)

    def sample_context(self, indices: np.ndarray) -> torch.Tensor:
        # 编码器缓冲区中与给定索引对应的任务的context示例
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            context_batch.append(
                np.concatenate((batch["cur_obs"], batch["actions"], batch["rewards"]), axis=-1),
            )
        return torch.Tensor(context_batch).to(self.device)

    def sample_transition(self, indices: np.ndarray) -> List[torch.Tensor]:
        # RL缓冲区中与给定索引对应的任务的路径示例
        cur_obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for index in indices:
            batch = self.rl_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            cur_obs.append(batch["cur_obs"])
            actions.append(batch["actions"])
            rewards.append(batch["rewards"])
            next_obs.append(batch["next_obs"])
            dones.append(batch["dones"])

        cur_obs = torch.Tensor(cur_obs).view(len(indices), self.batch_size, -1).to(self.device)
        actions = torch.Tensor(actions).view(len(indices), self.batch_size, -1).to(self.device)
        rewards = torch.Tensor(rewards).view(len(indices), self.batch_size, -1).to(self.device)
        next_obs = torch.Tensor(next_obs).view(len(indices), self.batch_size, -1).to(self.device)
        dones = torch.Tensor(dones).view(len(indices), self.batch_size, -1).to(self.device)
        return [cur_obs, actions, rewards, next_obs, dones]

    def meta_train(self) -> None:
        # total_start_time 和 start_time 记录总的训练时间和每次迭代开始的时间
        total_start_time: float = time.time()
        # self.num_iterations 是配置文件的 num_iterations ，元训练过程的迭代次数，数值是1000。
        for iteration in range(self.num_iterations):
            start_time: float = time.time()

            # 代码在第0次迭代搜集用于训练和验证的状态转移数据。仅在第一个重复步骤中，收集所有元训练任务的路径并将其保存到回放缓冲区
            if iteration == 0:
                print("Collecting initial pool of data for train and eval")
                # 循环变量 index 指引每个任务，对每个任务做初始化操作，然后开始搜集状态转移数据。self.train_tasks 表示训练集内部的总任务数量
                for index in tqdm(self.train_tasks):
                    self.env.reset_task(index)
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_init_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

            print(f"\n=============== Iteration {iteration} ===============")
            # 代码首先遍历 self.num_sample_tasks 次，从训练任务集合中抽取一个任务环境，重置并清空里面的编码器经验池子。如果 self.num_prior_samples 大于0，则用标准正态分布抽样得到的隐藏层变量然后获得一批 self.num_prior_samples 大小的数据，放入经验池子，此时不进行后验推断。随后，如果 self.num_posterior_samples 大于0，用后验分布抽样得到的隐藏层变量然后获得一批大小为 self.num_posterior_samples 的数据，不放入经验池子，此时进行后验推断。
            for i in range(self.num_sample_tasks):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                #  。。之后开始进行元梯度更新。在元梯度迭代次数 self.num_meta_grads 内，在 self.train_tasks 内采样self.meta_batch_size 大小的下标，用于指代这么多的任务
                if self.num_prior_samples > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with prior")
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_prior_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # 清除这些任务的隐藏层变量z，赋值标准正态分布的采样，随后采样上下文 context_batch 和状态转移 transition_batch 数据。
                # 学习RL策略时，还使用由z到posterior q（z | c）生成的路径
                if self.num_posterior_samples > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with posterior")
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_posterior_samples,
                        update_posterior=True,
                        add_to_enc_buffer=False,
                    )

            # 随后执行 self.agent.train_model() 方法进行元训练，然后再调用 self.meta_test() 做元测试。对于早结束的情况做了一些异常的提示。
            print(f"Start meta-gradient updates of iteration {iteration}")
            for i in range(self.num_meta_grads):
                indices: np.ndarray = np.random.choice(self.train_tasks, self.meta_batch_size)

                # 初始化编码器的context和隐藏状态
                self.agent.encoder.clear_z(num_tasks=len(indices))

                # Context布局示例
                context_batch: torch.Tensor = self.sample_context(indices)

                # 路径放置示例
                transition_batch: List[torch.Tensor] = self.sample_transition(indices)

                # 在SAC算法中学习策略、Q函数和编码器网络
                log_values: Dict[str, float] = self.agent.train_model(
                    meta_batch_size=self.meta_batch_size,
                    batch_size=self.batch_size,
                    context_batch=context_batch,
                    transition_batch=transition_batch,
                )

                # 阻止编码器中的Task变量z的Backpropagation
                self.agent.encoder.task_z.detach()

            # 元测试任务中的学习性能评估
            self.meta_test(iteration, total_start_time, start_time, log_values)

            if self.is_early_stopping:
                print(
                    f"\n==================================================\n"
                    f"The last {self.num_stop_conditions} meta-testing results are {self.dq}.\n"
                    f"And early stopping condition is {self.is_early_stopping}.\n"
                    f"Therefore, meta-training is terminated.",
                )
                break

    def collect_test_data(
        self,
        max_samples: int,
        update_posterior: bool,
    ) -> List[List[Dict[str, np.ndarray]]]:
        # 收集元测试任务的路径数据
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = True

        cur_trajs = []
        cur_samples = 0
        while cur_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - cur_samples,
                update_posterior=update_posterior,
                accum_context=True,
            )
            cur_trajs.append(trajs)
            cur_samples += num_samples

            # 根据Context更新posterior
            self.agent.encoder.infer_posterior(self.agent.encoder.context)
        return cur_trajs

    def visualize_within_tensorboard(self, test_results: Dict[str, Any], iteration: int) -> None:
        # 将元训练和元测试结果写入张量板
        self.writer.add_scalar(
            "test/return_before_infer",
            test_results["return_before_infer"],
            iteration,
        )
        self.writer.add_scalar("test/return_after_infer", test_results["return_after_infer"], iteration)
        if self.env_name == "vel":
            self.writer.add_scalar(
                "test/sum_run_cost_before_infer",
                test_results["sum_run_cost_before_infer"],
                iteration,
            )
            self.writer.add_scalar(
                "test/sum_run_cost_after_infer",
                test_results["sum_run_cost_after_infer"],
                iteration,
            )
            for step in range(len(test_results["run_cost_before_infer"])):
                self.writer.add_scalar(
                    "run_cost_before_infer/iteration_" + str(iteration),
                    test_results["run_cost_before_infer"][step],
                    step,
                )
                self.writer.add_scalar(
                    "run_cost_after_infer/iteration_" + str(iteration),
                    test_results["run_cost_after_infer"][step],
                    step,
                )
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("train/qf1_loss", test_results["qf1_loss"], iteration)
        self.writer.add_scalar("train/qf2_loss", test_results["qf2_loss"], iteration)
        self.writer.add_scalar("train/encoder_loss", test_results["encoder_loss"], iteration)
        self.writer.add_scalar("train/alpha_loss", test_results["alpha_loss"], iteration)
        self.writer.add_scalar("train/alpha", test_results["alpha"], iteration)
        self.writer.add_scalar("train/z_mean", test_results["z_mean"], iteration)
        self.writer.add_scalar("train/z_var", test_results["z_var"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        # 元测试
        test_results = {}
        return_before_infer = 0
        return_after_infer = 0
        run_cost_before_infer = np.zeros(self.max_step)
        run_cost_after_infer = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            trajs: List[List[Dict[str, np.ndarray]]] = self.collect_test_data(
                max_samples=self.max_step * 2,
                update_posterior=True,
            )

            return_before_infer += np.sum(trajs[0][0]["rewards"])
            return_after_infer += np.sum(trajs[1][0]["rewards"])
            if self.env_name == "vel":
                for i in range(self.max_step):
                    run_cost_before_infer[i] += trajs[0][0]["infos"][i]
                    run_cost_after_infer[i] += trajs[1][0]["infos"][i]

        test_results["return_before_infer"] = return_before_infer / len(self.test_tasks)
        test_results["return_after_infer"] = return_after_infer / len(self.test_tasks)
        if self.env_name == "vel":
            test_results["run_cost_before_infer"] = run_cost_before_infer / len(self.test_tasks)
            test_results["run_cost_after_infer"] = run_cost_after_infer / len(self.test_tasks)
            test_results["sum_run_cost_before_infer"] = sum(
                abs(run_cost_before_infer / len(self.test_tasks)),
            )
            test_results["sum_run_cost_after_infer"] = sum(
                abs(run_cost_after_infer / len(self.test_tasks)),
            )
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["qf1_loss"] = log_values["qf1_loss"]
        test_results["qf2_loss"] = log_values["qf2_loss"]
        test_results["encoder_loss"] = log_values["encoder_loss"]
        test_results["alpha_loss"] = log_values["alpha_loss"]
        test_results["alpha"] = log_values["alpha"]
        test_results["z_mean"] = log_values["z_mean"]
        test_results["z_var"] = log_values["z_var"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        # 检查学习结果是否满足提前停止条件
        if self.env_name == "dir":
            self.dq.append(test_results["return_after_infer"])
            if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
                self.is_early_stopping = True
        elif self.env_name == "vel":
            self.dq.append(test_results["sum_run_cost_after_infer"])
            if all(list(map((lambda x: x <= self.stop_goal), self.dq))):
                self.is_early_stopping = True

        # 保存学习模型
        if self.is_early_stopping:
            ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
            torch.save(
                {
                    "policy": self.agent.policy.state_dict(),
                    "encoder": self.agent.encoder.state_dict(),
                    "qf1": self.agent.qf1.state_dict(),
                    "qf2": self.agent.qf2.state_dict(),
                    "target_qf1": self.agent.target_qf1.state_dict(),
                    "target_qf2": self.agent.target_qf2.state_dict(),
                    "log_alpha": self.agent.log_alpha,
                    "alpha": self.agent.alpha,
                    "rl_replay_buffer": self.rl_replay_buffer,
                    "encoder_replay_buffer": self.encoder_replay_buffer,
                },
                ckpt_path,
            )
