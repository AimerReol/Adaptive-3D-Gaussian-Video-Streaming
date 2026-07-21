import copy
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
from env_qoe_train_1 import env_qoe
import datetime
from Exp_pool_train_1 import Exp_pool
from Config_train_1 import Config
import pandas as pd
from time import *

# 创建train_acc.csv和var_acc.csv文件，记录loss和accuracy
df = pd.DataFrame(columns=['episode', 'time', 'reward'])  # 列名
df.to_csv('./Date_out/train_record.csv', index=False)  # 路径可以根据需要更改

# 这样设置以后，程序就会按需占用GPU显存。
# TensorFlow 在分配显存的时候，设置为可以逐步分配，需要多少显存，就一点点往上添，直到够用为止。
# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# 设置当前程序可见的设备范围
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 通过 tf.config.experimental.set_memory_growth 将 GPU 的显存使用策略设置为 “仅在需要时申请显存空间”
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

# QoE参数（c，x）
# 参数只改Config文件即可
config = Config()
# N个GoF
N = config.N
# 切块数量
tile_num = config.tile_num
# 质量等级数量  4
level_num = config.level_num

# 质量（8*4）+数据量（8*4）+当前视角（1*8）+前一视角（1*8）+剩余段数（1）+缓冲区状态（1）+带宽状态（1） + 前一决策变量（8）
# 32+32+8+8+1+1+1+8=91
c_num = config.c_num
# CT 网络隐藏层神经元数量
hide_units = 128

# num_episodes训练的总episode数量
num_episodes = config.num_episodes
# num_exploration_episodes探索过程所占的episode数量
num_exploration_episodes = config.num_exploration_episodes
# 批次大小
batch_size = config.batch_size
# 学习率
learning_rate = config.learning_rate
# 经验池
memory_size = config.memory_size
# 折扣因子
gamma = config.gamma
# 探索起始时的探索率
initial_epsilon = config.initial_epsilon
# 探索终止时的探索率
final_epsilon = config.final_epsilon


# 输入是当前的状态和行动，输出则是采取该行动在特定状态下的预期收益。
# Q-network用于拟合Q函数，和前节的多层感知机类似。输入state（1*hide_units），输出各个action下的Q-value(x_num)
class QNetwork(tf.keras.Model):
    # @tf.function
    def __init__(self):
        super().__init__()
        # 全连接层，带有泄露的激活函数，units=s_len表示该层的神经元数量为s_len，也是输出空间的维度
        self.dense1 = tf.keras.layers.Dense(units=hide_units, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        # Dropout是一种正则化技术，用于防止过拟合。
        # 其中参数0.05表示每个神经元被设置为0的概率为5%。
        # 这意味着在训练过程中，大约5%的神经元会被随机设置为0。
        self.drop1 = tf.keras.layers.Dropout(0.05)
        # LayerNormalization是一种归一化技术，用于加速神经网络的训练过程并提高模型的性能。
        # 其中参数axis=-1表示对最后一个维度进行归一化。这意味着在训练过程中，该层的输入数据会被归一化到均值为0、标准差为1的分布上。
        self.bn1 = tf.keras.layers.LayerNormalization(axis=-1)
        # self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)

        self.dense2 = tf.keras.layers.Dense(units=hide_units, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop2 = tf.keras.layers.Dropout(0.05)
        self.bn2 = tf.keras.layers.LayerNormalization(axis=-1)

        self.dense3 = tf.keras.layers.Dense(units=hide_units, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop3 = tf.keras.layers.Dropout(0.05)
        self.bn3 = tf.keras.layers.LayerNormalization(axis=-1)

        # 创建了一个全连接层，其中参数units=1表示该层的神经元数量为1。激活函数使用的是带有泄露参数（alpha=0.02）的Leaky ReLU激活函数。
        # 这种全连接层通常用于构建神经网络模型中的输出层，用于对输入数据进行分类或回归预测。
        self.V_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.02))

        self.A_dense = tf.keras.layers.Dense(units=level_num * tile_num,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.02))

    @tf.function
    # 输出Q值(batch_size, tile_num*level_num) (256,32)   输入state(256*91) (batch_size, hide_units)
    def call(self, inputs):
        # 输入数据转化为tensor类型
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        FN = self.dense1(x)
        FN = self.drop1(FN)
        FN = self.bn1(FN)
        FN = self.dense2(FN)
        FN = self.drop2(FN)
        FN = self.bn2(FN)
        FN = self.dense3(FN)
        FN = self.drop3(FN)
        FN = self.bn3(FN)

        # 状态价值V函数
        s_value = self.V_dense(FN)
        # 状态下的动作价值函数A
        a_value = self.A_dense(FN)
        # 它首先使用tf.reduce_mean函数计算a_value在axis=1方向上的均值
        # 然后通过keepdims参数设置为True来保持结果的维度。
        # 最后，将计算得到的均值作为输出。
        # 用Lambda层，计算avg(a)
        mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(a_value)
        # a - avg(a)  输出的动作价值-平均动作价值
        advantage = a_value - mean
        output = s_value + advantage
        x = output

        return x

    # 输出action，输出每4组Q值中最大的下标，则为当前块的质量级别，输出8个值（32个值中每4个比较一下）
    @tf.function
    def predict(self, inputs):  # 输出一个数组，表示每行最大的值的下标
        # Q值表示在给定状态下采取不同动作的预期回报。  # q_values（Q表）的长度是8*4=32
        q_values = self(inputs)
        # q_values = self.call(inputs)
        # 使用tf.argmax(q_values, axis=-1)函数找到Q值中最大值对应的动作索引。
        # axis=-1表示沿着最后一个维度（列）进行操作。
        # axis=-1表示沿着第一个维度（行）进行操作。
        # 生成一个列表，长度为8，每个包含4个级别
        # q_values = tf.convert_to_tensor(q_values.numpy().reshape(8, 4))
        q_values = tf.reshape(q_values, [8, 4])
        # return tf.argmax(q_values, axis=-1)
        return tf.argmax(q_values, axis=1)


# Target网络的主要作用是生成目标Q值，以减小训练过程中的估计误差。
# 一开始，Q网络和Target网络的参数是相同的。然后每隔一定的时间步，就会将Q网络的参数同步给Target网络。
# 这样做的原因是，由于Target网络的参数相对固定，可以用它来获取稳定的Q值
# 从而减少高估现象的发生。而Q网络则负责不断地更新和优化，以逼近真实的Q值函数。
class Target_QNetwork(tf.keras.Model):
    # @tf.function
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=hide_units, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop1 = tf.keras.layers.Dropout(0.05)
        self.bn1 = tf.keras.layers.LayerNormalization(axis=-1)
        # self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dense2 = tf.keras.layers.Dense(units=hide_units, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop2 = tf.keras.layers.Dropout(0.05)
        self.bn2 = tf.keras.layers.LayerNormalization(axis=-1)

        self.dense3 = tf.keras.layers.Dense(units=hide_units, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.drop3 = tf.keras.layers.Dropout(0.05)
        self.bn3 = tf.keras.layers.LayerNormalization(axis=-1)

        self.V_dense = tf.keras.layers.Dense(units=1, activation=tf.keras.layers.LeakyReLU(alpha=0.02))
        self.A_dense = tf.keras.layers.Dense(units=level_num * tile_num,
                                             activation=tf.keras.layers.LeakyReLU(alpha=0.02))

    # 输出Q值
    @tf.function
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)
        FN = self.dense1(x)
        FN = self.drop1(FN)
        FN = self.bn1(FN)
        FN = self.dense2(FN)
        FN = self.drop2(FN)
        FN = self.bn2(FN)
        FN = self.dense3(FN)
        FN = self.drop3(FN)
        FN = self.bn3(FN)

        # 状态价值V函数
        s_value = self.V_dense(FN)
        # 状态下的动作价值函数A
        a_value = self.A_dense(FN)
        mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(a_value)
        advantage = a_value - mean
        output = s_value + advantage
        x = output
        return x

    # 输出行动
    @tf.function
    def predict(self, inputs):
        q_values = self(inputs)
        q_values = tf.reshape(q_values, [8, 4])
        # return tf.argmax(q_values, axis=-1)
        return tf.argmax(q_values, axis=1)

@tf.function
def train_step(model, batch_action, batch_state, y):
    with tf.GradientTape() as tape:
        temp = tf.one_hot(batch_action, depth=level_num)
        # prediction = tf.reduce_sum(model(batch_state) * tf.convert_to_tensor(temp.numpy().reshape(256, 32)), axis=1)
        prediction = tf.reduce_sum(model(batch_state) * tf.reshape(temp, [256, 32]), axis=1)
        # 计算了预测值
        # model(batch_state) 是模型对输入状态（batch_state）的输出
        # tf.one_hot(batch_action, depth=x_num) 将动作（batch_action）转换为 one-hot 编码
        # 然后与模型输出相乘，最后沿着轴 1 求和得到预测值。
        # 最小化 y 和 Q-value 的距离 计算了真实值（y_true）和预测值（y_pred）之间的均方误差
        loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
            y_true=y,
            y_pred=prediction
        )
    # 计算模型变量（model.variables）相对于损失函数（loss）的梯度
    # model.variables 这一属性直接获得模型中的所有变量
    grads = tape.gradient(loss, model.variables)
    # 计算梯度并更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

@tf.function
def train_step_CT(target_net, model, batch_action, batch_next_state, batch_state, batch_stop_signal):
    batch_next_state_float32 = tf.cast(batch_next_state, tf.float32)
    target_q_value = target_net(batch_next_state_float32)
    # process_1 = time()
    # tf.reduce_max计算张量 target_q_value 沿着指定轴（这里是轴1）的最大值。
    # 计算 y 值  输出维度（256*1） （batch_size, 1）
    y = batch_reward + (gamma * tf.reduce_max(target_q_value, axis=1)) * (1 - batch_stop_signal)
    with tf.GradientTape() as tape:
        temp = tf.one_hot(batch_action, depth=level_num)
        # prediction = tf.reduce_sum(model(batch_state) * tf.convert_to_tensor(temp.numpy().reshape(256, 32)), axis=1)
        prediction = tf.reduce_sum(model(batch_state) * tf.reshape(temp, [256, 32]), axis=1)
        # 计算了预测值
        # model(batch_state) 是模型对输入状态（batch_state）的输出
        # tf.one_hot(batch_action, depth=x_num) 将动作（batch_action）转换为 one-hot 编码
        # 然后与模型输出相乘，最后沿着轴 1 求和得到预测值。
        # 最小化 y 和 Q-value 的距离 计算了真实值（y_true）和预测值（y_pred）之间的均方误差
        loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
            y_true=y,
            y_pred=prediction
        )
    # 计算模型变量（model.variables）相对于损失函数（loss）的梯度
    # model.variables 这一属性直接获得模型中的所有变量
    grads = tape.gradient(loss, model.variables)
    # # 计算梯度并更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


if __name__ == '__main__':
    # 显示当前程序开始时间
    process_start_time = datetime.datetime.now()
    # 实例化一个环境
    env = env_qoe(config)
    # 均奖励集合（列表）
    reward_avg_set = []
    # 实例化模型和优化器
    # 创建新模型训练
    # 创建Q网络模型
    model = QNetwork()
    # 创建目标Q网络模型
    target_net = Target_QNetwork()

    # 模型权重
    weights = model.get_weights()
    # target_net.set_weights(weights) 是 Keras 中用于设置目标网络权重的方法。
    # 它接受一个包含模型所有层的权重的列表作为参数，并将这些权重应用到目标网络中。
    # 更新目标网络的权重以实现更好的性能
    # 目标网络权重
    target_net.set_weights(weights)

    # 读取模型继续训练
    # model = QNetwork()
    # target_net = Target_QNetwork()
    # model.load_weights('./checkpoints/my_checkpoint')
    # target_net.load_weights('./checkpoints/target_weight')
    # 使用 TensorFlow 的 Keras API 创建 Adam 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
    model_checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)
    target_net_checkpoint = tf.train.Checkpoint(myAwesomeModel=target_net, myAwesomeOptimizer=optimizer)

    # 读取训练集
    # tra = np.load('dataset.npy')   300*80  32+8+32+8
    tra = np.load('./Date/dateset_python_CT2_1.npy')
    train_set = tra.tolist()  # 将训练集转换为列表
    # 导入带宽数据 5*300
    tra_BW = np.load('./Date/dateset_python_BW_5.npy')
    train_BW_set = tra_BW.tolist()

    # 创建经验池
    pool = Exp_pool(config)
    # 探索起始时的探索率
    epsilon = initial_epsilon

    # num_episodes训练的总episode数量  每个episode的循环
    for episode_id in range(num_episodes):
        # 两 runs 之间清一次计算图
        # tf.compat.v1.reset_default_graph()
        # 开始循环时间
        begin_time_all = time()
        # 每个episode开始时，更新target_net参数
        # 当 episode_id 大于 200 且 episode_id 能被 3 整除时，将当前模型的权重赋值给目标网络。
        # 先初始化目标网络，然后隔一段时间进行更新
        if episode_id > 200 and episode_id % 3 == 0:
            # if episode_id > 50 and episode_id % 5 == 0:
            weights = model.get_weights()
            target_net.set_weights(weights)
        # 一个episode时间记录
        episode_time_record = 0

        # 随机5类选取带宽数据  1*300
        random_BW = random.randint(0, 4)
        train_BW__current = train_BW_set[random_BW]

        # 初始化环境，获得初始状态　
        # 0-82 300*83  不包含当前决策  0-71:Q,f,S  72:remain 73:BW 74:buff 75-82:x_1
        # reset维度扩展300*91  同时写入带宽和剩余段落数据
        state_s = env.reset(train_set, train_BW__current)
        # 拷贝扩展后的状态
        state_s_temp = copy.deepcopy(state_s)
        # 初始化段落  N（300）个-1 表示未被选择
        # download_segment = np.array([-1] * N)
        down_id = 0
        # 创建动作空间（数组）
        action = np.array([0] * tile_num)
        # 缓冲区初始化
        buf = 0
        # 总奖励初始化
        reward_total_chunk = 0
        # 平均奖励初始化
        reward_avg = 0

        # 计算当前探索率
        # 具体来说，这段代码首先计算了初始epsilon值与最终epsilon值之间的线性插值。
        # 初始epsilon值是initial_epsilon 最终epsilon值是final_epsilon，当前episode的id是episode_id
        # 总共的探索episode数量是num_exploration_episodes。
        # epsilon是一种在强化学习中常用的探索-利用权衡策略，它决定了智能体在每个决策阶段应该进行探索还是利用。
        # 这个公式的意思是，随着episode的进行，epsilon的值从初始值逐渐减小到最终值。
        # 开始时，epsilon的值接近于初始值，因为智能体需要更多的时间来探索环境；
        # 而当episode达到一定数量后，epsilon的值会逐渐减小，因为智能体开始更多地利用其已经学到的知识。
        epsilon = max(initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes,
                      final_epsilon)
        # 步停止标志初始化
        stop_signal = 0
        # 步计数初始化
        step_count = 0
        # 程序各部分时间初始化
        # 从经验池中抽取数据的时间
        batch_process_time = 0
        # 进行网络模型训练进行梯度下降的时间
        net_train_time = 0
        # 智能体交互时间，包括动作选择，并获得奖励更新状态，添加经验池
        agent_interaction_time = 0

        # while一直循环,直到最后一个片段被选择（300段落循环）stop_signal = 1
        while stop_signal == 0:
            step_count += 1
            agent_interaction_begin_time = time()
            # id变为298，则当前状态是298，未来状态是299，结束，并计算总奖励
            if down_id == N - 2:
                stop_signal = 1
            # 获取当前chunk状态  1*91  作为模型的输入，输出预测的动作
            state_current_temp = state_s_temp[down_id]
            # epsilon-greedy 探索策略，以 epsilon 的概率选择随机动作 一次选择tile_num数量的动作
            # action = model.predict(np.expand_dims(state_current_temp, axis=0)).numpy()
            if random.random() < epsilon:  # epsilon-greedy探索策略
                # 选择随机动作（探索）避免机械化动作选择（随着训练次数越多，探索的概率越低）
                # 0-level-1中随机选择8个动作
                action = env.random_action()  # 以epsilon的概率选择随机动作
            else:
                # 使用np.expand_dims(state[0], axis=0)将状态state[0]扩展为一个形状为(1, ...)的数组
                # 其中...表示其他维度的大小。这样做是为了确保输入数据的形状与模型期望的输入形状一致。
                # 调用model.predict()方法对扩展后的状态进行预测，得到一个包含预测动作的数组。
                # 使用.numpy()方法将预测结果转换为NumPy数组。
                # tf.constant()：将扩展后的数组转换为 TensorFlow 常量张量。
                # dtype=tf.float32：指定张量的数据类型为 32 位浮点数（即单精度浮点数）。

                # 选择模型计算出的 Q Value 最大的动作 该state只有300*91  action是长度为8的数组
                action = model.predict(np.expand_dims(state_current_temp, axis=0)).numpy()

            # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
            # 通过使用deepcopy函数（深拷贝），我们可以确保在对state进行修改时，不会影响到原始的state对象。
            # 这样可以避免意外地修改了原始数据，导致不可预期的结果。

            # 动作更新，视野外给基础层
            # fov_temp = np.array([0] * config.tile_num)
            fov_temp = state_current_temp[config.QSP_num:
                                          config.QSP_num + config.fov_num]
            fov_current = np.array(fov_temp, dtype=np.int32)
            for i in range(tile_num):
                if fov_current[i] == 0:
                    action[i] = 0

            # 使用step函数获取更新的buffer，当前奖励
            reward_current, buf_next = env.step(action, state_s_temp, down_id)

            # 更新状态id
            down_id += 1
            # 将当前的动作写入未来状态,表示未来状态中的前一动作选择
            for i in range(tile_num):
                state_s_temp[down_id][75 + i] = action[i]
            # 将当前更新的buf写入未来状态,初始为0
            state_s_temp[down_id][74] = buf_next
            # 选出更新后的chunk的状态   1*83
            state_next_temp = state_s_temp[down_id]

            # 向经验池添加项目()
            # state_temp 1*83  action  8  reward 1  state_next_temp 1*83  stop  1
            # 将(state, action, reward, next_state)的四元组（外加 done 标签表示是否结束）放入经验回放池
            pool.add(state_current_temp, action, reward_current, state_next_temp, stop_signal)

            # 记录总奖励
            reward_total_chunk += reward_current
            # 智能体交互结束时间，包括动作选择，并获得奖励更新状态，添加经验池
            agent_interaction_end_time = time()
            # 经验池容量大于批次后，从经验池中随机选取batch进行训练
            if pool.count >= batch_size:
                # 开始从经验池中抽取数据的时间
                batch_process_begin_time = time()
                # 从经验回放池中随机取一个批次的四元组
                batch_state, batch_action, batch_reward, batch_next_state, batch_stop_signal = pool.get_batch()

                # 将四个列表（batch_state, batch_reward, batch_next_state, batch_done）转换为NumPy数组，并将它们的数据类型设置为float32。
                batch_state, batch_reward, batch_next_state, batch_stop_signal = \
                    [np.array(a, dtype=np.float32) for a in
                     [batch_state, batch_reward, batch_next_state, batch_stop_signal]]
                # 将batch_action列表转换为NumPy数组，并将数据类型设置为int32。
                batch_action = np.array(batch_action, dtype=np.int32)

                # 经验池批量抽取结束
                batch_process_end_time = time()

                # 依次判断target_net对每个batch的输出是否满足约束，不满足的项目剔除，不参与损失函数计算
                # target_net_action = np.argmax(q_value, axis=1)
                # ti_01 = np.empty(batch_size)
                # for ti in range(batch_size):
                #     ti_01[ti] = env.wrong_choose(batch_next_state[ti], target_net_action[ti])
                # ti_index = np.where(np.array(ti_01) == 1)
                # # 剔除操作
                # batch_state = np.delete(batch_state, ti_index, axis=0)
                # batch_action = np.delete(batch_action, ti_index, axis=0)
                # batch_reward = np.delete(batch_reward, ti_index, axis=0)
                # batch_next_state = np.delete(batch_next_state, ti_index, axis=0)
                # batch_done = np.delete(batch_done, ti_index, axis=0)
                # # q_value = np.delete(q_value, ti_index, axis=0)
                # q_value = model(batch_next_state)

                # 开始进行网络模型训练进行梯度下降的时间
                net_train_begin_time = time()
                # 目标值计算
                # target_q_value = target_net(batch_next_state) 输入是256*91  输出是256*4
                target_q_value = target_net(tf.constant(batch_next_state, dtype=tf.float32))
                # tf.reduce_max计算张量 target_q_value 沿着指定轴（这里是轴1）的最大值。
                # 计算 y 值  输出维度（256*1） （batch_size, 1）
                y = batch_reward + (gamma * tf.reduce_max(target_q_value, axis=1)) * (1 - batch_stop_signal)
                #
                # # 梯度下降更新模型
                # with tf.GradientTape() as tape:
                #     temp = tf.one_hot(batch_action, depth=level_num)
                #     # prediction = tf.reduce_sum(model(batch_state) * tf.convert_to_tensor(temp.numpy().reshape(256, 32)), axis=1)
                #     prediction = tf.reduce_sum(model(batch_state) * tf.reshape(temp, [256, 32]), axis=1)
                #     # 计算了预测值
                #     # model(batch_state) 是模型对输入状态（batch_state）的输出
                #     # tf.one_hot(batch_action, depth=x_num) 将动作（batch_action）转换为 one-hot 编码
                #     # 然后与模型输出相乘，最后沿着轴 1 求和得到预测值。
                #     # 最小化 y 和 Q-value 的距离 计算了真实值（y_true）和预测值（y_pred）之间的均方误差
                #     loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
                #         y_true=y,
                #         y_pred=prediction
                #     )
                # # 计算模型变量（model.variables）相对于损失函数（loss）的梯度
                # # model.variables 这一属性直接获得模型中的所有变量
                # grads = tape.gradient(loss, model.variables)
                # # 计算梯度并更新参数
                # optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
                # CT 代码加速梯度下降
                train_step(model, batch_action, batch_state, y)
                # train_step_CT(target_net, model, batch_action, batch_next_state, batch_state, batch_stop_signal)

                # 网络更新结束
                net_train_end_time = time()

                # 分块时间计算
                # 从经验池中抽取批次的时间
                batch_process_time += batch_process_end_time - batch_process_begin_time
                # 网络训练时间
                net_train_time += net_train_end_time - net_train_begin_time
                # 智能体交互时间，选取动作获得奖励和更新状态，添加经验池的时间
                agent_interaction_time += agent_interaction_end_time - agent_interaction_begin_time

        # 计算平均奖励
        reward_avg = reward_total_chunk / N
        reward_avg_set.append(reward_avg)

        # 结束训练
        end_time_all = time()
        # 一个episode开始到结束的时间记录
        episode_time_record = end_time_all - begin_time_all
        # 本地时间结束
        local_end_time = datetime.datetime.now()
        redundancy_time = episode_time_record - batch_process_time - net_train_time - agent_interaction_time
        # 将数据保存在一维列表
        list_record = [episode_id, episode_time_record, np.round(reward_avg, 1)]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list_record])
        data.to_csv('./Date_out/train_record.csv', mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

        # 训练过程数据存储，所有episode结束后进行保存 CT  训练过程中保存数据
        mm = np.array(reward_avg_set)
        # 将名为accuracy的数组保存为名为J_track_DDQN_clean.npy的文件
        np.save('./Date_out/QoE_1_accuracy.npy', mm)

        # 参数打印  np.round（a, n）保留a中所有数的n位小数
        print('模型训练进度：', np.round(episode_id / num_episodes * 100, 4), '%',
              '  ', '第', episode_id, 'episode', '平均QoE分数', np.round(reward_avg, 1))
        print('时间统计：', ' step数：', step_count, ' 一个episode运行时间:', episode_time_record, 's')
        print(' 经验池抽取数据时间和占比：', batch_process_time, 's',
              np.round(batch_process_time / episode_time_record * 100, 1), '%')
        print(' 网络模型训练时间和占比:', net_train_time, 's', np.round(net_train_time / episode_time_record * 100, 1),
              '%')
        print(' 智能体交互时间和占比:', agent_interaction_time, 's',
              np.round(agent_interaction_time / episode_time_record * 100, 1), '%')
        print(' 冗余时间和占比:', redundancy_time, 's', np.round(redundancy_time * 100, 1), '%')
        # 本地训练结束时间
        print('第', episode_id, 'episode结束时间', datetime.datetime.now())

    process_end_time = datetime.datetime.now()
    print('显示程序开始的时间:', process_start_time)
    print('显示程序结束的时间:', process_end_time)
    
    day_now = datetime.date.today()

    model.save('./DDQN_model', save_format="tf")

    # 打印网络信息
    # model.summary()
    # target_net.summary()

    # # 训练过程数据存储，所有episode结束后进行保存
    # mm = np.array(reward_avg_set)
    # # 将名为accuracy的数组保存为名为J_track_DDQN_clean.npy的文件
    # np.save('QoE_1_accuracy.npy', mm)

    # 模型参数保存代码修改
    # model.save_weights()主要用于保存模型的权重和优化器的状态，这是在进行模型训练时常用的一个方法。
    # model.save_weights('./DDQN_model/DDQN_model_variables/model_weight')
    # target_net.save_weights('./DDQN_model/DDQN_model_variables/target_net_weight')
    # 模型训练完毕后将参数保存到文件（也可以在模型训练过程中每隔一段时间就保存一次）
    # 每隔一段时间保存一次
    # if batch_index % 100 == 0:                              # 每隔100个Batch保存一次
    #             path = checkpoint.save('./save/model.ckpt')         # 保存模型参数到文件
    #             print("model saved to %s" % path)
    model_checkpoint.save('./DDQN_model_variables/model_weight/model_weight.ckpt')
    target_net_checkpoint.save('./DDQN_model_variables/target_net_weight/target_net_weight.ckpt')

    # 绘图
    plt.figure(1)
    plt.xlabel('number of episode')
    plt.plot(reward_avg_set)
    plt.ylabel('reward(QoE)')
    plt.savefig('./Date_out/train_reward_B{}_L{}_time_{}.jpg'.format(batch_size, learning_rate, day_now))

    # plt.figure(2)
    # plt.xlabel('number of episode')
    # plt.plot(accuracy)
    # plt.ylabel('QoE')
    # plt.savefig('./train_QoE_B{}_L{}.jpg'.format(batch_size, learning_rate))

    plt.show()

    tf.saved_model.save(model, './DDQN_model_h5')

    '''
    # train.py 模型训练阶段
    model = MyModel()
    checkpoint = tf.train.Checkpoint(myModel=model)  # 实例化Checkpoint，指定保存对象为model（如果需要保存Optimizer的参数也可加入）
    # 模型训练代码
    checkpoint.save('./save/model.ckpt')  # 模型训练完毕后将参数保存到文件，也可以在模型训练过程中每隔一段时间就保存一次
    # test.py 模型使用阶段
    model = MyModel()
    checkpoint = tf.train.Checkpoint(myModel=model)  # 实例化Checkpoint，指定恢复对象为model
    checkpoint.restore(tf.train.latest_checkpoint('./save'))  # 从文件恢复模型参数
    '''

    # 使用了tf.saved_model.save()函数将模型保存为SavedModel格式
    # tf.saved_model.save()则是用来保存整个模型的，包括模型的结构、权重以及优化器的状态等信息。
    # 当你的模型需要部署到生产环境时，你可以使用这个方法来保存模型
    # tf.saved_model.save(my_model, "the_saved_model")
    # tf.saved_model.save(model, './DDQN_model')
    # 模型加载函数
    # new_model = tf.saved_model.load("the_saved_model")
    # model = tf.saved_model.load("保存的目标文件夹名称")
    # 使用model.save_weights()和target_net.save_weights()函数将模型的权重保存到指定的文件中。
    # 对于使用继承 tf.keras.Model 类建立的 Keras 模型 model
    # 使用 SavedModel 载入后将无法再使用evaluate，predict方法，可以使用call方法
    # y_pred = model.call(data_loader.test_data[start_index: end_index])
    # data_loader.test_data[start_index: end_index]是输入，改为state
    # tf_2_model_train.py
    # res = mymodel.call(data_loader.test_data)
    # print(res)
