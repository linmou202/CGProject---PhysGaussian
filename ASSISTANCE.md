# Train Material



#### train_one_step 流程:

1. 将粒子重置为第 0 帧时的状态 (初始状态)
2. 

![image-20260607203010336](C:\Users\linmo\AppData\Roaming\Typora\typora-user-images\image-20260607203010336.png)

![image-20260607203047711](C:\Users\linmo\AppData\Roaming\Typora\typora-user-images\image-20260607203047711.png)

window_scheduler: LiinearStepAnneal:

total_iters: 训练总共的步数

warmup_step: 启动步数，这几步的窗口统一按第 0 步的窗口处理。

start_state: 最小窗口

end_state: 最大窗口

第 warmup_step + i 步的时间窗口线性增加



self.scheduler:

控制优化器的学习率的，在 warm_up 后逐渐减小学习率。



总训练：

1. 步
   1. 分片
      1. substep

步级：

train_iters: 总训练步数

gradient_accumulate_steps: 多少步的 backward 做一次 step

window_size: 这一步将进行多少帧的仿真

片级

temporal_stride: 每一次分片进行多少帧的仿真

start/end_time_idx: 这一个分片的起始/终止帧号



delta_time: 一帧的长度

substep_size: 一个 substep 的步长

num_substep: 一帧由多少个 substep 构成



checkpoint_steps: 多少 substep 后存一次档



已经实施的改动：

删去了速度相关部分

删去了 freeze_mask，因为所有点都需要仿真

删去了 # setup boundary condition 部分，因为这一部分似乎是用来固定不动点用的。

用 E_module 取代了 E_nu_list



待实施的改动：

修改 init_boundary_conditions 函数

增加参数 reference_video 的传入

在 set_require_grad 里面为 cov 和 rotation 加上梯度计算
