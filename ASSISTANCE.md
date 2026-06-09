# Train Material



window_scheduler: LinearStepAnneal:

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

修改 init_boundary_conditions 函数

从 ref_path 读入名字为 f"{frame_num}.png" 的图片，其中 frame_num 为该帧的编号

现在 render_utils 只会返回未经处理的图片

在 set_require_grad 里面为 cov 和 rotation 加上梯度计算



待实施的改动：

注释掉 calculate_c_and_r 的 backward 中的 assertion







smoke test 指令参考：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_config.json --lr 1e-3 --render_img --white_bg --compile_video
```



运行指令参考：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_config.json --train_iters 10 --warmup_steps 2 --lr 1e-3 --num_frames 50 --temporal_stride 3 --loss_decay 0.95 --render_img --compile_video --white_bg
```



安装指令：

```bash
conda create -n PhysGaussian python=3.9 mkl=2023.1.0 -y
conda activate PhysGaussian

conda install cudatoolkit=11.8

pip install -r requirements.txt

pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/

pip install -e gaussian-splatting/submodules/simple-knn/
```

