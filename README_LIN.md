这次框架修改 `config, mpm_solver_warp, utils, reference_data` 这三个文件夹以及 `gs_simulation.py, gs_test.py, train_materials.py` 这三个文件。为了防止对环境的破坏，只替换这些即可。

还将 config 中的 frame_dt 从 2e-2 改成了 5e-3, frame_num 则相应的从 100 改成了400，以减少显存占用。



测试的 smoke test 指令参考：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_config.json --lr 1e-3 --render_img --white_bg --compile_video
```



正式运行指令参考：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_config.json --train_iters 10 --warmup_steps 2 --lr 1e-3 --num_frames 50 --temporal_stride 3 --loss_decay 0.95 --render_img --compile_video --white_bg
```



出错可能：

1. 梯度计算错误 (访问了非叶子节点的梯度，中间某个地方不可微，梯度爆炸/消失)；可能导致报错或者输出的 E 有问题。
2. 参考图像和渲染图像格式不同，导致烟雾测试中输出的损失函数比较大 (应该非常接近 0 才对)。
3. 从物质点求解器导出的数据格式 (尤其是协方差矩阵和旋转矩阵) 不正确，导致渲染出来的图片完全错误。
