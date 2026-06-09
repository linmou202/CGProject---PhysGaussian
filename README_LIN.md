这次框架修改 `config, mpm_solver_warp, utils, ref_data` 这三个文件夹以及 `gs_simulation.py, gs_test.py, train_materials.py` 这三个文件。为了防止对环境的破坏，只替换这些即可。





测试的 smoke test 指令参考：

```shell
python gs_test.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_config.json --lr 1e-3 --render_img --white_bg --compile_video
```



正式运行指令参考：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_config.json --train_iters 10 --warmup_steps 2 --lr 1e-3 --num_frames 50 --temporal_stride 3 --loss_decay 0.95 --render_img --compile_video --white_bg
```

