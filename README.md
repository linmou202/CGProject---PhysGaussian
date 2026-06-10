smoke test 指令：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_train_config.json --lr 1e-3 --render_img --white_bg --compile_video
```



渲染参考视频指令：

```shell
python render_reference.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path reference_data --ref_path reference_data --config ./config/pillow2sofa_config.json --lr 1e-3 --render_img --white_bg --compile_video
```



训练并仿真指令：

```shell
python gs_simulation.py --model_path ./model/pillow2sofa_whitebg-trained/ --output_path output --ref_path reference_data --config ./config/pillow2sofa_train_config.json --train_iters 10 --warmup_steps 2 --lr 1e-2 --num_frames 50 --temporal_stride 1 --loss_decay 0.95 --render_img --compile_video --white_bg
```



安装指令：

```shell
conda create -n PhysGaussian python=3.10 -y
conda activate PhysGaussian

conda install -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev -y
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y

pip install -r requirements.txt
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```



