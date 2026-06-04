# Zhou - 内部填充部分修改说明

这份说明记录我在内部填充这一部分做的改动，方便后面继续改主流程、VLM 或优化部分时对齐接口。

## 修改范围

主要改了两个文件：

```text
particle_filling/filling.py
gs_simulation.py
utils/decode_param.py
```

没有改：

```text
mpm_solver_warp/
utils/vlm_utils.py
utils/video_utils.py
loss/
gaussian-splatting/
```

另外在项目外层 `research/tests` 里放了几个独立测试脚本，不参与正式运行。

## 对应任务

对应 `初步设计 v4.pdf` 里的第一部分：

```text
改进内部填充方法
```

具体包括：

1. 实现 `DBSCAN_cluster`。
2. 实现 `fill_particles_MCIS`。
3. 让原来的 `fill_particles` 支持逐物体填充。
4. 完成手动版 optional：可以在 config 里为每个物体指定不同填充方法。

## filling.py 里做了什么

### 1. DBSCAN 聚类

新增并实现：

```python
DBSCAN_cluster(pos, opacity, cov, screen_points, shs)
```

输入是所有 Gaussian 的属性，输出是按物体分好的 list：

```python
cls_pos
cls_opacity
cls_cov
cls_screen_points
cls_shs
cls_size
```

这里 `cls_size[i]` 是第 `i` 个物体簇的原始 Gaussian 数量。

默认参数在 `filling.py` 顶部：

```python
DBSCAN_EPS = 0.03
DBSCAN_MIN_SAMPLES = 8
DBSCAN_MIN_CLUSTER_SIZE = 16
DBSCAN_NOISE_ATTACH_RADIUS = 0.06
```

这些值是按当前 MPM 归一化空间先定的，后面如果某个场景聚类不好，可以优先调 `DBSCAN_EPS`。

### 2. AABB

实现：

```python
get_aabb(pos, opacity, cov, screen_points, shs)
```

返回一个物体簇的 AABB 两个角点：

```python
coords_ldb
coords_ruf
```

内部优先用 `scipy.spatial.ConvexHull`，对应 FastPhysGS 里 Quickhull 的思路。为了避免包围盒太紧，还会根据 covariance 加一点 padding。后面 VLM 画框可以直接复用这个函数。

### 3. MCIS 填充

实现：

```python
fill_particles_MCIS(...)
```

流程按 FastPhysGS 的 IPF/MCIS：

```text
单物体点云
-> ConvexHull / AABB
-> AABB 内均匀采样候选点
-> 凸包 inside 筛选
-> 计算候选点到原始 Gaussian 点的最近距离
-> Gaussian 权重
-> importance sampling
-> 返回 原始点 + 新填充点
```

主要参数：

```python
MCIS_SIGMA = 0.02
MCIS_MAX_CANDIDATES = 200000
MCIS_MAX_SAMPLES = 50000
```

其中 `sigma=0.02` 是按 FastPhysGS 论文里的经验值来的。

### 4. fill_particles 新增 method 参数

原来的 `fill_particles` 默认行为不变，仍然是 PhysGaussian 原本的密度场 + 射线填充。

现在可以额外传：

```python
method="legacy"  # 原填充方法
method="mcis"    # FastPhysGS 风格 MCIS
method="none"    # 不填充
```

没有传 `method` 时默认是：

```python
method="legacy"
```

所以旧配置不会因为这个改动直接失效。

为了让 optional 不只写在主流程里，我还加了两个辅助函数：

```python
resolve_filling_methods(methods, default_method, object_count)
allocate_filling_budgets(cluster_sizes, total_budget, min_budget_per_object)
```

前一个负责把 config 里的 `method/methods` 解析成每个物体一个方法，后一个负责把全局新增粒子预算分到每个 DBSCAN 物体上。

## gs_simulation.py 里做了什么

原来主流程虽然调用了 `DBSCAN_cluster`，但后面填充时还是按单个 Gaussian 遍历，这样实际上没法逐物体填充。我把这部分改成了按 DBSCAN 得到的物体簇遍历。

关键点：

1. 只有 `particle_filling` 不为 `None` 时才跑 DBSCAN。
2. 每个物体簇单独调用 `fill_particles`。
3. 每个物体可以有自己的填充方法。
4. `mpm_init_pos[:gs_num]` 仍然保持原始 `transformed_pos` 顺序。

第 4 点很重要。后面渲染时 `init_shs / init_cov / init_opacity` 默认和原始 Gaussian 顺序对应。如果直接把每个簇的 `原始点 + 填充点` 拼起来，会导致位置和外观属性错位。所以现在的做法是：

```text
mpm_init_pos = 原始 transformed_pos + 所有新增填充粒子
```

每个簇返回结果中的原始点部分只用来确定新增粒子，不直接改变原始点顺序。

`utils/decode_param.py` 里补了 optional 相关默认值：

```python
method = "legacy"
methods = None
min_particles_num_per_object = 8192
mcis_sigma = 0.02
mcis_interior_num = None
mcis_sample_num = None
use_vlm = False
```

## 手动版 optional 怎么用

在 config 的 `particle_filling` 里可以加字段。

### 全局方法

所有物体都用同一种方法：

```json
"particle_filling": {
    "n_grid": 128,
    "density_threshold": 40.0,
    "search_threshold": 0.5,
    "search_exclude_direction": 5,
    "ray_cast_direction": 0,
    "max_particles_num": 2000000,
    "max_partciels_per_cell": 4,
    "method": "mcis"
}
```

`method` 可选：

```text
legacy
mcis
none
```

### 逐物体方法

如果一个场景里有多个 DBSCAN 物体，可以写：

```json
"particle_filling": {
    "n_grid": 128,
    "density_threshold": 40.0,
    "search_threshold": 0.5,
    "search_exclude_direction": 5,
    "ray_cast_direction": 0,
    "max_particles_num": 2000000,
    "max_partciels_per_cell": 4,
    "method": "legacy",
    "methods": ["mcis", "legacy", "none"]
}
```

含义：

```text
第 0 个物体用 mcis
第 1 个物体用 legacy
第 2 个物体用 none
```

如果 `methods` 比物体数量短，后面的物体会用全局 `method`。如果 `methods` 比物体数量长，多出来的会忽略。

### MCIS 额外参数

可以按需要加：

```json
"mcis_sigma": 0.02,
"mcis_interior_num": 80000,
"mcis_sample_num": 6000
```

不写时会自动按物体点数和预算估一个值。

### 填充预算

`max_particles_num` 现在按整个场景理解。代码会按每个物体的原始 Gaussian 数量比例分配预算。

为了避免小物体 legacy 填充时临时 buffer 太小，还加了：

```json
"min_particles_num_per_object": 8192
```

这个字段不写时默认就是 `8192`。

这里有一个细节：`min_particles_num_per_object` 只是“预算充足时尽量给每个物体的最低新增粒子数”，不会让总新增粒子数超过 `max_particles_num`。如果场景里物体很多、总预算不够，代码会退回到按物体大小比例分配。

另外我也给旧填充的 Taichi kernel 加了边界保护。以前如果某个物体内部格子太多，临时粒子 buffer 有可能被写爆；现在超过当前物体预算的点不会继续写入。

## 推荐使用方式

目前建议：

```text
实心、块状物体：legacy
空心、网状、条状、复杂非凸物体：mcis
薄片、叶片、布、纸这类不适合体积填充的物体：none
```

如果不确定，先用 `legacy`，然后看 `--debug` 导出的 `filled_particles.ply`。

## 和 VLM 部分的关系

目前 VLM 自动选择填充方式还没接。为了不影响手动测试，我把 `gs_simulation.py` 里的 VLM 调用放到了：

```python
if filling_params.get("use_vlm", False):
```

也就是说默认不会调用未完成的 VLM 函数。

后面如果 VLM 部分做好，建议让它返回：

```python
fill_methods = ["mcis", "legacy", "none"]
```

顺序必须和 `cls_pos` 的顺序一致。最稳妥的做法是 VLM 画框时就用 `get_aabb(cls_pos[i], ...)` 画第 `i` 个框，这样 VLM 输出的编号和 DBSCAN 簇编号能对上。

## 测试情况

我用 Python 3.13 做了独立测试，没有跑完整 `gs_simulation.py`，因为完整主流程还需要 `warp` 和本地 rasterizer 扩展。

已通过的测试：

```text
py -3.13 -m py_compile particle_filling/filling.py gs_simulation.py utils/decode_param.py
py -3.13 ../research/tests/verify_particle_filling_algorithms.py
py -3.13 ../research/tests/verify_particle_filling_runtime.py
py -3.13 ../research/tests/verify_particle_filling_optional.py
```

runtime 测试覆盖：

```text
DBSCAN_cluster
get_aabb
fill_particles_MCIS
fill_particles(method="mcis")
fill_particles(method="none")
fill_particles(method="legacy") 的小规模 Taichi CPU smoke test
手动 optional 下原始 Gaussian 顺序保持不变
method/methods 解析
decode_param.py 的 optional 默认值
全局预算不会超过 max_particles_num
none/mcis/legacy 混合使用
```

测试输出点云在：

```text
../research/tests/output/
```

里面有：

```text
mcis_runtime_direct.ply
mcis_runtime_wrapped.ply
legacy_runtime_smoke.ply
```

## 后续同学修改时注意

1. 不要把 `mpm_init_pos` 改成按 cluster 直接拼接原始点，否则后面渲染属性会错位。
2. 如果要接 VLM 的填充方式，保证 VLM 输出列表顺序和 `cls_pos` 一致。
3. 如果要调 DBSCAN，优先调 `DBSCAN_EPS`。
4. 如果 MCIS 生成点太少，先调 `mcis_interior_num` 和 `mcis_sample_num`。
5. 如果旧填充生成点明显不够，调大 `min_particles_num_per_object` 或 `max_particles_num`。
6. `legacy` 和 `mcis` 各有适用对象，不建议直接把所有场景都固定成一种。
