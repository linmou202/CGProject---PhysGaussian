import torch
import os
import numpy as np
import taichi as ti
import mcubes


# Defaults are for the normalized MPM space used by gs_simulation.py.
DBSCAN_EPS = 0.03
DBSCAN_MIN_SAMPLES = 8
DBSCAN_MIN_CLUSTER_SIZE = 16
DBSCAN_NOISE_ATTACH_RADIUS = 0.06
MCIS_SIGMA = 0.02
MCIS_EPS = 1e-6
MCIS_OCC_THRESHOLD = 0.6
MCIS_MAX_CANDIDATES = 200000
MCIS_MAX_SAMPLES = 50000
FILLING_METHOD_ALIASES = {
    "legacy": "legacy",
    "original": "legacy",
    "physgaussian": "legacy",
    "mcis": "mcis",
    "ipf": "mcis",
    "none": "none",
    "no_fill": "none",
    "disable": "none",
    "disabled": "none",
}

# 1. densify grids
# 2. identify grids whose density is larger than some threshold
# 3. filling grids with particles
# 4. identify and fill internal grids


@ti.func
def compute_density(index, pos, opacity, cov, grid_dx):
    gaussian_weight = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                node_pos = (index + ti.Vector([i, j, k])) * grid_dx
                dist = pos - node_pos
                gaussian_weight += ti.exp(-0.5 * dist.dot(cov @ dist))

    return opacity * gaussian_weight / 8.0


@ti.kernel
def densify_grids(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
):
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)
        cov = ti.Matrix(
            [
                [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
                [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
                [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
            ]
        )
        sig, Q = ti.sym_eig(cov)
        sig[0] = ti.max(sig[0], 1e-8)
        sig[1] = ti.max(sig[1], 1e-8)
        sig[2] = ti.max(sig[2], 1e-8)
        sig_mat = ti.Matrix(
            [[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]]
        )
        cov = Q @ sig_mat @ Q.transpose()
        r = 0.0
        for idx in ti.static(range(3)):
            if sig[idx] < 0:
                sig[idx] = ti.sqrt(-sig[idx])
            else:
                sig[idx] = ti.sqrt(sig[idx])

            r = ti.max(r, sig[idx])

        r = ti.ceil(r / grid_dx, dtype=int)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (
                        i + dx >= 0
                        and i + dx < grid_density.shape[0]
                        and j + dy >= 0
                        and j + dy < grid_density.shape[1]
                        and k + dz >= 0
                        and k + dz < grid_density.shape[2]
                    ):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos,
                            opacity[pi],
                            cov,
                            grid_dx,
                        )
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)


@ti.kernel
def fill_dense_grids(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    density_thres: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid_density:
        if grid_density[i, j, k] > density_thres:
            if grid[i, j, k] < max_particles_per_cell:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)

                if tmp_start_idx < new_particles.shape[0]:
                    end_idx = ti.min(tmp_start_idx + diff, new_particles.shape[0])
                    for index in range(tmp_start_idx, end_idx):
                        di = ti.random()
                        dj = ti.random()
                        dk = ti.random()
                        new_particles[index] = (
                            ti.Vector([i + di, j + dj, k + dk]) * grid_dx
                        )

    return new_start_idx


@ti.func
def collision_search(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> bool:
    dir = ti.Vector([0, 0, 0])
    if dir_type == 0:
        dir[0] = 1
    elif dir_type == 1:
        dir[0] = -1
    elif dir_type == 2:
        dir[1] = 1
    elif dir_type == 3:
        dir[1] = -1
    elif dir_type == 4:
        dir[2] = 1
    elif dir_type == 5:
        dir[2] = -1

    flag = False
    index += dir
    i, j, k = index
    while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
        if grid_density[index] > threshold:
            flag = True
            break
        index += dir
        i, j, k = index

    return flag


@ti.func
def collision_times(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> int:
    dir = ti.Vector([0, 0, 0])
    times = 0
    if dir_type > 5 or dir_type < 0:
        times = 1
    else:
        if dir_type == 0:
            dir[0] = 1
        elif dir_type == 1:
            dir[0] = -1
        elif dir_type == 2:
            dir[1] = 1
        elif dir_type == 3:
            dir[1] = -1
        elif dir_type == 4:
            dir[2] = 1
        elif dir_type == 5:
            dir[2] = -1

        state = grid[index] > 0
        index += dir
        i, j, k = index
        while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
            new_state = grid_density[index] > threshold
            if new_state != state and state == False:
                times += 1
            state = new_state
            index += dir
            i, j, k = index

    return times


@ti.kernel
def internal_filling(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid:
        if grid[i, j, k] == 0:
            collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    collision_hit = collision_hit and hit_test

            if collision_hit:
                hit_times = collision_times(
                    grid=grid,
                    grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir,
                    size=grid.shape[0],
                    threshold=threshold,
                )

                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    if tmp_start_idx < new_particles.shape[0]:
                        end_idx = ti.min(tmp_start_idx + diff, new_particles.shape[0])
                        for index in range(tmp_start_idx, end_idx):
                            di = ti.random()
                            dj = ti.random()
                            dk = ti.random()
                            new_particles[index] = (
                                ti.Vector([i + di, j + dj, k + dk]) * grid_dx
                            )

    return new_start_idx


@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


@ti.kernel
def compute_particle_volume(
    pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]


@ti.kernel
def assign_particle_to_grid(
    pos: ti.template(),
    grid: ti.template(),
    grid_dx: float,
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


def get_particle_volume(pos, grid_n: int, grid_dx: float, unifrom: bool = False):
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if unifrom:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()


def _to_numpy(array):
    if array is None:
        return None
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _index_array(array, indices, device=None):
    if array is None:
        return None
    if hasattr(array, "index_select"):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=array.device)
        return array.index_select(0, index_tensor)
    return array[indices]


def _tensor_from_numpy_like(array, like):
    array = np.asarray(array, dtype=np.float32)
    if hasattr(like, "device"):
        return torch.from_numpy(array).to(device=like.device, dtype=like.dtype)
    return array


def _try_build_tree(points):
    try:
        from scipy.spatial import cKDTree

        return cKDTree(points)
    except Exception:
        return None


def _query_ball(points, index, radius, tree=None):
    if tree is not None:
        return list(tree.query_ball_point(points[index], radius))
    diff = points - points[index]
    dist2 = np.einsum("ij,ij->i", diff, diff)
    return np.flatnonzero(dist2 <= radius * radius).tolist()


def _nearest_distances(query_points, surface_points):
    tree = _try_build_tree(surface_points)
    if tree is not None:
        distances, _ = tree.query(query_points, k=1)
        return distances

    chunk_size = max(1, min(1024, 1000000 // max(1, surface_points.shape[0])))
    distances = []
    for start in range(0, query_points.shape[0], chunk_size):
        chunk = query_points[start : start + chunk_size]
        diff = chunk[:, None, :] - surface_points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        distances.append(np.sqrt(np.min(dist2, axis=1)))
    return np.concatenate(distances, axis=0)


def _dbscan_labels(points, eps, min_samples):
    point_num = points.shape[0]
    labels = np.full(point_num, -1, dtype=np.int64)
    visited = np.zeros(point_num, dtype=bool)
    tree = _try_build_tree(points)
    cluster_id = 0

    for point_id in range(point_num):
        if visited[point_id]:
            continue

        visited[point_id] = True
        neighbors = _query_ball(points, point_id, eps, tree)
        if len(neighbors) < min_samples:
            continue

        labels[point_id] = cluster_id
        seeds = list(neighbors)
        seed_set = set(seeds)
        cursor = 0

        while cursor < len(seeds):
            neighbor_id = seeds[cursor]
            if not visited[neighbor_id]:
                visited[neighbor_id] = True
                neighbor_neighbors = _query_ball(points, neighbor_id, eps, tree)
                if len(neighbor_neighbors) >= min_samples:
                    for expanded_id in neighbor_neighbors:
                        if expanded_id not in seed_set:
                            seeds.append(expanded_id)
                            seed_set.add(expanded_id)

            if labels[neighbor_id] == -1:
                labels[neighbor_id] = cluster_id
            cursor += 1

        cluster_id += 1

    return labels


def _labels_to_cluster_indices(
    labels,
    points,
    min_cluster_size=DBSCAN_MIN_CLUSTER_SIZE,
    attach_radius=DBSCAN_NOISE_ATTACH_RADIUS,
):
    clusters = []
    noise_indices = []

    for label in sorted(label for label in np.unique(labels) if label >= 0):
        indices = np.flatnonzero(labels == label).tolist()
        if len(indices) >= min_cluster_size:
            clusters.append(indices)
        else:
            noise_indices.extend(indices)

    noise_indices.extend(np.flatnonzero(labels < 0).tolist())

    if not clusters:
        if points.shape[0] == 0:
            return []
        return [np.arange(points.shape[0], dtype=np.int64)]

    centers = np.asarray([points[indices].mean(axis=0) for indices in clusters])
    detached_noise = []
    for point_id in noise_indices:
        distances = np.linalg.norm(centers - points[point_id], axis=1)
        closest = int(np.argmin(distances))
        if distances[closest] <= attach_radius:
            clusters[closest].append(point_id)
        else:
            detached_noise.append(point_id)

    if detached_noise:
        clusters.append(detached_noise)

    order = sorted(
        range(len(clusters)),
        key=lambda cluster_idx: (
            float(points[clusters[cluster_idx]].mean(axis=0)[0]),
            float(points[clusters[cluster_idx]].mean(axis=0)[1]),
            float(points[clusters[cluster_idx]].mean(axis=0)[2]),
            -len(clusters[cluster_idx]),
        ),
    )
    return [np.asarray(sorted(clusters[idx]), dtype=np.int64) for idx in order]


def _cov_upper_to_mats(cov_upper):
    if cov_upper is None or cov_upper.size == 0:
        return None
    cov_upper = np.asarray(cov_upper).reshape(-1, 6)
    mats = np.zeros((cov_upper.shape[0], 3, 3), dtype=np.float64)
    mats[:, 0, 0] = cov_upper[:, 0]
    mats[:, 0, 1] = cov_upper[:, 1]
    mats[:, 1, 0] = cov_upper[:, 1]
    mats[:, 0, 2] = cov_upper[:, 2]
    mats[:, 2, 0] = cov_upper[:, 2]
    mats[:, 1, 1] = cov_upper[:, 3]
    mats[:, 1, 2] = cov_upper[:, 4]
    mats[:, 2, 1] = cov_upper[:, 4]
    mats[:, 2, 2] = cov_upper[:, 5]
    return mats


def _estimate_cov_padding(cov_upper, fallback=0.01):
    mats = _cov_upper_to_mats(cov_upper)
    if mats is None:
        return fallback
    try:
        eigvals = np.linalg.eigvalsh(mats)
        max_eigvals = np.maximum(np.max(eigvals, axis=1), 0.0)
        radii = np.sqrt(max_eigvals)
        radii = radii[np.isfinite(radii) & (radii > 0)]
        if radii.size == 0:
            return fallback
        return float(np.median(radii))
    except Exception:
        return fallback


def _build_convex_hull(points):
    if points.shape[0] < 4:
        return None
    try:
        from scipy.spatial import ConvexHull, QhullError

        try:
            return ConvexHull(points)
        except QhullError:
            return None
    except Exception:
        return None


def _compute_aabb(points, cov=None, min_padding=0.01):
    if points.shape[0] == 0:
        zeros = np.zeros(3, dtype=np.float64)
        return zeros, zeros, None

    hull = _build_convex_hull(points)
    if hull is not None:
        hull_points = points[hull.vertices]
    else:
        hull_points = points

    padding = max(_estimate_cov_padding(cov, fallback=min_padding), min_padding)
    coords_ldb = np.min(hull_points, axis=0) - padding
    coords_ruf = np.max(hull_points, axis=0) + padding
    return coords_ldb, coords_ruf, hull


def _sample_uniform_aabb(coords_ldb, coords_ruf, sample_num, grid_dx, rng):
    coords_ldb = np.asarray(coords_ldb, dtype=np.float64)
    coords_ruf = np.asarray(coords_ruf, dtype=np.float64)
    center = 0.5 * (coords_ldb + coords_ruf)
    extent = np.maximum(coords_ruf - coords_ldb, max(2.0 * grid_dx, 1e-4))
    coords_ldb = center - 0.5 * extent
    coords_ruf = center + 0.5 * extent
    return coords_ldb + rng.random((sample_num, 3)) * (coords_ruf - coords_ldb)


def _inside_convex_hull(points, hull, coords_ldb, coords_ruf, tolerance=1e-6):
    if hull is None:
        return np.all(
            (points >= coords_ldb.reshape(1, 3) - tolerance)
            & (points <= coords_ruf.reshape(1, 3) + tolerance),
            axis=1,
        )

    equations = hull.equations
    normal = equations[:, :3]
    offset = equations[:, 3]
    mask = np.zeros(points.shape[0], dtype=bool)
    chunk_size = max(1, min(8192, 2000000 // max(1, normal.shape[0])))
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        max_side = np.max(chunk @ normal.T + offset.reshape(1, -1), axis=1)
        mask[start : start + chunk.shape[0]] = max_side <= tolerance
    return mask


def _importance_sample(points, distances, sample_num, sigma, grid_dx, rng):
    if points.shape[0] == 0 or sample_num <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    sigma = max(float(sigma), 1e-6)
    weights = np.exp(-(distances * distances) / (2.0 * sigma * sigma))
    weights = np.maximum(weights, MCIS_EPS)
    probabilities = weights / np.sum(weights)
    replace = sample_num > points.shape[0]
    indices = rng.choice(points.shape[0], size=sample_num, replace=replace, p=probabilities)
    sampled = points[indices].copy()

    if replace:
        jitter = rng.uniform(-0.25 * grid_dx, 0.25 * grid_dx, size=sampled.shape)
        sampled += jitter

    return sampled.astype(np.float32)


def normalize_filling_method(method):
    method = (method or "legacy").lower()
    if method not in FILLING_METHOD_ALIASES:
        raise ValueError(f"Unknown particle filling method: {method}")
    return FILLING_METHOD_ALIASES[method]


def resolve_filling_methods(methods, default_method, object_count):
    """Resolve one global method or a per-object method list."""
    default_method = normalize_filling_method(default_method)
    if methods is None:
        resolved = [default_method] * object_count
    elif isinstance(methods, str):
        resolved = [normalize_filling_method(methods)] * object_count
    else:
        resolved = [normalize_filling_method(method) for method in list(methods)]
        if len(resolved) < object_count:
            resolved += [default_method] * (object_count - len(resolved))
        elif len(resolved) > object_count:
            resolved = resolved[:object_count]

    return resolved


def allocate_filling_budgets(cluster_sizes, total_budget, min_budget_per_object=8192):
    """Allocate a global new-particle budget across DBSCAN clusters."""
    cluster_sizes = [max(0, int(size)) for size in cluster_sizes]
    object_count = len(cluster_sizes)
    total_budget = max(0, int(total_budget))
    min_budget_per_object = max(0, int(min_budget_per_object))

    if object_count == 0 or total_budget == 0:
        return [0] * object_count

    total_cluster_size = max(sum(cluster_sizes), 1)
    base_budget = 0
    if min_budget_per_object > 0 and total_budget >= min_budget_per_object * object_count:
        base_budget = min_budget_per_object

    remaining_budget = total_budget - base_budget * object_count
    raw_extra = [
        remaining_budget * cluster_size / total_cluster_size
        for cluster_size in cluster_sizes
    ]
    budgets = [base_budget + int(extra) for extra in raw_extra]
    spare_budget = total_budget - sum(budgets)
    if spare_budget > 0:
        order = sorted(
            range(object_count),
            key=lambda idx: raw_extra[idx] - int(raw_extra[idx]),
            reverse=True,
        )
        for idx in order[:spare_budget]:
            budgets[idx] += 1

    return budgets


def DBSCAN_cluster(
    pos,
    opacity,
    cov,
    screen_points,
    shs,
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
    min_cluster_size=DBSCAN_MIN_CLUSTER_SIZE,
    noise_attach_radius=DBSCAN_NOISE_ATTACH_RADIUS,
):
    """Cluster Gaussian kernels into object-level subsets."""
    pos_np = _to_numpy(pos).reshape(-1, 3).astype(np.float64)
    if pos_np.shape[0] == 0:
        cls_size = torch.zeros(0, dtype=torch.long, device=pos.device)
        return [], [], [], [], [], cls_size

    labels = _dbscan_labels(pos_np, eps=eps, min_samples=min_samples)
    cluster_indices = _labels_to_cluster_indices(
        labels,
        pos_np,
        min_cluster_size=min_cluster_size,
        attach_radius=noise_attach_radius,
    )

    cls_pos = [_index_array(pos, indices) for indices in cluster_indices]
    cls_opacity = [_index_array(opacity, indices) for indices in cluster_indices]
    cls_cov = [_index_array(cov, indices) for indices in cluster_indices]
    cls_screen_points = [_index_array(screen_points, indices) for indices in cluster_indices]
    cls_shs = [_index_array(shs, indices) for indices in cluster_indices]
    sizes = [indices.shape[0] for indices in cluster_indices]
    cls_size = torch.tensor(sizes, dtype=torch.long, device=pos.device)
    return cls_pos, cls_opacity, cls_cov, cls_screen_points, cls_shs, cls_size

def get_aabb(
    pos,
    opacity,
    cov,
    screen_points,
    shs
):
    """Return lower and upper AABB corners for one Gaussian cluster."""
    pos_np = _to_numpy(pos).reshape(-1, 3).astype(np.float64)
    cov_np = _to_numpy(cov)
    coords_ldb, coords_ruf, _ = _compute_aabb(pos_np, cov_np)
    coords_ldb = _tensor_from_numpy_like(coords_ldb, pos)
    coords_ruf = _tensor_from_numpy_like(coords_ruf, pos)
    return coords_ldb, coords_ruf


def fill_particles_MCIS(
    pos,
    opacity,
    cov,
    grid_n: int,
    interior_num: int,
    sample_num: int,
    grid_dx: float,
    search_thres=1.0,
    boundary: list = None,
    sigma: float = MCIS_SIGMA,
    occ_threshold: float = MCIS_OCC_THRESHOLD,
    random_seed=None,
):
    """FastPhysGS-style MCIS particle filling for one object cluster."""
    pos_clone = pos.clone()
    active_pos = pos
    active_cov = cov

    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool, device=pos_clone.device)
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
        active_pos = active_pos[mask]
        active_cov = active_cov[mask]

    if active_pos.shape[0] < 4 or interior_num <= 0 or sample_num <= 0:
        return pos_clone

    interior_num = int(min(interior_num, MCIS_MAX_CANDIDATES))
    sample_num = int(min(sample_num, MCIS_MAX_SAMPLES))
    rng = np.random.default_rng(random_seed)

    pos_np = _to_numpy(active_pos).reshape(-1, 3).astype(np.float64)
    cov_np = _to_numpy(active_cov)
    coords_ldb, coords_ruf, hull = _compute_aabb(pos_np, cov_np, min_padding=max(grid_dx, 0.01))
    candidate_points = _sample_uniform_aabb(
        coords_ldb,
        coords_ruf,
        interior_num,
        grid_dx,
        rng,
    )

    inside_mask = _inside_convex_hull(candidate_points, hull, coords_ldb, coords_ruf)
    # FastPhysGS uses occ(q; H_k) > 0.6. For the convex-hull half-space test,
    # inside points have occupancy 1 and outside points have occupancy 0.
    if occ_threshold > 1.0:
        inside_mask = np.zeros_like(inside_mask, dtype=bool)

    inside_points = candidate_points[inside_mask]
    if inside_points.shape[0] == 0:
        print("MCIS warning: no interior candidates, returning original particles.")
        return pos_clone

    distances = _nearest_distances(inside_points, pos_np)
    sampled_points = _importance_sample(
        inside_points,
        distances,
        sample_num,
        sigma=sigma,
        grid_dx=grid_dx,
        rng=rng,
    )

    sampled_tensor = _tensor_from_numpy_like(sampled_points, pos_clone)
    particles_tensor = torch.cat([pos_clone, sampled_tensor], dim=0)
    return particles_tensor

def fill_particles(
    pos,
    opacity,
    cov,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres=2.0,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=5,
    ray_cast_dir=4,
    boundary: list = None,
    smooth: bool = False,
    method: str = "legacy",
    interior_num: int = None,
    sample_num: int = None,
    mcis_sigma: float = MCIS_SIGMA,
):
    method = (method or "legacy").lower()
    normalized_method = normalize_filling_method(method)
    if normalized_method == "mcis":
        particle_budget = max(0, int(max_samples))
        if sample_num is None:
            sample_num = min(particle_budget, max(int(pos.shape[0]) * 2, 1024))
        else:
            sample_num = min(int(sample_num), particle_budget)
        if interior_num is None:
            interior_num = min(
                MCIS_MAX_CANDIDATES,
                max(int(sample_num) * 10, int(pos.shape[0]) * 20),
            )
        return fill_particles_MCIS(
            pos=pos,
            opacity=opacity,
            cov=cov,
            grid_n=grid_n,
            interior_num=interior_num,
            sample_num=sample_num,
            grid_dx=grid_dx,
            search_thres=search_thres,
            boundary=boundary,
            sigma=mcis_sigma,
        )

    if normalized_method == "none":
        return pos.clone()

    pos_clone = pos.clone()
    max_samples = max(0, int(max_samples))
    if max_samples == 0:
        return pos_clone

    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool, device=pos_clone.device)
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        if pos.shape[0] == 0:
            return pos_clone

        grid_dx = max_diff / grid_n
        new_origin = torch.tensor(
            [boundary[0], boundary[2], boundary[4]],
            device=pos_clone.device,
            dtype=pos_clone.dtype,
        )
        pos = pos - new_origin

    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
    particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
    fill_num = 0

    # compute density_field
    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)

    # fill dense grids
    fill_num = fill_dense_grids(
        grid,
        grid_density,
        grid_dx,
        density_thres,
        particles,
        0,
        max_particles_per_cell,
    )
    fill_num = min(fill_num, max_samples)
    print("after dense grids: ", fill_num)

    # smooth density_field
    if smooth:
        df = grid_density.to_numpy()
        smoothed_df = mcubes.smooth(df, method="constrained", max_iters=500).astype(
            np.float32
        )
        grid_density.from_numpy(smoothed_df)
        print("smooth finished")

    # fill internal grids
    fill_num = internal_filling(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        exclude_dir=search_exclude_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        ray_cast_dir=ray_cast_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        threshold=search_thres,
    )
    fill_num = min(fill_num, max_samples)
    print("after internal grids: ", fill_num)

    # put new particles together with original particles
    particles_tensor = particles.to_torch()[:fill_num].to(
        device=pos_clone.device,
        dtype=pos_clone.dtype,
    )
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
    particles_tensor = torch.cat([pos_clone, particles_tensor], dim=0)

    return particles_tensor


@ti.kernel
def get_attr_from_closest(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_cov: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_cov: ti.template(),
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_cov[pi] = ti_cov[min_idx]


def init_filled_particles(pos, shs, cov, opacity, new_pos):
    shs = shs.reshape(pos.shape[0], -1)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=shs.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_cov.from_torch(cov.reshape(-1, 6))
    ti_shs.from_torch(shs)
    ti_opacity.from_torch(opacity.reshape(-1))

    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).cuda()
    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_cov = ti.Vector.field(n=6, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))
    ti_new_shs.from_torch(new_shs)

    get_attr_from_closest(
        ti_pos,
        ti_shs,
        ti_opacity,
        ti_cov,
        ti_new_pos,
        ti_new_shs,
        ti_new_opacity,
        ti_new_cov,
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    cov_tensor = ti_new_cov.to_torch().cuda()

    shs_tensor = torch.cat([shs, shs_tensor], dim=0)
    shs_tensor = shs_tensor.view(shs_tensor.shape[0], -1, 3)
    opacity_tensor = torch.cat([opacity, opacity_tensor.reshape(-1, 1)], dim=0)
    cov_tensor = torch.cat([cov, cov_tensor], dim=0)
    return shs_tensor, opacity_tensor, cov_tensor
