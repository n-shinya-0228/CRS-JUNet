import numpy as np


def _make_bev_projection(points, remissions, point_labels):
    height = 512
    width = 512
    max_radius = 51.2
    resolution = max_radius / height

    radius = np.hypot(points[:, 0], points[:, 1])
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    visible = (radius > 0.0) & (radius < max_radius)
    point_indices = np.flatnonzero(visible)
    row = np.floor(radius[point_indices] / resolution).astype(np.int32)
    col = np.floor((azimuth[point_indices] + np.pi) / (2.0 * np.pi / width)).astype(np.int32)
    col = np.clip(col, 0, width - 1)
    cell_indices = row * width + col
    cell_count = height * width

    counts = np.bincount(cell_indices, minlength=cell_count)
    max_z = np.full(cell_count, -np.inf, dtype=np.float32)
    min_z = np.full(cell_count, np.inf, dtype=np.float32)
    max_remission = np.full(cell_count, -np.inf, dtype=np.float32)
    max_x = np.full(cell_count, -np.inf, dtype=np.float32)
    min_x = np.full(cell_count, np.inf, dtype=np.float32)
    max_y = np.full(cell_count, -np.inf, dtype=np.float32)
    min_y = np.full(cell_count, np.inf, dtype=np.float32)
    np.maximum.at(max_z, cell_indices, points[point_indices, 2])
    np.minimum.at(min_z, cell_indices, points[point_indices, 2])
    np.maximum.at(max_remission, cell_indices, remissions[point_indices])
    np.maximum.at(max_x, cell_indices, points[point_indices, 0])
    np.minimum.at(min_x, cell_indices, points[point_indices, 0])
    np.maximum.at(max_y, cell_indices, points[point_indices, 1])
    np.minimum.at(min_y, cell_indices, points[point_indices, 1])
    mean_z = np.bincount(
        cell_indices, weights=points[point_indices, 2], minlength=cell_count
    ) / np.maximum(counts, 1)

    occupied = counts > 0
    features = np.zeros((7, cell_count), dtype=np.float32)
    features[0, occupied] = max_z[occupied]
    features[1, occupied] = mean_z[occupied]
    features[2, occupied] = max_remission[occupied]
    features[3, occupied] = counts[occupied] / 100.0
    features[4, occupied] = max_z[occupied] - min_z[occupied]
    features[5, occupied] = max_x[occupied] - min_x[occupied]
    features[6, occupied] = max_y[occupied] - min_y[occupied]

    representative = np.full(cell_count, -1, dtype=np.int64)
    order = np.lexsort((points[point_indices, 2], cell_indices))
    sorted_cells = cell_indices[order]
    last = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        last[:-1] = sorted_cells[:-1] != sorted_cells[1:]
    representative[sorted_cells[last]] = point_indices[order[last]]
    labels = np.zeros(cell_count, dtype=np.int64)
    labels[occupied] = point_labels[representative[occupied]]
    mask = occupied.astype(np.float32)
    return (
        _normalize_bev(features.reshape(7, height, width)),
        mask.reshape(height, width),
        labels.reshape(height, width),
        representative.reshape(height, width),
    )


def _normalize_bev(features):
    features = features.copy().astype(np.float32)
    features[0] = (np.clip(features[0], -5.0, 15.0) - 1.0) / 5.0
    features[1] = (np.clip(features[1], -5.0, 15.0) - 1.0) / 5.0
    max_remission = features[2].max()
    if max_remission > 0.0:
        features[2] /= max_remission
    features[2] = np.clip(features[2], 0.0, 1.0)
    features[3] = np.clip(features[3], 0.0, 5.0) / 5.0
    features[4] = np.clip(features[4], 0.0, 10.0) / 10.0
    features[5] = np.clip(features[5], 0.0, 10.0) / 10.0
    features[6] = np.clip(features[6], 0.0, 10.0) / 10.0
    return features


def _make_spherical_projection(points, remissions, point_labels, height, width,
                               fov_up, fov_down, means, stds):
    depth = np.linalg.norm(points, axis=1)
    pitch = np.degrees(np.arcsin(np.clip(points[:, 2] / np.maximum(depth, 1e-6), -1.0, 1.0)))
    yaw = np.arctan2(points[:, 1], points[:, 0])

    row = np.floor((fov_up - pitch) / (fov_up - fov_down) * height).astype(np.int32)
    col = np.floor((yaw + np.pi) / (2.0 * np.pi) * width).astype(np.int32)
    visible = (depth > 0.0) & (pitch <= fov_up) & (pitch >= fov_down)
    visible &= (row >= 0) & (row < height)
    row = np.clip(row, 0, height - 1)
    col = np.mod(col, width)

    features = np.zeros((6, height, width), dtype=np.float32)
    labels = np.zeros((height, width), dtype=np.int64)
    point_index = np.full((height, width), -1, dtype=np.int64)
    mask = np.zeros((height, width), dtype=np.float32)

    visible_indices = np.flatnonzero(visible)
    linear_index = row[visible_indices] * width + col[visible_indices]
    order = np.lexsort((-depth[visible_indices], linear_index))
    sorted_indices = visible_indices[order]
    sorted_pixels = linear_index[order]
    nearest = np.ones(sorted_pixels.shape, dtype=bool)
    if sorted_pixels.size > 1:
        nearest[:-1] = sorted_pixels[:-1] != sorted_pixels[1:]
    selected = sorted_indices[nearest]
    selected_y = row[selected]
    selected_x = col[selected]
    features[:5, selected_y, selected_x] = np.stack(
        (
            depth[selected],
            points[selected, 0],
            points[selected, 1],
            points[selected, 2],
            remissions[selected],
        ),
        axis=0,
    )
    labels[selected_y, selected_x] = point_labels[selected]
    point_index[selected_y, selected_x] = selected
    mask[selected_y, selected_x] = 1.0

    means = np.asarray(means, dtype=np.float32).reshape(5, 1, 1)
    stds = np.maximum(np.asarray(stds, dtype=np.float32).reshape(5, 1, 1), 1e-6)
    features[:5] = (features[:5] - means) / stds
    features[:5] *= mask[None, ...]
    features[5] = mask
    return features, labels, mask, point_index, row, col, visible


def build_dual_projection(points, remissions, point_labels, sphere_config):
    """Build synchronized polar-BEV and spherical projections from one point order."""
    points = np.asarray(points, dtype=np.float32)
    remissions = np.asarray(remissions, dtype=np.float32).reshape(-1)
    point_labels = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not (len(points) == len(remissions) == len(point_labels)):
        raise ValueError("points, remissions, and point_labels must have equal length")

    bev_features, bev_mask, bev_labels, bev_point_index = _make_bev_projection(
        points, remissions, point_labels
    )

    sphere_features, sphere_labels, sphere_mask, sphere_point_index, sphere_row, sphere_col, sphere_visible = (
        _make_spherical_projection(
            points,
            remissions,
            point_labels,
            height=int(sphere_config["height"]),
            width=int(sphere_config["width"]),
            fov_up=float(sphere_config["fov_up"]),
            fov_down=float(sphere_config["fov_down"]),
            means=sphere_config["means"],
            stds=sphere_config["stds"],
        )
    )

    bev_to_sphere = np.full((*bev_point_index.shape, 2), -2.0, dtype=np.float32)
    bev_to_sphere_valid = np.zeros(bev_point_index.shape, dtype=np.float32)
    bev_valid = bev_point_index >= 0
    bev_y, bev_x = np.where(bev_valid)
    representative_indices = bev_point_index[bev_y, bev_x]
    sphere_y = sphere_row[representative_indices]
    sphere_x = sphere_col[representative_indices]
    matches_visible_point = sphere_visible[representative_indices]
    matches_sphere_pixel = np.zeros_like(matches_visible_point)
    matches_sphere_pixel[matches_visible_point] = (
        sphere_point_index[
            sphere_y[matches_visible_point], sphere_x[matches_visible_point]
        ] == representative_indices[matches_visible_point]
    )
    matched = matches_visible_point & matches_sphere_pixel
    bev_y, bev_x = bev_y[matched], bev_x[matched]
    sphere_y, sphere_x = sphere_y[matched], sphere_x[matched]

    bev_to_sphere[bev_y, bev_x, 0] = 2.0 * sphere_x / (sphere_features.shape[2] - 1) - 1.0
    bev_to_sphere[bev_y, bev_x, 1] = 2.0 * sphere_y / (sphere_features.shape[1] - 1) - 1.0
    bev_to_sphere_valid[bev_y, bev_x] = 1.0

    return {
        "bev": bev_features,
        "bev_mask": bev_mask,
        "bev_labels": bev_labels,
        "sphere": sphere_features,
        "sphere_mask": sphere_mask,
        "sphere_labels": sphere_labels,
        "bev_to_sphere": bev_to_sphere,
        "bev_to_sphere_valid": bev_to_sphere_valid,
    }