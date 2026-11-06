"""Feature-space metrics for the DINOv2 evaluation stage."""

from collections import defaultdict

import torch
import torch.nn.functional as F


def group_indices_by_prompt(image_files, prompt_dict):
    """Group by exact prompt text, independently of filenames and GPU shards."""
    groups = defaultdict(list)
    for index, filename in enumerate(image_files):
        if filename in prompt_dict:
            groups[prompt_dict[filename]].append(index)
    return dict(groups)


def mean_pairwise_cosine_distance(features):
    """Mean 1 - cosine over unordered pairs, using O(ND) sufficient statistics."""
    count = len(features)
    if count < 2:
        raise ValueError("DINO Feature Diversity requires at least two images per prompt")
    features = features.to(torch.float64)
    if not torch.isfinite(features).all() or (features.norm(dim=1) == 0).any():
        raise ValueError("DINO features must be finite and have nonzero norms")
    normalized = F.normalize(features, dim=1)
    # ||sum_i f_i||^2 = N + 2 * sum_{i<j} cosine(f_i, f_j).
    similarity = (normalized.sum(dim=0).square().sum() - count) / (count * (count - 1))
    return (1 - similarity).clamp(0, 2).item()


def _squared_distances(rows, columns):
    distances = rows.square().sum(dim=1, keepdim=True) + columns.square().sum(dim=1).unsqueeze(0)
    return distances.addmm_(rows, columns.T, beta=1, alpha=-2).clamp_min_(0)


def knn_squared_radii(features, query_indices, k=3, chunk_size=1024):
    """Student manifold radii; exclude only each point itself, keeping duplicates."""
    if k < 1 or len(features) <= k:
        raise ValueError(f"Recall requires 1 <= k < student count; got k={k}, count={len(features)}")
    if chunk_size < 1:
        raise ValueError("Distance chunk size must be positive")
    radii = features.new_empty(len(query_indices))
    for start in range(0, len(query_indices), chunk_size):
        indices = query_indices[start : start + chunk_size]
        rows = features[indices]
        nearest = features.new_full((len(rows), k), float("inf"))
        for column_start in range(0, len(features), chunk_size):
            column_end = min(column_start + chunk_size, len(features))
            distances = _squared_distances(rows, features[column_start:column_end])
            columns = torch.arange(column_start, column_end, device=features.device)
            distances.masked_fill_(indices[:, None] == columns[None, :], float("inf"))
            nearest = torch.cat((nearest, distances), dim=1).topk(k, dim=1, largest=False).values
        radii[start : start + len(rows)] = nearest.max(dim=1).values
    return radii


def manifold_coverage(query_features, reference_features, squared_radii, chunk_size=1024):
    """Whether each teacher query lies inside any student k-NN ball (recall)."""
    if chunk_size < 1:
        raise ValueError("Distance chunk size must be positive")
    if len(reference_features) != len(squared_radii):
        raise ValueError("Every student feature must have one manifold radius")
    covered = torch.zeros(len(query_features), dtype=torch.bool, device=query_features.device)
    for start in range(0, len(query_features), chunk_size):
        rows = query_features[start : start + chunk_size]
        batch_covered = covered[start : start + len(rows)]
        for column_start in range(0, len(reference_features), chunk_size):
            column_end = min(column_start + chunk_size, len(reference_features))
            distances = _squared_distances(rows, reference_features[column_start:column_end])
            batch_covered |= (distances <= squared_radii[column_start:column_end]).any(dim=1)
            if batch_covered.all():
                break
    return covered
