import time
import torch
from botorch.acquisition import UpperConfidenceBound # type: ignore
from botorch.utils.sampling import draw_sobol_samples # type: ignore

class TargetRegion:
    """Class representing the target region for Bayesian optimization."""
    def __init__(self, global_bounds: torch.Tensor, num_initial_points: int = 1000, seed: int = 0, iteration: int = 0):
        self.global_bounds = global_bounds
        self.num_initial_points = num_initial_points
        if global_bounds.shape[1] > 0:
            deterministic_seed = seed * 10 + iteration
            self.samples = draw_sobol_samples(
                bounds=global_bounds, n=num_initial_points, q=1, seed=deterministic_seed
            ).squeeze(1).to(dtype=torch.float64)
            # Expected distance between samples based on volume and density
            # Volume of the hypercube
            domain_volume = torch.prod(global_bounds[1] - global_bounds[0])
            # Expected distance: (volume / n_samples) ^ (1/d) scaled by dimensionality
            d = global_bounds.shape[1]
            self.min_sample_distance = (domain_volume / num_initial_points) ** (1.0 / d) * 2.0
            self.initial_samples = self.samples.clone()
        else:
            samples_per_dim = int(num_initial_points ** (1 / global_bounds.shape[1]))
            linspaces = [
                torch.linspace(global_bounds[0, d], global_bounds[1, d], samples_per_dim)
                for d in range(global_bounds.shape[1])
            ]
            # Use a larger clustering distance - multiple grid cells
            grid_spacing = linspaces[0][1] - linspaces[0][0]
            self.min_sample_distance = grid_spacing * 2.0  # Cluster 2x2 grid cells together
            mesh = torch.meshgrid(*linspaces, indexing='ij')
            self.samples = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
            self.intital_samples = self.samples.clone()

    def update(self, model):
        with torch.no_grad():
            ucb_func = UpperConfidenceBound(model=model, beta=2.5673)
            ucb = ucb_func(self.samples.reshape(self.samples.shape[0], 1, self.samples.shape[1])).squeeze()
            mean = model.posterior(self.samples).mean.squeeze()
            lcb = 2 * mean - ucb
        max_lcb = lcb.max().item()
        larger = torch.where(ucb >= max_lcb, True, False)
        self.samples = self.samples[larger]

    def get_bounds(self):
        """Get the bounding boxes of the target region clusters."""
        t0 = time.time()
        
        # Build connectivity graph using distance threshold
        n_points = self.samples.shape[0]
        visited = torch.zeros(n_points, dtype=torch.bool)
        clusters = []
        
        # For each unvisited point, find all connected points via BFS/DFS
        for i in range(n_points):
            if visited[i]:
                continue
            
            # Start a new cluster with BFS
            cluster_indices = [i]
            visited[i] = True
            queue = [i]
            
            while queue:
                current_idx = queue.pop(0)
                current_point = self.samples[current_idx]
                
                # Find all unvisited neighbors within distance threshold
                distances = torch.norm(self.samples - current_point, dim=1)
                neighbors = torch.where((distances <= self.min_sample_distance) & ~visited)[0]
                
                for neighbor_idx in neighbors:
                    visited[neighbor_idx] = True
                    cluster_indices.append(neighbor_idx.item())
                    queue.append(neighbor_idx.item())
            
            # Store cluster points
            cluster_points = self.samples[cluster_indices]
            clusters.append(cluster_points)
        
        # Compute bounding boxes for each cluster
        all_bounds = []
        for cluster in clusters:
            min_bounds = torch.min(cluster, dim=0).values
            max_bounds = torch.max(cluster, dim=0).values
            all_bounds.append((min_bounds, max_bounds))

        print(f"Target region clustering took {time.time() - t0:.2f} seconds for {len(clusters)} clusters.")
        if all_bounds == []:
            # If no clusters found, return the global bounds
            all_bounds.append((self.global_bounds[0], self.global_bounds[1]))
        return all_bounds
