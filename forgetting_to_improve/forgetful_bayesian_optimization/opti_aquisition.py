import torch
from botorch.optim import optimize_acqf # type: ignore

def get_optimize_acqf_and_get_observation(num_restarts, raw_samples, obj, obs_noise):
    """Returns a function that optimizes the acquisition function and gets a new observation."""
    def optimize_acqf_and_get_observation(acq_func, bounds):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        try:
            candidates, acquisition_values = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,  # used for intialization heuristic
                options={"batch_limit": 10, "maxiter": 500},
            )
        except Exception as _:
            new_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(200, bounds.shape[1])
            acquisition_values = acq_func(new_x).unsqueeze(-1)
            max_index = torch.argmax(acquisition_values)
            new_x = new_x[max_index].unsqueeze(0)
            acquisition_values = acquisition_values[max_index].unsqueeze(0)
            return new_x, acquisition_values
        new_x = candidates.detach()
        acqu_x = acquisition_values.detach()
        return new_x, acqu_x
    
    return optimize_acqf_and_get_observation