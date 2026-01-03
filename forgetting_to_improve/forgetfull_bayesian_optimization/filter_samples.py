from botorch import fit_gpytorch_mll
from ..forgetting_to_improve.optimize import sequentially_optimize_samples, batch_optimize_samples

def filter_samples(
        model,
        mll,
        train_x,
        train_obj,
        target_region,
        algorithm='joint',
        method='sequential',
        min_samples=5,
        initialize_model=None,
        global_bounds=None,
    ):
    if algorithm == 'joint' or algorithm == 'epistemic':
        if method == 'sequential':
            x_samples, y_samples, deleted_x_samples = train_x, train_obj, []
            for _ in range(train_x.shape[0] - min_samples):
                mll, model = initialize_model(x_samples, y_samples, global_bounds)
                try:
                    fit_gpytorch_mll(mll)
                except Exception as e:
                    print(f"Error fitting model at iteration {e}")
                if target_region.samples.shape[0] > 5:
                   target_region.update(model)
                x_samples, y_samples, deleted_x_samples = sequentially_optimize_samples(
                    model=model,
                    likelihood=mll.likelihood,
                    x_samples=x_samples,
                    y_samples=y_samples.squeeze(-1),
                    x_test=target_region.samples,
                    max_iter=1,
                    noise_type=algorithm,
                    show_hessian=False
                )
                y_samples = y_samples.unsqueeze(-1)
        elif method == 'batch':
            x_samples, y_samples, deleted_x_samples = batch_optimize_samples(
                model=model,
                likelihood=mll.likelihood,
                x_samples=train_x,
                y_samples=train_obj.squeeze(-1),
                x_test=target_region.samples,
                max_iter=train_x.shape[0]-min_samples,
                batch_size=5,
                noise_type=algorithm,
            )
            y_samples = y_samples.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown method: {method}")
        return x_samples, y_samples, deleted_x_samples
    else:
        return train_x, train_obj, []