
import numpy as np
import torch

def relative_l2_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.sum((x-y)**2, axis = (1,2,3))
    temp = np.sum((x)**2, axis = (1,2,3))
    return np.sqrt(mse/temp), mse, temp, np.sqrt(np.sum(mse)/np.sum(temp))

def relative_rmse_error_ornl(x, y, axis=None):
    """Compute the relative RMSE between two arrays or tensors, returns a list of floats per batch."""
    try:
        assert x.shape == y.shape
    except:
        print("x.shape != y.shape", x.shape, y.shape)
    
    if isinstance(x, torch.Tensor):
        if axis is None:
            mse  = torch.mean((x - y) ** 2)
            maxv = torch.amax(x)
            minv = torch.amin(x)
        else:
            mse  = torch.mean((x - y) ** 2, dim=axis, keepdim=True)
            maxv = torch.amax(x, dim=axis, keepdim=True)
            minv = torch.amin(x, dim=axis, keepdim=True)
        
        error = torch.sqrt(mse) / (maxv - minv + 1e-8)
        error = error.view(-1).cpu().tolist()

    elif isinstance(x, np.ndarray):
        mse  = np.mean((x - y) ** 2, axis=axis, keepdims=True)
        maxv = np.amax(x, axis=axis, keepdims=True)
        minv = np.amin(x, axis=axis, keepdims=True)
        error = np.sqrt(mse) / (maxv - minv + 1e-8)
        error = error.reshape(-1).tolist()

    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    return error[0] if axis is None else error

        

def mean_relative_rmse_error_ornl(x, y, axis = 1):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2, axis = axis)
    maxv = np.max(x, axis = axis)
    minv = np.min(x, axis = axis)
    return np.mean(np.sqrt(mse)/(maxv - minv))


def relative_l2_error_mgard(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.sum((x-y)**2)
    temp = np.sum((x)**2)
    return np.sqrt(mse/temp), mse, temp

def max_relative_l2_error(original_data,  recons_data, shape = [-1, 20*16*16]):
    original_data = original_data.reshape(shape)
    recons_data = recons_data.reshape(shape)
    diff = np.abs(original_data-recons_data)
    error_norm = np.linalg.norm(diff, axis=1)
    return np.max(error_norm), np.max(diff)

