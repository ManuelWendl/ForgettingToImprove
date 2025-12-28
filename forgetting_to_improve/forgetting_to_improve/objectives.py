import numpy as np

def sin_symmetric_lengthscale_increase(x, base_freq=0.5, freq_growth=0.39, amplitude=1.0):
    x = np.asarray(x)
    frequency = base_freq + freq_growth * np.abs(x)
    y = amplitude * np.sin(frequency * x) 
    return y

def sin_asymmetric_lengthscale_increase(x, base_freq=0.5, freq_growth_pos=0.39, amplitude=1.0):
    x = np.asarray(x)
    frequency = base_freq + freq_growth_pos * x
    y = amplitude * np.sin(frequency * x)
    return y

# def sin_periodic_lengthscale_increase(x, base_freq=0.5, freq_growth=0.3, amplitude=1):
#     x = np.asarray(x)
#     frequency = 2 - np.cos(freq_growth * x)
#     # Disturbin high frequencies between 5 and 10 
#     hf = 0.75 * np.random.randn(len(x)) * np.exp(-0.15 * (x - 6)**2)
#     y = amplitude * np.sin(frequency * x) + 0.1 * x + hf + 0.5 *np.random.randn(len(x))* (x <= -10)
#     return y

def sin_periodic_lengthscale_increase(x, base_freq=0.5, freq_growth=0.3, amplitude=1):
    x = np.asarray(x)
    frequency1 = 1
    frequency2 = 0.33
    # Disturbin high frequencies between 5 and 10 
    y = amplitude * np.sin(frequency1 * x) * (x >= 0) + amplitude * np.sin(frequency2 * x) * (x < 0)
    y = y + np.random.randn(len(x)) * 0.1
    return y