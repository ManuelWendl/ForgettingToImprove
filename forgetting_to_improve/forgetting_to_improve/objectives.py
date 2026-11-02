import numpy as np

def sin_symmetric_lengthscale_increase(x, base_freq=0.5, freq_growth=0.39, amplitude=1.0):
    x = np.asarray(x)
    frequency = base_freq + freq_growth * np.abs(x)
    y = amplitude * np.sin(frequency * x) 
    return y

def sin_asymmetric_lengthscale_increase(x, base_freq=0.5, freq_growth_pos=0.39, amplitude=1.0):
    x = np.asarray(x)
    frequency = np.where(x >= 0, base_freq + freq_growth_pos * x, base_freq + freq_growth_pos * 1/np.abs(x))
    y = amplitude * np.sin(frequency * x)
    return y
