import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Union

def clip_waveform(waveform: np.ndarray, frame_length: int = 2048, threshold: float = 0.1, plot_rms: bool = False) -> np.ndarray:
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length).flatten()
    start_idx = 0
    end_idx = len(waveform) // frame_length
    
    if plot_rms:
        plt.plot(rms)
        plt.xlabel('Frame')
        plt.ylabel('RMS')
        plt.title('RMS Plot')
        plt.show()
        plt.close()
    
    for i in range(len(rms)):
        if rms[i] >= threshold:
            start_idx = i
            break
    
    for i in range(len(rms)-1, -1, -1):
        if rms[i] >= threshold:
            end_idx = i + 1
            break
    
    start_idx *= frame_length
    end_idx *= frame_length
    
    """
    plt.plot(waveform)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Waveform Plot')
    plt.axvline(x=start_idx, color='r', linestyle='--')
    plt.axvline(x=end_idx, color='r', linestyle='--')
    plt.show()
    plt.close()
    """
    return waveform[start_idx:end_idx]


if __name__ == '__main__':
    y, sr = librosa.core.load('test.wav')
    clip_waveform(y, threshold=0.1, plot_rms=True)