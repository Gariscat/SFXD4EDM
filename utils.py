import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Union
import os
import shutil
from tqdm import tqdm
import soundfile as sf

SFX_CLASSES = [
    ### FX
    'glitch',
    'impact',
    'stab',
    'sub_drop',
    'sweep',
    'lazer',
    'alarm',
    'white_noise',
    'vocal_chop',
    'dubstep_growl',
    ### DRUMS
    'kick',
    'snare',
    'open_hat',
    'closed_hat',
    'crash',
    'snap',
    'toms',
    'clap',
]

def check_waveform_class(waveform_path: str) -> Union[str, None]:
    waveform_path = waveform_path.lower()
    
    for sfx_class in SFX_CLASSES:
        if sfx_class.lower() in waveform_path:
            return sfx_class
    
    return None


def organize_waveforms(source_dir: str, target_dir: str):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Enumerate recursively into the source directory
    for root, dirs, files in tqdm(os.walk(source_dir), desc='Enumerating waveforms'):
        for file in files:
            category = check_waveform_class(file)
            if file.endswith('.wav') and category is not None:
                # Load the waveform
                waveform, sr = librosa.core.load(os.path.join(root, file))
                
                # Clip the waveform
                clipped_waveform = clip_waveform(waveform)
                
                # Create the subfolder in the target directory
                category_dir = os.path.join(target_dir, category)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)
                
                # Save the clipped waveform in the subfolder
                target_file = os.path.join(category_dir, file)
                sf.write(target_file, clipped_waveform, sr)



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
    """y, sr = librosa.core.load('test.wav')
    clip_waveform(y, threshold=0.1, plot_rms=True)"""
    organize_waveforms('/Users/ca7ax/Packs', '/Users/ca7ax/Packs-Organized')