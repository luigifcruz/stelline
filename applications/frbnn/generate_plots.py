import os
import time
import numpy as np 
import pandas as pd
from tqdm import tqdm
from scipy import constants
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import softmax

BSIZE = 32
NCHAN = 192
NSAMPLES = 2048
TIME_LENGTH = 128e-3
SYNC_TIME = 1725998555

CENTER_FREQ = 1800e6
BANDWIDTH = 96e6
SOURCE_DM = 56.7

@dataclass
class Hit:
    telescope_timestamp: int
    unix_timestamp: int
    batch_index: int
    confidence: float

    def __str__(self):
        str = "Hit:\n"
        str += f"  Telescope Timestamp: {self.telescope_timestamp}\n"
        str += f"  UNIX Timestamp: {self.unix_timestamp}\n"
        str += f"  Batch Index: {self.batch_index}\n"
        str += f"  Confidence: {self.confidence}"
        return str

@dataclass
class Analysis:
    threshold: float
    number_of_inferences: int
    number_of_hits: int
    number_of_misses: int
    prelimary_hit_rate: float

    def __str__(self):
        str = "Analysis:\n"
        str += f"  Threshold: {self.threshold}\n"
        str += f"  Number of Inferences: {self.number_of_inferences}\n"
        str += f"  Number of Hits: {self.number_of_hits}\n"
        str += f"  Number of Misses: {self.number_of_misses}\n"
        str += f"  Preliminary Hit Rate: {self.prelimary_hit_rate}"
        return str

def parse_csv(filename, threshold=0.992):
    # Parse CSV.

    # Iteration,Result A,Result B,Argmax,Hit,Batch Index,Telescope Timestamp
    data = pd.read_csv(filename)
    number_of_inferences = len(data)
    data = data[data['Argmax'] == 1]

    print("Parsing CSV...")
    t = tqdm(total=len(data))

    hits = []
    for _, row in data.iterrows():
        telescope_timestamp = int(row['Telescope Timestamp'])
        result_a = float(row['Result A'])
        result_b = float(row['Result B'])
        argmax = int(row['Argmax'])
        batch_index = int(row['Batch Index'])

        if argmax == 1:
            unix_timestamp = int(telescope_timestamp * 2e-6 + SYNC_TIME)
            confidence = softmax(np.array([result_a, result_b]))[-1]

            if confidence < threshold:
                continue

            hits.append(Hit(telescope_timestamp, unix_timestamp, batch_index, confidence))

        t.update(1)

    hits.sort(key=lambda x: -x.confidence)

    t.close()

    number_of_hits = len(hits)
    number_of_misses = number_of_inferences - number_of_hits

    analysis = Analysis(
        threshold,
        number_of_inferences, 
        number_of_hits, 
        number_of_misses, 
        number_of_hits / number_of_misses
    )

    return hits, analysis

def plot_hit(hit, data, dedispersed_data):
    timestamp = hit.unix_timestamp
    timestamp += hit.batch_index * TIME_LENGTH

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Set the main title for the entire figure
    fig.suptitle(f"Hit ({datetime.fromtimestamp(timestamp).isoformat()}Z) [{hit.confidence*100:.6f}%]", fontsize=16)
    
    # Plot original data
    im1 = ax1.imshow(10 * np.log10(data), aspect=10)
    ax1.set_title('Original Data')
    ax1.set_ylabel('Frequency Channel (0.5 MHz/channel)')
    ax1.set_xlabel('Time Steps (0.0625 ms/step)')
    
    # Plot dedispersed data
    im2 = ax2.imshow(10 * np.log10(dedispersed_data), aspect=10)
    ax2.set_title('Dedispersed Data')
    ax2.set_xlabel('Time Steps (0.0625 ms/step)')
    
    # Add colorbars
    fig.colorbar(im1, ax=ax1, label='Power (dB)')
    fig.colorbar(im2, ax=ax2, label='Power (dB)')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'./hits/hit_{int(hit.confidence*1e8)}_{hit.unix_timestamp}_{hit.batch_index}.png', dpi=200)
    plt.close()

def save_hit(hit, data):
    np.save(f'./hits/hit_{int(hit.confidence*1e8)}_{hit.unix_timestamp}_{hit.batch_index}.npy', data)

def dedisperse_hit(hit, data):
    C = constants.c                               # Speed of light in m/s
    F_LOW = (CENTER_FREQ - BANDWIDTH / 2) / 1e9   # Lower frequency bound in Hz
    F_HIGH = (CENTER_FREQ + BANDWIDTH / 2) / 1e9  # Upper frequency bound in Hz

    freqs = np.linspace(F_LOW, F_HIGH, NCHAN)
    times = np.linspace(0, TIME_LENGTH, NSAMPLES)

    def dedisperse(data, freqs, times, dm):
        dedispersed = np.zeros_like(data)
        f_ref = freqs[-1]
        
        for i, f in enumerate(freqs):
            dt = 4.148808e-3 * dm * (1/f**2 - 1/f_ref**2)  # Dispersion delay
            shift = int(dt / (times[1] - times[0]))        # Convert to sample shift
            dedispersed[i] = np.roll(data[i], -shift)
        
        return dedispersed

    return dedisperse(data, freqs, times, SOURCE_DM)

def parse_hits(hits):
    print("Parsing Hits...")
    t = tqdm(total=len(hits))

    for hit in hits:
        try:
            f = open(f'../../build/FRBNN-HIT-{hit.telescope_timestamp}.bin', 'rb')
            data = np.fromfile(f, dtype=np.float32)

            if data.shape[0] != BSIZE*NCHAN*NSAMPLES:
                tqdm.write(f'Data shape mismatch for hit {hit.telescope_timestamp}.')
                data = data[0:BSIZE*NCHAN*NSAMPLES]

            data = data.reshape((BSIZE, NCHAN, NSAMPLES))
            data = data[hit.batch_index, :, :]
            f.close()
        except Exception as e:
            tqdm.write(f'Failed to load data for hit {hit.telescope_timestamp}.')
            if os.getenv('DEBUG'):
                tqdm.write(f'  {e}')
            continue

        save_hit(hit, data)
        dedispersed_data = dedisperse_hit(hit, data)
        plot_hit(hit, data, dedispersed_data)

        t.update(1)
    
    t.close()

hits, analysis = parse_csv('../../build/FRBNN-RUN.csv', 0.5)

print(analysis)

parse_hits(hits)

print('Done.')