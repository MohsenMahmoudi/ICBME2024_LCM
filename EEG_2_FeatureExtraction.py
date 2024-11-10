import numpy as np
from scipy.signal import butter, lfilter, welch

class RealTimeEEGFeatureExtractor:
    def __init__(self, fs=250, segment_length=1.0):
        """
        Initialize the real-time EEG feature extractor.

        Parameters:
        - fs: Sampling frequency in Hz (default: 250 Hz)
        - segment_length: Length of the data segment in seconds for feature extraction (default: 1.0 s)
        """
        self.fs = fs
        self.segment_length = segment_length
        self.segment_samples = int(self.segment_length * self.fs)
        self.buffer = None

    def extract_features(self, filtered_chunk):
        """
        Extract alpha and beta band power features from the filtered EEG data chunk.

        Parameters:
        - filtered_chunk: A numpy array of shape (n_samples, n_channels)

        Returns:
        - features: A numpy array of shape (n_channels * 2,), containing alpha and beta band powers
        """
        # Initialize buffer if necessary
        if self.buffer is None:
            self.buffer = filtered_chunk
        else:
            # Append new data to the buffer
            self.buffer = np.vstack((self.buffer, filtered_chunk))

        # Keep only the latest segment_length seconds of data
        if self.buffer.shape[0] > self.segment_samples:
            self.buffer = self.buffer[-self.segment_samples:, :]

        # Check if we have enough data for feature extraction
        if self.buffer.shape[0] < self.segment_samples:
            return None  # Not enough data yet

        # Initialize list to hold features
        features = []

        # For each channel, compute alpha and beta band powers
        for ch in range(self.buffer.shape[1]):
            f, Pxx = welch(self.buffer[:, ch], fs=self.fs, nperseg=256, noverlap=128)

            # Compute band powers
            alpha_band = np.logical_and(f >= 8, f <= 13)
            beta_band = np.logical_and(f > 13, f <= 30)

            alpha_power = np.trapz(Pxx[alpha_band], f[alpha_band])
            beta_power = np.trapz(Pxx[beta_band], f[beta_band])

            features.extend([alpha_power, beta_power])

        return np.array(features)

# Example usage within the real-time processing loop
if __name__ == "__main__":
    import time

    # Simulate real-time data acquisition and preprocessing
    fs = 250  # Sampling frequency in Hz
    n_channels = 11  # Number of EEG channels
    chunk_duration = 0.1  # Duration of each data chunk in seconds (100 ms)
    chunk_size = int(chunk_duration * fs)  # Number of samples per chunk
    total_duration = 10  # Total duration of simulation in seconds
    n_chunks = int(total_duration / chunk_duration)

    # Initialize preprocessor and feature extractor
    preprocessor = RealTimeEEGPreprocessor(fs=fs)
    feature_extractor = RealTimeEEGFeatureExtractor(fs=fs, segment_length=1.0)

    for i in range(n_chunks):
        # Simulate data acquisition delay
        time.sleep(chunk_duration)

        # Simulate incoming data chunk (random data for demonstration)
        data_chunk = np.random.randn(chunk_size, n_channels)

        # Preprocess the data chunk
        filtered_chunk = preprocessor.process_chunk(data_chunk)

        # Extract features
        features = feature_extractor.extract_features(filtered_chunk)

        # Check if features are available
        if features is not None:
            # Proceed to machine learning inference
            print(f"Features extracted at chunk {i+1}: {features}")
        else:
            print(f"Not enough data to extract features at chunk {i+1}")
