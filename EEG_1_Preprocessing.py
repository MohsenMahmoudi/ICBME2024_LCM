import numpy as np
from scipy.signal import butter, lfilter

class RealTimeEEGPreprocessor:
    def __init__(self, fs=250, lowcut=1.0, highcut=50.0, order=2):
        """
        Initialize the real-time EEG preprocessor.

        Parameters:
        - fs: Sampling frequency in Hz (default: 250 Hz)
        - lowcut: Low cutoff frequency in Hz (default: 1.0 Hz)
        - highcut: High cutoff frequency in Hz (default: 50.0 Hz)
        - order: Order of the Butterworth filter (default: 2)
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        # Design a causal Butterworth band-pass filter
        nyquist = 0.5 * fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.b, self.a = butter(self.order, [low, high], btype='bandpass', analog=False)

        # Initialize filter state for real-time processing
        self.zi = None

    def process_chunk(self, data_chunk):
        """
        Process a chunk of EEG data.

        Parameters:
        - data_chunk: A numpy array of shape (n_samples, n_channels)

        Returns:
        - filtered_chunk: The filtered data chunk
        """
        if self.zi is None:
            # Initialize filter state for each channel
            self.zi = np.zeros((self.a.size - 1, data_chunk.shape[1]))

        # Apply the filter to each channel
        filtered_chunk = np.zeros_like(data_chunk)
        for ch in range(data_chunk.shape[1]):
            filtered_chunk[:, ch], self.zi[:, ch] = lfilter(
                self.b, self.a, data_chunk[:, ch], zi=self.zi[:, ch]
            )

        return filtered_chunk

# Example usage
if __name__ == "__main__":
    import time

    # Simulate real-time data acquisition
    fs = 250  # Sampling frequency in Hz
    n_channels = 11  # Number of EEG channels (including auxiliary channels)
    chunk_duration = 0.1  # Duration of each data chunk in seconds (100 ms)
    chunk_size = int(chunk_duration * fs)  # Number of samples per chunk
    total_duration = 10  # Total duration of simulation in seconds
    n_chunks = int(total_duration / chunk_duration)

    # Initialize the preprocessor
    preprocessor = RealTimeEEGPreprocessor(fs=fs)

    # Simulate real-time processing
    for i in range(n_chunks):
        # Simulate data acquisition delay (mimicking real-time streaming)
        time.sleep(chunk_duration)

        # Simulate incoming data chunk (random data for demonstration)
        data_chunk = np.random.randn(chunk_size, n_channels)

        # Record processing start time
        start_time = time.time()

        # Process the data chunk
        filtered_chunk = preprocessor.process_chunk(data_chunk)

        # Calculate processing lag time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Print processing time for the current chunk
        print(f"Chunk {i+1}/{n_chunks}: Processing time = {processing_time:.2f} ms")

        # Ensure processing time is within the acceptable lag time
        if processing_time > 20:
            print("Warning: Processing lag exceeds 20 ms!")
