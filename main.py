import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Generate white noise
sr = 22050  # Sample rate
duration = 2.0  # seconds
white_noise = np.random.randn(int(sr * duration))

# Load a sample speech file
audio_path = librosa.example('trumpet')
speech, sr_speech = librosa.load(audio_path, sr=sr)

# Compute MFCCs for white noise
mfcc_noise = librosa.feature.mfcc(y=white_noise, sr=sr, n_mfcc=13)
print(np.shape(mfcc_noise))

# Compute MFCCs for speech
mfcc_speech = librosa.feature.mfcc(y=speech, sr=sr_speech, n_mfcc=13)

# Calculate variance of MFCC coefficients
variance_noise = np.var(mfcc_noise, axis=1)
variance_speech = np.var(mfcc_speech, axis=1)

# Print the variance
print("Variance of MFCC coefficients for White Noise:")
print(variance_noise)
print("\nVariance of MFCC coefficients for Speech:")
print(variance_speech)

# Plot MFCCs for white noise
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
librosa.display.specshow(mfcc_noise, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC of White Noise')

# Plot MFCCs for speech
plt.subplot(2, 1, 2)
librosa.display.specshow(mfcc_speech, sr=sr_speech, x_axis='time')
plt.colorbar()
plt.title('MFCC of Speech')

plt.tight_layout()
plt.show()
