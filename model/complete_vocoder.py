import numpy as np
import torch
import torch.nn.functional as F
import json
import os
from scipy.io.wavfile import write
from hifi_gan.models import Generator
from hifi_gan.meldataset import MAX_WAV_VALUE

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class HiFiVocoder():
    def __init__(self, ckpt_path, output_dir, device, config_path='src/HiFiGan_config.json'):
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.device = device

        # Load configuration
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)

        # Calculate expected mel frames per segment
        self.expected_frames = self.h.segment_size // self.h.hop_size
        print(f"Vocoder initialized with segment_size: {self.h.segment_size}")
        print(f"Expected mel frames per segment: {self.expected_frames}")
        print(f"Expected audio duration per segment: {self.expected_frames * self.h.hop_size / self.h.sampling_rate:.3f} seconds")

        self.model_init()

    def model_init(self):
        self.generator = Generator(self.h).to(self.device)
        state_dict_g = torch.load(self.ckpt_path, map_location=self.device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        print(f"Model loaded from {self.ckpt_path}")

    def _process_chunk(self, mel_chunk):
        """Process a single chunk of mel spectrogram"""
        with torch.no_grad():
            if len(mel_chunk.shape) < 3:
                mel_chunk = mel_chunk.unsqueeze(0)

            # Ensure chunk has the right length
            current_frames = mel_chunk.shape[-1]
            if current_frames < self.expected_frames:
                # Pad if too short
                pad_size = self.expected_frames - current_frames
                mel_chunk = F.pad(mel_chunk, (0, pad_size))
            elif current_frames > self.expected_frames:
                # Truncate if too long (shouldn't happen with proper chunking)
                mel_chunk = mel_chunk[:, :, :self.expected_frames]

            y_g_hat = self.generator(mel_chunk)
            audio = y_g_hat.squeeze() * MAX_WAV_VALUE
            return audio.cpu().numpy().astype('int16')

    def __call__(self, mel_spectrogram, write_fname=None, overlap_frames=0, crossfade=True):
        """
        Process mel spectrogram with automatic chunking for long sequences

        Args:
            mel_spectrogram: Input mel spectrogram (shape: [n_mels, time] or [1, n_mels, time])
            write_fname: Optional filename for saving output
            overlap_frames: Number of frames to overlap between chunks (for smoother transitions)
            crossfade: Whether to apply crossfading between overlapping chunks
        """
        # Ensure mel_spectrogram is in the right format
        if isinstance(mel_spectrogram, np.ndarray):
            mel = torch.FloatTensor(mel_spectrogram).to(self.device)
        else:
            mel = mel_spectrogram.to(self.device)

        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)  # Add batch dimension

        total_frames = mel.shape[-1]
        print(f"Input mel spectrogram shape: {mel.shape}")
        print(f"Total frames: {total_frames}")
        print(f"Expected duration: {total_frames * self.h.hop_size / self.h.sampling_rate:.3f} seconds")

        # If input is shorter than expected, process directly
        if total_frames <= self.expected_frames:
            print("Input shorter than segment size, processing directly...")
            audio = self._process_chunk(mel)
        else:
            # Process in chunks
            print(f"Input longer than segment size, chunking into {self.expected_frames}-frame segments...")
            audio_chunks = []

            hop_size = self.expected_frames - overlap_frames
            if hop_size <= 0:
                hop_size = self.expected_frames // 2
                print(f"Warning: overlap_frames too large, using {hop_size} instead")

            start = 0
            chunk_idx = 0

            while start < total_frames:
                end = min(start + self.expected_frames, total_frames)
                chunk = mel[:, :, start:end]

                print(f"Processing chunk {chunk_idx + 1}: frames {start}-{end}")

                chunk_audio = self._process_chunk(chunk)
                audio_chunks.append(chunk_audio)

                # Move to next chunk
                start += hop_size
                chunk_idx += 1

                if start >= total_frames:
                    break

            print(f"Processed {len(audio_chunks)} chunks")

            # Combine chunks
            if crossfade and overlap_frames > 0 and len(audio_chunks) > 1:
                audio = self._crossfade_chunks(audio_chunks, overlap_frames, hop_size)
            else:
                audio = np.concatenate(audio_chunks, axis=0)

        # Save if requested
        if write_fname:
            output_file = os.path.join(self.output_dir, str(write_fname) + '_generated.wav')
            write(output_file, self.h.sampling_rate, audio)
            print(f"Saved: {output_file}")
            print(f"Output duration: {len(audio) / self.h.sampling_rate:.3f} seconds")

        return audio

    def _crossfade_chunks(self, audio_chunks, overlap_frames, hop_size):
        """Apply crossfading between overlapping chunks"""
        if len(audio_chunks) < 2:
            return audio_chunks[0] if audio_chunks else np.array([])

        # Calculate overlap in audio samples
        overlap_samples = int(overlap_frames * self.h.hop_size)

        result = [audio_chunks[0]]  # First chunk without crossfading

        for i in range(1, len(audio_chunks)):
            current_chunk = audio_chunks[i]

            # Get the overlapping portion of the previous chunk
            prev_chunk_end = result[-1]
            if len(prev_chunk_end) >= overlap_samples:
                prev_overlap = prev_chunk_end[-overlap_samples:]
                current_overlap = current_chunk[:overlap_samples]

                # Crossfade
                crossfade_window = np.linspace(0, 1, overlap_samples)
                crossfaded_overlap = (prev_overlap * (1 - crossfade_window) +
                                    current_overlap * crossfade_window)

                # Combine non-overlapping part of current chunk with crossfaded overlap
                non_overlap = current_chunk[overlap_samples:]
                crossfaded_chunk = np.concatenate([crossfaded_overlap, non_overlap])

                result.append(crossfaded_chunk)
            else:
                # No overlap possible, just concatenate
                result.append(current_chunk)

        return np.concatenate(result, axis=0)

    def get_expected_duration(self, mel_frames):
        """Calculate expected audio duration for given number of mel frames"""
        return mel_frames * self.h.hop_size / self.h.sampling_rate

    def get_max_mel_frames(self):
        """Get maximum mel frames that can be processed in one chunk"""
        return self.expected_frames


def main():
    """Example usage"""
    print('HiFi-GAN Complete Vocoder Example')

    # Initialize vocoder
    vocoder = HiFiVocoder(
        ckpt_path='/home/nikhil/jukedrummer/hifi_gan/cp_hifigan/g_00055000',
        output_dir='output',
        device='cuda:0'  # or 'cpu'
    )

    # Example: Process a mel spectrogram
    # mel = np.load('path/to/your/mel_spectrogram.npy')  # Shape: (80, time)
    # audio = vocoder(mel, write_fname='test_output')

    print("Vocoder ready for use!")
    print(f"Max mel frames per chunk: {vocoder.get_max_mel_frames()}")
    print(f"Max audio duration per chunk: {vocoder.get_expected_duration(vocoder.get_max_mel_frames()):.3f} seconds")


if __name__ == '__main__':
    main()