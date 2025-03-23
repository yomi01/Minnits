import pyaudio
import wave
import time
import threading
import os
import numpy as np

class AudioRecorder:
    def __init__(self, output_dir="recordings"):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = 44100
        self.output_dir = output_dir
        self.recording = False
        self.frames = []
        self.max_duration = 3600  # Maximum recording duration in seconds (1 hour)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.frames = []
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
        
        print("Recording started...")
        return True
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.recording:
            return False
        
        self.recording = False
        self.record_thread.join()
        
        # Generate output filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        # Save audio file
        self._save_audio(output_filename)
        
        print(f"Recording stopped. File saved as {output_filename}")
        return output_filename
    
    def _record(self):
        """Record audio data"""
        audio = pyaudio.PyAudio()
        
        # Open audio stream
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        start_time = time.time()
        
        # Record until stopped or max duration reached
        while self.recording and (time.time() - start_time) < self.max_duration:
            data = stream.read(self.chunk)
            self.frames.append(data)
        
        # Clean up
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if (time.time() - start_time) >= self.max_duration:
            print("Maximum recording duration reached (1 hour)")
            self.recording = False
    
    def _save_audio(self, output_filename):
        """Save recorded audio to a WAV file"""
        audio = pyaudio.PyAudio()
        
        # Create and save WAV file
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        audio.terminate()
        
        return output_filename

if __name__ == "__main__":
    # Simple test of the recorder
    recorder = AudioRecorder()
    recorder.start_recording()
    print("Recording for 5 seconds...")
    time.sleep(5)
    output_file = recorder.stop_recording()
    print(f"Test recording saved to {output_file}")