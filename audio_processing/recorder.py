import pyaudio
import wave
import time
import threading
import os
import numpy as np
import platform

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
        self.audio_source = "microphone"  # Default to microphone
        self.input_device_index = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize PyAudio and discover audio devices
        self._init_audio_devices()
    
    def _init_audio_devices(self):
        """Initialize and discover available audio devices"""
        self.audio = pyaudio.PyAudio()
        self.devices = self._get_audio_devices()
        
    def _get_audio_devices(self):
        """Get list of available audio devices with their capabilities"""
        devices = []
        device_count = self.audio.get_device_count()
        
        for i in range(device_count):
            try:
                info = self.audio.get_device_info_by_index(i)
                # Check if device supports input
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'rate': int(info['defaultSampleRate']),
                        'is_loopback': self._is_loopback_device(info)
                    })
            except Exception as e:
                # Skip devices that cause errors
                continue
                
        return devices
    
    def _is_loopback_device(self, device_info):
        """Check if device is a loopback/system audio device"""
        device_name = device_info['name'].lower()
        
        # Common loopback device names on Windows
        loopback_keywords = [
            'stereo mix', 'what u hear', 'wave out mix', 'mixed output',
            'speakers', 'loopback', 'monitor', 'wasapi'
        ]
        
        # Check if it's a WASAPI loopback device (Windows)
        if platform.system() == "Windows":
            # WASAPI devices often have specific patterns
            if any(keyword in device_name for keyword in loopback_keywords):
                return True
            # Windows WASAPI loopback devices sometimes have "- Output" in the name
            if "speakers" in device_name and "wasapi" in device_name:
                return True
                
        return False
    
    def get_available_sources(self):
        """Get list of available audio sources"""
        sources = [
            {
                'id': 'microphone',
                'name': 'Microphone (Default)',
                'description': 'Record from default microphone input'
            }
        ]
        
        # Add system audio sources
        loopback_devices = [d for d in self.devices if d['is_loopback']]
        if loopback_devices:
            sources.append({
                'id': 'system_audio',
                'name': 'System Audio (Loopback)',
                'description': 'Record audio playing on the computer (YouTube, Zoom, etc.)'
            })
            
        return sources
    
    def set_audio_source(self, source_type):
        """Set the audio source type"""
        if source_type == "system_audio":
            self.audio_source = "system_audio"
            # Try to find the best loopback device
            loopback_devices = [d for d in self.devices if d['is_loopback']]
            if loopback_devices:
                # Prefer WASAPI devices or devices with "speakers" in name
                best_device = None
                for device in loopback_devices:
                    if "wasapi" in device['name'].lower() and "speakers" in device['name'].lower():
                        best_device = device
                        break
                if not best_device:
                    best_device = loopback_devices[0]  # Use first available
                    
                self.input_device_index = best_device['index']
                self.channels = min(2, best_device['channels'])  # Use stereo if available
                self.rate = best_device['rate']
                return True
            else:
                return False
        else:
            self.audio_source = "microphone"
            self.input_device_index = None
            self.channels = 2
            self.rate = 44100
            return True
    
    def start_recording(self):
        """Start recording audio"""
        self.recording = True
        self.frames = []
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
        
        source_desc = "system audio" if self.audio_source == "system_audio" else "microphone"
        print(f"Recording started from {source_desc}...")
        return True
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.recording:
            return False
        
        self.recording = False
        self.record_thread.join()
        
        # Generate output filename with timestamp and source type
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        source_suffix = "_system" if self.audio_source == "system_audio" else "_mic"
        output_filename = os.path.join(self.output_dir, f"recording_{timestamp}{source_suffix}.wav")
        
        # Save audio file
        self._save_audio(output_filename)
        
        source_desc = "system audio" if self.audio_source == "system_audio" else "microphone"
        print(f"Recording stopped. {source_desc.title()} recording saved as {output_filename}")
        return output_filename
    
    def _record(self):
        """Record audio data"""
        audio = pyaudio.PyAudio()
        
        try:
            # Configure stream parameters based on audio source
            stream_params = {
                'format': self.format,
                'channels': self.channels,
                'rate': self.rate,
                'input': True,
                'frames_per_buffer': self.chunk
            }
            
            # Add device index if specified (for system audio)
            if self.input_device_index is not None:
                stream_params['input_device_index'] = self.input_device_index
                
            # Open audio stream
            stream = audio.open(**stream_params)
            
            start_time = time.time()
            
            # Record until stopped or max duration reached
            while self.recording and (time.time() - start_time) < self.max_duration:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Warning: Audio read error: {e}")
                    # Continue recording despite minor errors
                    continue
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Error setting up audio stream: {e}")
            if self.audio_source == "system_audio":
                print("System audio recording failed. Make sure:")
                print("1. Stereo Mix or similar loopback device is enabled")
                print("2. Audio is currently playing on the system")
                print("3. You have permission to access audio devices")
        finally:
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
    
    def list_audio_devices(self):
        """List all available audio devices for debugging"""
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        
        for device in self.devices:
            device_type = "LOOPBACK" if device['is_loopback'] else "INPUT"
            print(f"[{device['index']}] {device['name']}")
            print(f"    Type: {device_type}")
            print(f"    Channels: {device['channels']}")
            print(f"    Sample Rate: {device['rate']} Hz")
            print()
    
    def cleanup(self):
        """Clean up audio resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass

if __name__ == "__main__":
    # Simple test of the recorder
    recorder = AudioRecorder()
    recorder.start_recording()
    print("Recording for 5 seconds...")
    time.sleep(5)
    output_file = recorder.stop_recording()
    print(f"Test recording saved to {output_file}")