"""
Enhanced audio recording with system audio capture support.
This module provides fallback methods for system audio recording on Windows.
"""

import pyaudio
import platform
import subprocess
import os

def check_windows_stereo_mix():
    """Check if Stereo Mix is available and enabled on Windows"""
    if platform.system() != "Windows":
        return False
    
    try:
        # Try to run a PowerShell command to check audio devices
        # This is a more reliable way to detect system audio capabilities
        cmd = [
            "powershell", "-Command",
            "Get-WmiObject -Class Win32_SoundDevice | Select-Object Name,Status"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "Stereo Mix" in result.stdout or "Loopback" in result.stdout
    except:
        return False

def enable_windows_stereo_mix():
    """Provide instructions to enable Stereo Mix on Windows"""
    instructions = """
    To enable System Audio Recording on Windows:
    
    1. Right-click the speaker icon in the system tray
    2. Select "Open Sound settings" or "Sounds"
    3. Click on "Sound Control Panel" or "Recording" tab
    4. Right-click in the empty space and check "Show Disabled Devices"
    5. Look for "Stereo Mix" and right-click it
    6. Select "Enable"
    7. Set it as the default recording device if needed
    
    Alternative: Use Windows 10/11 built-in app isolation:
    - Some modern Windows versions have per-app audio routing
    - Check Windows Settings > System > Sound > App volume and device preferences
    """
    return instructions

def get_wasapi_devices():
    """Try to get WASAPI loopback devices on Windows"""
    if platform.system() != "Windows":
        return []
    
    try:
        audio = pyaudio.PyAudio()
        wasapi_devices = []
        
        device_count = audio.get_device_count()
        for i in range(device_count):
            try:
                info = audio.get_device_info_by_index(i)
                name = info['name'].lower()
                
                # Look for WASAPI devices that support loopback
                if 'wasapi' in name and ('speakers' in name or 'output' in name):
                    # Try to determine if this device supports loopback
                    if info['maxInputChannels'] > 0:
                        wasapi_devices.append({
                            'index': i,
                            'name': info['name'],
                            'channels': info['maxInputChannels'],
                            'rate': int(info['defaultSampleRate'])
                        })
            except:
                continue
                
        audio.terminate()
        return wasapi_devices
    except:
        return []
