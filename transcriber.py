import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import whisper
import moviepy
from moviepy.video.io.VideoFileClip import VideoFileClip
import threading
import torch
import time
import psutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import shutil
from pathlib import Path
import json
import concurrent.futures
import opencc  # Add this import at the top

def check_system_resources():
    """Check available system resources."""
    cpu_count = multiprocessing.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_total': memory.total,
        'memory_available': memory.available,
        'memory_percent': memory.percent
    }

def get_optimal_workers():
    """Calculate optimal number of worker processes based on system resources."""
    cpu_count = multiprocessing.cpu_count()
    memory = psutil.virtual_memory()
    
    # More aggressive CPU utilization while keeping system responsive
    # Use 75% of available cores by default
    recommended_workers = max(1, int(cpu_count * 0.75))
    
    # Ensure we have enough memory (1.5GB per worker for Whisper)
    memory_based_workers = max(1, int(memory.available / (1.5 * 1024 * 1024 * 1024)))
    
    # Use the lower number to be safe
    optimal_workers = min(recommended_workers, memory_based_workers)
    
    # Cap at 8 workers maximum to prevent diminishing returns
    return min(optimal_workers, 8)

def transcribe_segment(args):
    """Transcribe a segment of audio using Whisper model."""
    model, audio_segment, start_time, device = args
    try:
        result = model.transcribe(
            audio_segment,
            language=None,
            task="transcribe"
        )
        return start_time, result
    except Exception as e:
        return start_time, str(e)

def check_cuda_availability():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'device_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda
        }
    return {
        'available': False,
        'reason': "CUDA not available"
    }

class ModelCache:
    """Handles caching of Whisper models."""
    def __init__(self, initial_cache_dir=None):
        if initial_cache_dir:
            self.cache_dir = initial_cache_dir
        else:
            # Default to Cache folder in program directory
            self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cache")
        
        self.cache_info_file = os.path.join(self.cache_dir, "cache_info.json")
        self.ensure_cache_dir()
        self.load_cache_info()
        self.scan_cache_directory()  # Scan for existing models
        
    def ensure_cache_dir(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def load_cache_info(self):
        """Load cache information from JSON file."""
        if os.path.exists(self.cache_info_file):
            try:
                with open(self.cache_info_file, 'r') as f:
                    self.cache_info = json.load(f)
            except:
                self.cache_info = {}
        else:
            self.cache_info = {}
            
    def save_cache_info(self):
        """Save cache information to JSON file."""
        with open(self.cache_info_file, 'w') as f:
            json.dump(self.cache_info, f)
            
    def get_model_size(self, model_name):
        """Get model size in MB."""
        sizes = {
            "tiny": 75,      # ~75MB download (39M parameters)
            "base": 142,     # ~142MB download (74M parameters)
            "small": 461,    # ~461MB download (244M parameters)
            "medium": 1420,  # ~1.42GB download (769M parameters)
            "large": 2870,   # ~2.87GB download (1.55B parameters)
            "large-v2": 2870 # ~2.87GB download (1.55B parameters)
        }
        return sizes.get(model_name, 0)
            
    def scan_cache_directory(self):
        """Scan cache directory for existing models and update cache info."""
        if not os.path.exists(self.cache_dir):
            return
            
        # List of valid model names
        valid_models = ["tiny", "base", "small", "medium", "large"]
        
        # Check each model directory
        for model_name in valid_models:
            model_dir = os.path.join(self.cache_dir, model_name)
            if os.path.exists(model_dir):
                # Check for model files
                model_files = os.listdir(model_dir)
                has_pt = any(f.endswith('.pt') for f in model_files)
                has_bin = any(f.endswith('.bin') for f in model_files)
                
                if has_pt or has_bin:  # If either file exists, consider it cached
                    if model_name not in self.cache_info:
                        self.cache_info[model_name] = {
                            "cached_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "size_mb": self.get_model_size(model_name)
                        }
        
        # Save updated cache info
        self.save_cache_info()

    def verify_model_files(self, model_name):
        """Verify that model files exist in the cache."""
        model_dir = os.path.join(self.cache_dir, model_name)
        if not os.path.exists(model_dir):
            return False
            
        # Check for either .pt or .bin file
        model_files = os.listdir(model_dir)
        has_pt = any(f.endswith('.pt') for f in model_files)
        has_bin = any(f.endswith('.bin') for f in model_files)
        
        return has_pt or has_bin  # Consider cached if either file exists

    def is_model_cached(self, model_name):
        """Check if model is properly cached with all required files."""
        # First check cache info
        if model_name not in self.cache_info:
            return False
            
        # Then verify actual files exist
        return self.verify_model_files(model_name)
        
    def get_cache_path(self, model_name):
        """Get path where model should be cached."""
        return os.path.join(self.cache_dir, model_name)
        
    def add_to_cache(self, model_name):
        """Add model to cache information after verifying files."""
        if self.verify_model_files(model_name):
            self.cache_info[model_name] = {
                "cached_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "size_mb": self.get_model_size(model_name)
            }
            self.save_cache_info()
            return True
        return False
        
    def clear_cache(self):
        """Clear all cached models."""
        shutil.rmtree(self.cache_dir)
        self.ensure_cache_dir()
        self.cache_info = {}
        self.save_cache_info()
        
    def change_cache_location(self, new_location):
        """Change cache location and move existing cache."""
        if new_location == self.cache_dir:
            return
            
        # Create new cache directory
        new_cache_dir = os.path.abspath(new_location)
        os.makedirs(new_cache_dir, exist_ok=True)
        
        # Move existing cache if it exists
        if os.path.exists(self.cache_dir):
            for item in os.listdir(self.cache_dir):
                src = os.path.join(self.cache_dir, item)
                dst = os.path.join(new_cache_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            shutil.rmtree(self.cache_dir)
        
        # Update cache location
        self.cache_dir = new_cache_dir
        self.cache_info_file = os.path.join(self.cache_dir, "cache_info.json")
        self.save_cache_info()

class TranscriberApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Echoes")
        
        # Get screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        # Calculate default window size (40% of screen width, 80% of screen height)
        window_width = min(int(screen_width * 0.4), 800)  # More vertical ratio, cap at 800px
        window_height = min(int(screen_height * 0.8), 1000)  # Taller window, cap at 1000px
        
        # Calculate center position
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        
        # Set window geometry
        self.window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Set minimum window size
        self.window.minsize(500, 800)  # Increased minimum height to prevent squishing
        self.window.resizable(True, True)
        
        # Initialize model cache
        self.model_cache = ModelCache()
        
        # Check system resources
        self.system_info = check_system_resources()
        self.cuda_info = check_cuda_availability()
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Echoes - Multilingual Audio/Video Transcriber",
            font=("Helvetica", 20, "bold")
        )
        self.title_label.pack(pady=20, fill="x")
        
        # File selection frame
        self.file_frame = ctk.CTkFrame(self.main_frame)
        self.file_frame.pack(pady=10, padx=20, fill="x")
        
        self.file_paths = []  # List to store multiple file paths
        self.file_entry = ctk.CTkEntry(
            self.file_frame,
            textvariable=tk.StringVar(),
            placeholder_text="Select audio/video files...",
            state="readonly"  # Make entry read-only
        )
        self.file_entry.pack(side="left", padx=10, fill="x", expand=True)
        
        # Browse and Settings buttons frame
        self.button_frame = ctk.CTkFrame(self.file_frame, fg_color="transparent")
        self.button_frame.pack(side="right", padx=10)
        
        # Browse button
        self.browse_button = ctk.CTkButton(
            self.button_frame,
            text="Browse",
            command=self.browse_file,
            width=100
        )
        self.browse_button.pack(side="left", padx=5)
        
        # Settings button
        self.settings_button = ctk.CTkButton(
            self.button_frame,
            text="⚙️ Settings",
            command=self.show_settings,
            width=100
        )
        self.settings_button.pack(side="left", padx=5)
        
        # Transcribe button frame
        self.transcribe_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.transcribe_frame.pack(pady=10, padx=20, fill="x")
        
        # Transcribe button (renamed to Start)
        self.transcribe_button = ctk.CTkButton(
            self.transcribe_frame,
            text="Start",
            command=self.start_transcription,
            width=200,
            height=40,
            font=("Helvetica", 13, "bold")  # Make text more prominent
        )
        self.transcribe_button.pack(pady=5)
        
        # File list frame
        self.file_list_frame = ctk.CTkFrame(self.main_frame)
        self.file_list_frame.pack(pady=5, padx=20, fill="both", expand=True)
        
        # File list label
        self.file_list_label = ctk.CTkLabel(
            self.file_list_frame,
            text="Selected Files:",
            font=("Helvetica", 12)
        )
        self.file_list_label.pack(pady=5)
        
        # File list with scrollbar
        self.file_listbox = tk.Listbox(
            self.file_list_frame,
            selectmode=tk.EXTENDED,
            font=("Helvetica", 10),
            bg="#2b2b2b",
            fg="white",
            selectbackground="#1f538d",
            selectforeground="white"
        )
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        scrollbar = ctk.CTkScrollbar(self.file_list_frame, command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y", pady=5)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Progress frame with batch status
        self.progress_frame = ctk.CTkFrame(self.main_frame)
        self.progress_frame.pack(pady=10, padx=20, fill="x")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(pady=5, fill="x")
        self.progress_bar.set(0)
        
        # Status label with wrapping
        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready",
            font=("Helvetica", 12),
            wraplength=400  # Allow text to wrap
        )
        self.status_label.pack(pady=2, fill="x")
        
        # Time estimate label
        self.time_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=("Helvetica", 10)
        )
        self.time_label.pack(pady=2, fill="x")
        
        # Options frame (contains model, device selection)
        self.options_frame = ctk.CTkFrame(self.main_frame)
        self.options_frame.pack(pady=10, padx=20, fill="x")
        
        # Left column for labels
        self.labels_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        self.labels_frame.pack(side="left", padx=10, fill="both", expand=True)
        
        # Model selection
        self.model_label = ctk.CTkLabel(
            self.labels_frame,
            text="Model:",
            font=("Helvetica", 12)
        )
        self.model_label.pack(pady=5, anchor="w", fill="x")
        
        self.model_var = tk.StringVar(value="medium")
        self.model_menu = ctk.CTkOptionMenu(
            self.labels_frame,
            values=["tiny", "base", "small", "medium", "large"],
            variable=self.model_var,
            command=self.update_model_info
        )
        self.model_menu.pack(pady=5, fill="x")
        
        # Device selection
        self.device_label = ctk.CTkLabel(
            self.labels_frame,
            text="Processing Device:",
            font=("Helvetica", 12)
        )
        self.device_label.pack(pady=5, anchor="w", fill="x")
        
        if self.cuda_info['available']:
            gpu_info = f"GPU: {self.cuda_info['device_name']} (CUDA {self.cuda_info['cuda_version']})"
            self.device_var = tk.StringVar(value="cuda")
            device_options = ["cuda", "cpu"]
        else:
            gpu_info = "GPU: Not available - Using CPU"
            self.device_var = tk.StringVar(value="cpu")
            device_options = ["cpu"]
        
        self.device_menu = ctk.CTkOptionMenu(
            self.labels_frame,
            values=device_options,
            variable=self.device_var,
            command=self.update_device_selection
        )
        self.device_menu.pack(pady=5, fill="x")
        
        # System info frame with scrollable text
        self.system_frame = ctk.CTkFrame(self.main_frame)
        self.system_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Model info
        self.model_info_label = ctk.CTkLabel(
            self.system_frame,
            text="Model: Large - Best accuracy (1.5B parameters)",
            font=("Helvetica", 10)
        )
        self.model_info_label.pack(pady=2, fill="x")
        
        # GPU/CPU info
        self.gpu_info_label = ctk.CTkLabel(
            self.system_frame,
            text=gpu_info,
            font=("Helvetica", 10)
        )
        self.gpu_info_label.pack(pady=2, fill="x")
        
        self.cpu_info_label = ctk.CTkLabel(
            self.system_frame,
            text=f"CPU: {self.system_info['cpu_count']} cores available",
            font=("Helvetica", 10)
        )
        self.cpu_info_label.pack(pady=2, fill="x")
        
        # Cache info
        self.cache_label = ctk.CTkLabel(
            self.system_frame,
            text=f"Cache: {os.path.abspath(self.model_cache.cache_dir)}",
            font=("Helvetica", 10),
            wraplength=400  # Allow text to wrap
        )
        self.cache_label.pack(pady=2, fill="x")
        
        self.model = None
        self.is_processing = False
        self.start_time = None
        
        # Update model info initially
        self.update_model_info(self.model_var.get())
        
        # Set minimum window size to ensure all elements are visible
        self.window.minsize(600, 800)  # Increased minimum width
        self.window.geometry(f"{800}x{900}+{x_position}+{y_position}")  # Set default size
        
    def update_device_selection(self, value):
        """Handle device selection changes."""
        # Clear current model when device changes to force reload on correct device
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
            
        # Update device-specific information
        if value == "cuda" and self.cuda_info['available']:
            gpu_info = f"GPU: {self.cuda_info['device_name']} (CUDA {self.cuda_info['cuda_version']})"
        else:
            gpu_info = "GPU: Not available - Using CPU"
        
        self.gpu_info_label.configure(text=gpu_info)
        
        # Update model info to reflect new device
        self.update_model_info(self.model_var.get())

    def update_model_info(self, value):
        """Update model info and clear current model to force reload."""
        model_info = {
            "tiny": {"params": "39M", "size": "75MB", "desc": "Fastest"},
            "base": {"params": "74M", "size": "142MB", "desc": "Good balance"},
            "small": {"params": "244M", "size": "461MB", "desc": "Better accuracy"},
            "medium": {"params": "769M", "size": "1.42GB", "desc": "High accuracy"},
            "large": {"params": "1.5B", "size": "2.87GB", "desc": "Best accuracy"}
        }
        
        info = model_info[value]
        status = "Cached" if self.model_cache.is_model_cached(value) else "Not cached"
        device = self.device_var.get()
        device_str = "GPU" if device == "cuda" and self.cuda_info['available'] else "CPU"
        
        info_text = (
            f"Model: {value.capitalize()} - {info['desc']}\n"
            f"Parameters: {info['params']} • Size: {info['size']} • {status} • Using {device_str}"
        )
        self.model_info_label.configure(text=info_text)
        
        # Clear current model when selection changes
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
        
    def update_progress(self, progress, status, time_estimate=""):
        """Update progress with current file and batch information."""
        self.progress_bar.set(progress)
        
        # Get batch status
        total_files = len(self.file_paths) if hasattr(self, 'file_paths') else 0
        processed_files = len(self.processed_files) if hasattr(self, 'processed_files') else 0
        
        # Combine current file status with batch status
        if total_files > 0:
            batch_info = f"\nBatch Progress: {processed_files}/{total_files} files completed"
            combined_status = f"{status}{batch_info}"
        else:
            combined_status = status
            
        self.status_label.configure(text=combined_status)
        self.time_label.configure(text=time_estimate)
        self.window.update()
        
    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
        
    def format_srt_time(self, seconds):
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        
    def browse_file(self):
        filetypes = (
            ("Audio/Video files", "*.mp3 *.mp4 *.wav *.avi *.mkv"),
            ("All files", "*.*")
        )
        filenames = filedialog.askopenfilenames(
            title="Select files",
            filetypes=filetypes
        )
        if filenames:
            self.file_paths = list(filenames)
            self.file_listbox.delete(0, tk.END)
            for file in self.file_paths:
                self.file_listbox.insert(tk.END, os.path.basename(file))
            self.file_entry.configure(state="normal")
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, f"{len(self.file_paths)} file(s) selected")
            self.file_entry.configure(state="readonly")
            self.update_batch_status()

    def update_batch_status(self):
        """Update the batch processing status display."""
        if not self.file_paths:
            self.status_label.configure(text="Ready")
            self.progress_bar.set(0)
            return
        
        total_files = len(self.file_paths)
        processed_files = sum(1 for file in self.file_paths if hasattr(self, 'processed_files') and file in self.processed_files)
        current_progress = processed_files / total_files if total_files > 0 else 0
        
        # Update progress bar
        self.progress_bar.set(current_progress)

    def play_completion_sound(self):
        """Play a completion sound."""
        try:
            import winsound
            winsound.MessageBeep(-1)  # System default sound
        except:
            pass  # Ignore if sound playback fails

    def handle_existing_file(self, filepath):
        """Handle existing file with user prompt. Returns None if user cancels."""
        if not os.path.exists(filepath):
            return filepath
            
        response = messagebox.askyesnocancel(
            "File Already Exists",
            f"The file '{os.path.basename(filepath)}' already exists.\n\n"
            "Yes: Create new file with unique name\n"
            "No: Overwrite existing file\n"
            "Cancel: Abort operation",
            icon="warning"
        )
        
        if response is None:  # Cancel
            return None
        elif response:  # Yes - Create new
            return self.get_unique_filename(filepath, "")
        else:  # No - Overwrite
            return filepath
            
    def extract_audio(self, video_path):
        temp_base = video_path.rsplit(".", 1)[0] + "_temp_audio.mp3"
        audio_path = self.get_unique_filename(temp_base, "")
        self.update_progress(0.1, "Extracting audio from video...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        return audio_path
        
    def split_audio(self, audio_path, chunk_duration=30):
        """Split audio into chunks for parallel processing."""
        import soundfile as sf
        
        # Read audio file
        audio, sample_rate = sf.read(audio_path)
        duration = len(audio) / sample_rate
        
        # Calculate optimal chunk size based on file duration and worker count
        n_workers = get_optimal_workers()
        chunks_per_worker = 2  # Each worker gets 2 chunks for better load balancing
        total_chunks = n_workers * chunks_per_worker
        
        # Ensure chunks aren't too small (minimum 30 seconds)
        chunk_duration = max(30, min(duration / total_chunks, 300))
        
        chunks = []
        chunk_times = []
        
        # Split audio into chunks
        chunk_samples = int(chunk_duration * sample_rate)
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            chunk_path = f"{audio_path}_chunk_{i}.wav"
            sf.write(chunk_path, chunk, sample_rate)
            chunks.append(chunk_path)
            chunk_times.append(i / sample_rate)
            
        return chunks, chunk_times

    def load_model(self, model_name, device):
        """Load model with proper caching support."""
        is_cached = self.model_cache.is_model_cached(model_name)
        cache_path = self.model_cache.get_cache_path(model_name)
        
        if not is_cached:
            self.update_progress(0.1, 
                f"Downloading {model_name} model...", 
                f"Downloading {self.model_cache.get_model_size(model_name)}MB"
            )
            
        try:
            # Ensure we're using the correct device
            if device == "cuda" and not self.cuda_info['available']:
                device = "cpu"
                self.device_var.set("cpu")
                self.update_device_selection("cpu")
                
            model = whisper.load_model(
                model_name,
                device=device,
                download_root=cache_path
            )
            
            # Add to cache after successful load
            if not is_cached and self.model_cache.verify_model_files(model_name):
                self.model_cache.add_to_cache(model_name)
                self.update_model_info(model_name)
            
            return model
            
        except Exception as e:
            messagebox.showerror(
                "Model Loading Error",
                f"Error loading {model_name} model: {str(e)}\n"
                "The model will be downloaded again."
            )
            # Clear cache entry if loading fails
            if model_name in self.model_cache.cache_info:
                del self.model_cache.cache_info[model_name]
                self.model_cache.save_cache_info()
            raise

    def merge_similar_segments(self, segments, max_time_diff=0.5):
        """
        Merge segments that have identical text and are very close in time.
        max_time_diff: maximum time difference in seconds between segments to consider merging
        """
        if not segments:
            return []
            
        merged = []
        current = None
        
        for segment in segments:
            if current is None:
                current = segment.copy()
                continue
                
            # Check if segments have identical text and are close in time
            time_diff = float(segment["start"]) - float(current["end"])
            if (segment["text"].strip() == current["text"].strip() and 
                time_diff < max_time_diff and time_diff >= 0):
                # Merge by extending end time
                current["end"] = segment["end"]
            else:
                merged.append(current)
                current = segment.copy()
        
        if current is not None:
            merged.append(current)
            
        return merged

    def convert_to_simplified_chinese(self, text):
        """Convert traditional Chinese characters to simplified Chinese."""
        try:
            # Initialize the converter
            converter = opencc.OpenCC('t2s')
            # Convert the text
            return converter.convert(text)
        except Exception as e:
            # If conversion fails, return original text
            print(f"Warning: Chinese conversion failed: {str(e)}")
            return text

    def process_segments(self, segments):
        """Process segments to handle duplicates and ensure proper text formatting."""
        # First merge similar segments
        merged_segments = self.merge_similar_segments(segments)
        
        # Process text and create clean segments
        processed_segments = []
        seen_texts = set()  # Track unique texts within a very short time window
        current_time = 0
        time_window = 2.0  # 2 second window to check for duplicates
        
        for segment in merged_segments:
            # Reset seen_texts if we've moved beyond the time window
            if float(segment["start"]) - current_time > time_window:
                seen_texts.clear()
                current_time = float(segment["start"])
            
            # Convert Chinese characters to simplified
            text = self.convert_to_simplified_chinese(segment["text"].strip())
            
            # Skip if this exact text was seen very recently
            if text in seen_texts:
                continue
                
            seen_texts.add(text)
            # Update the segment with converted text
            segment["text"] = text
            processed_segments.append(segment)
        
        return processed_segments

    def transcribe(self):
        try:
            if not self.file_paths:
                messagebox.showerror("Error", "Please select at least one file!")
                return
            
            # Check if opencc is installed
            try:
                import opencc
            except ImportError:
                messagebox.showwarning(
                    "Missing Dependency",
                    "The opencc library is not installed. Chinese character conversion will not be available.\n\n"
                    "To install it, run:\n"
                    "pip install opencc-python-reimplemented"
                )
            
            self.is_processing = True
            self.transcribe_button.configure(state="disabled")
            self.processed_files = set()  # Track processed files
            
            # Initialize batch progress
            self.update_batch_status()
            
            for file_path in self.file_paths:
                try:
                    self.start_time = time.time()
                    
                    # Prepare output paths
                    model_name = self.model_var.get()
                    base_output = file_path.rsplit(".", 1)[0]
                    srt_path = f"{base_output}_transcript_{model_name}.srt"
                    txt_path = f"{base_output}_transcript_{model_name}.txt"
                    
                    # Check if output files exist
                    srt_path = self.handle_existing_file(srt_path)
                    if srt_path is None:
                        continue  # Skip this file if user cancelled
                    txt_path = self.handle_existing_file(txt_path)
                    if txt_path is None:
                        continue  # Skip this file if user cancelled
                    
                    # Update status for current file
                    self.update_progress(0.1, f"Processing: {os.path.basename(file_path)}", "")
                    
                    # Initialize monitoring variables
                    start_memory = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
                    peak_memory = start_memory
                    device = self.device_var.get()
                    n_workers = get_optimal_workers() if device == "cpu" else None
                    
                    def update_peak_memory():
                        nonlocal peak_memory
                        current_memory = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
                        peak_memory = max(peak_memory, current_memory)
                    
                    # Always reload model when transcribing to ensure correct model is used
                    if device == "cuda" and not self.cuda_info['available']:
                        messagebox.showwarning("GPU Not Available", 
                            "CUDA (GPU) is not available. Falling back to CPU.\n\n"
                            "To use GPU acceleration, install CUDA:\n"
                            "1. Check if your GPU supports CUDA\n"
                            "2. Install NVIDIA GPU drivers\n"
                            "3. Install CUDA Toolkit\n"
                            "4. Reinstall PyTorch with CUDA support")
                        device = "cpu"
                        self.device_var.set("cpu")
                        n_workers = get_optimal_workers()
                    
                    # Clear existing model and load new one
                    if hasattr(self, 'model') and self.model is not None:
                        del self.model
                        torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                    
                    self.model = self.load_model(model_name, device)
                    update_peak_memory()
                    
                    # Extract audio if needed
                    temp_audio_path = None
                    if file_path.lower().endswith(('.mp4', '.avi', '.mkv')):
                        self.update_progress(0.2, "Extracting audio...", "This may take a few minutes")
                        temp_audio_path = self.extract_audio(file_path)
                        audio_path = temp_audio_path
                    else:
                        audio_path = file_path

                    # Process based on device type
                    if device == "cpu":
                        # CPU Processing code...
                        self.update_progress(0.3, "Preparing audio chunks...", "Splitting audio for parallel processing")
                        chunks, chunk_times = self.split_audio(audio_path)
                        
                        cpu_percent = psutil.cpu_percent(interval=1)
                        memory_percent = psutil.virtual_memory().percent
                        
                        self.update_progress(0.4, 
                            f"Starting transcription with {n_workers} CPU cores...\n"
                            f"CPU Usage: {cpu_percent}%\n"
                            f"Memory Usage: {memory_percent}%",
                            "This may take several minutes")
                        
                        # Process chunks in parallel
                        results = []
                        with ProcessPoolExecutor(max_workers=n_workers) as executor:
                            futures = [executor.submit(transcribe_segment, (self.model, chunk, start_time, device))
                                      for chunk, start_time in zip(chunks, chunk_times)]
                            
                            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                                start_time, result = future.result()
                                results.append((start_time, result))
                                progress = 0.4 + (0.5 * (i + 1) / len(chunks))
                                
                                # Update progress with resource usage
                                current_cpu = psutil.cpu_percent()
                                current_memory = psutil.virtual_memory().percent
                                self.update_progress(progress, 
                                    f"Processing chunk {i+1}/{len(chunks)}\n"
                                    f"CPU Usage: {current_cpu}%\n"
                                    f"Memory Usage: {current_memory}%")
                        
                        # Sort and combine results
                        results.sort(key=lambda x: x[0])
                        
                        # Clean up temporary chunk files
                        for chunk in chunks:
                            try:
                                os.remove(chunk)
                            except:
                                pass
                        
                        # Combine results
                        combined_result = {
                            "text": "",
                            "segments": []
                        }
                        
                        for _, chunk_result in results:
                            if isinstance(chunk_result, str):  # Error occurred
                                raise Exception(f"Error in chunk processing: {chunk_result}")
                            combined_result["text"] += chunk_result["text"] + "\n"
                            combined_result["segments"].extend(chunk_result["segments"])
                        
                        result = combined_result
                    else:
                        # GPU Processing
                        self.update_progress(0.3, "Transcribing audio with GPU...", "This may take several minutes")
                        
                        # Get initial GPU memory usage
                        torch.cuda.reset_peak_memory_stats()  # Reset peak stats
                        initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
                        
                        self.update_progress(0.4, 
                            f"Processing with GPU: {self.cuda_info['device_name']}\n"
                            f"Initial GPU Memory: {initial_gpu_memory:.2f}GB",
                            "This may take several minutes")
                        
                        result = self.model.transcribe(
                            audio_path,
                            language=None,
                            task="transcribe"
                        )
                        
                        # Get peak GPU memory usage
                        gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB
                        gpu_memory_current = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
                        update_peak_memory()  # Update system memory peak
                    
                    # Calculate memory usage
                    final_memory = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
                    memory_increase = peak_memory - start_memory
                    
                    # Generate outputs
                    self.update_progress(0.9, "Saving transcripts...", "Almost done")
                    
                    try:
                        # Process segments to remove duplicates
                        processed_segments = self.process_segments(result["segments"])
                        
                        # Save SRT format first
                        self.update_progress(0.92, "Generating SRT file...", "")
                        full_text = []  # Collect text for plain format
                        
                        with open(srt_path, "w", encoding="utf-8") as f:
                            for i, segment in enumerate(processed_segments, 1):
                                start_time = self.format_srt_time(float(segment["start"]))
                                end_time = self.format_srt_time(float(segment["end"]))
                                text = segment["text"].strip()
                                full_text.append(text)  # Collect text
                                
                                # Write SRT entry
                                f.write(f"{i}\n")  # Subtitle number
                                f.write(f"{start_time} --> {end_time}\n")  # Timestamp range
                                f.write(f"{text}\n\n")  # Text and blank line
                                
                                # Update progress for large files
                                if i % 100 == 0:  # Update every 100 segments
                                    self.update_progress(0.92 + (0.04 * i/len(processed_segments)),
                                        f"Writing SRT file... ({i}/{len(processed_segments)} segments)")
                        
                        # Save plain text format with proper line breaks
                        self.update_progress(0.96, "Generating text file...", "")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            # Join with double newlines to ensure proper spacing
                            f.write("\n".join(full_text))
                        
                        # Clear memory
                        full_text.clear()
                        del full_text
                        
                    except Exception as e:
                        self.update_progress(0.9, f"Error saving transcripts: {str(e)}", "")
                        # Try to clean up partial files
                        for file_path in [srt_path, txt_path]:
                            try:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            except:
                                pass
                        raise  # Re-raise the exception to be caught by outer try-except
                    
                    # Clear result from memory after successful file writing
                    result.clear()  # Clear the dictionary
                    del result
                    
                    # Clean up temporary audio if needed
                    if temp_audio_path:
                        try:
                            os.remove(temp_audio_path)
                        except:
                            pass
                    
                    elapsed_time = time.time() - self.start_time
                    time_str = self.format_time(elapsed_time)
                    
                    # After successful processing, add to processed files
                    self.processed_files.add(file_path)
                    self.update_batch_status()
                    
                    # Update status to show completion of current file
                    self.update_progress(1.0, f"Completed: {os.path.basename(file_path)}", f"Time: {time_str}")
                    
                except Exception as e:
                    self.update_progress(0, f"Error processing {os.path.basename(file_path)}: {str(e)}", "")
                    continue  # Continue with next file even if current one fails
            
            # All files processed
            self.update_progress(1.0, "All files processed!", "")
            self.play_completion_sound()  # Play completion sound
            
        except Exception as e:
            self.update_progress(0, f"Batch processing error: {str(e)}", "")
        finally:
            self.is_processing = False
            self.transcribe_button.configure(state="normal")
            self.update_batch_status()

    def start_transcription(self):
        if not self.is_processing:
            thread = threading.Thread(target=self.transcribe)
            thread.start()

    def get_unique_filename(self, base_path, suffix):
        """Generate a unique filename by adding a number if file exists."""
        directory = os.path.dirname(base_path)
        filename = os.path.basename(base_path)
        name, ext = os.path.splitext(filename)
        counter = 1
        new_path = os.path.join(directory, f"{name}{suffix}{ext}")
        
        while os.path.exists(new_path):
            new_path = os.path.join(directory, f"{name}{suffix}_{counter}{ext}")
            counter += 1
        
        return new_path

    def show_settings(self):
        # Check if settings window already exists
        if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
            self.settings_window.lift()  # Bring existing window to front
            return
        
        # Create new settings window
        self.settings_window = ctk.CTkToplevel(self.window)
        self.settings_window.title("Settings")
        self.settings_window.minsize(500, 650)  # Increased minimum height to ensure all content is visible
        self.settings_window.geometry("500x650")  # Set default size to match minimum
        
        # Make window modal
        self.settings_window.transient(self.window)
        self.settings_window.grab_set()
        
        # Center the window
        self.settings_window.update_idletasks()
        x = self.window.winfo_x() + (self.window.winfo_width() - self.settings_window.winfo_width()) // 2
        y = self.window.winfo_y() + (self.window.winfo_height() - self.settings_window.winfo_height()) // 2
        self.settings_window.geometry(f"+{x}+{y}")
        
        # Main container frame to hold all sections
        main_container = ctk.CTkFrame(self.settings_window)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Section 1: Cache location frame (fixed height)
        cache_frame = ctk.CTkFrame(main_container)
        cache_frame.pack(fill="x", pady=(0, 20))
        
        cache_label = ctk.CTkLabel(
            cache_frame,
            text="Cache Location:",
            font=("Helvetica", 12, "bold")
        )
        cache_label.pack(pady=5)
        
        current_cache = ctk.CTkLabel(
            cache_frame,
            text=self.model_cache.cache_dir,
            font=("Helvetica", 10)
        )
        current_cache.pack(pady=5)
        
        # Button frame for cache controls
        cache_buttons_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
        cache_buttons_frame.pack(pady=10)
        
        def change_cache():
            new_location = filedialog.askdirectory(
                title="Select Cache Directory",
                initialdir=self.model_cache.cache_dir
            )
            if new_location:
                try:
                    self.model_cache.change_cache_location(new_location)
                    current_cache.configure(text=new_location)
                    self.cache_label.configure(text=f"Cache: {os.path.abspath(new_location)}")
                    messagebox.showinfo("Success", "Cache location changed successfully!")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to change cache location: {str(e)}")
        
        def open_cache():
            try:
                if os.path.exists(self.model_cache.cache_dir):
                    os.startfile(self.model_cache.cache_dir) if os.name == 'nt' else os.system(f'xdg-open "{self.model_cache.cache_dir}"')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open cache folder: {str(e)}")
        
        def clear_cache():
            # Calculate total cache size before clearing
            cache_size = sum(info["size_mb"] for info in self.model_cache.cache_info.values())
            cached_models = list(self.model_cache.cache_info.keys())
            
            warning_message = (
                "⚠️ Warning: You are about to clear all cached models!\n\n"
                f"Currently cached models ({len(cached_models)}):\n"
                f"{', '.join(cached_models)}\n\n"
                f"Total cache size: {cache_size:.1f}MB\n\n"
                "If you clear the cache:\n"
                "• All downloaded model files will be deleted\n"
                "• You will need to download models again when using them\n"
                "• This may require significant download time and bandwidth\n\n"
                "Are you sure you want to clear the cache?"
            )
            
            if messagebox.askyesno(
                "Clear Cache - Confirmation Required",
                warning_message,
                icon="warning"
            ):
                try:
                    self.model_cache.clear_cache()
                    messagebox.showinfo(
                        "Success",
                        "Cache cleared successfully!\n\n"
                        "Models will be downloaded automatically when needed."
                    )
                    self.update_model_info(self.model_var.get())
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to clear cache: {str(e)}")
        
        # Cache control buttons
        change_button = ctk.CTkButton(
            cache_buttons_frame,
            text="Change Location",
            command=change_cache,
            width=120
        )
        change_button.pack(side="left", padx=5)
        
        open_button = ctk.CTkButton(
            cache_buttons_frame,
            text="Open Folder",
            command=open_cache,
            width=120
        )
        open_button.pack(side="left", padx=5)
        
        clear_button = ctk.CTkButton(
            cache_buttons_frame,
            text="Clear Cache",
            command=clear_cache,
            width=120
        )
        clear_button.pack(side="left", padx=5)
        
        # Section 2: Storage requirements (fixed height)
        storage_frame = ctk.CTkFrame(main_container)
        storage_frame.pack(fill="x", pady=(0, 20))
        
        storage_label = ctk.CTkLabel(
            storage_frame,
            text="Storage Requirements:",
            font=("Helvetica", 12, "bold")
        )
        storage_label.pack(pady=5)
        
        storage_info = (
            "Minimum free space needed:\n"
            "• tiny: 75MB\n"
            "• base: 142MB\n"
            "• small: 461MB\n"
            "• medium: 1.42GB\n"
            "• large: 2.87GB\n\n"
            "Recommended: At least 3.5GB free space for all models\n"
            "Additional space needed for temporary files during processing"
        )
        
        storage_info_label = ctk.CTkLabel(
            storage_frame,
            text=storage_info,
            font=("Helvetica", 10),
            justify="left"
        )
        storage_info_label.pack(pady=5)
        
        # Section 3: Cache info (fixed height)
        info_frame = ctk.CTkFrame(main_container)
        info_frame.pack(fill="x", pady=(0, 20))
        
        cache_size = sum(info["size_mb"] for info in self.model_cache.cache_info.values())
        info_text = f"Cached Models: {len(self.model_cache.cache_info)}\n"
        info_text += f"Total Cache Size: {cache_size:.1f}MB\n\n"
        info_text += "Cached Models:\n"
        for model, info in self.model_cache.cache_info.items():
            info_text += f"- {model}: {info['size_mb']}MB (cached on {info['cached_date']})\n"
        
        info_label = ctk.CTkLabel(
            info_frame,
            text=info_text,
            font=("Helvetica", 10),
            justify="left",
            wraplength=460  # Set wraplength to prevent text from being cut off
        )
        info_label.pack(pady=10, padx=10)
        
        # Section 4: OK button (fixed position at bottom)
        button_frame = ctk.CTkFrame(main_container, fg_color="transparent", height=50)
        button_frame.pack(fill="x", pady=(0, 10))
        button_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        ok_button = ctk.CTkButton(
            button_frame,
            text="OK",
            command=self.settings_window.destroy,
            width=120
        )
        ok_button.pack(expand=True)
        
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TranscriberApp()
    app.run() 