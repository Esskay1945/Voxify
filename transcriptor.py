"""
Voxify - Professional Medical Voice Transcription
One-time training setup with elegant batch processing

Requirements:
pip install customtkinter pillow whisper torch torchaudio soundfile numpy scipy

Usage:
python voxify.py
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
from datetime import datetime
import whisper
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import re
from collections import Counter
import pickle

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class Voxify(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Voxify")
        self.geometry("1200x800")
        
        # Initialize
        self.model = None
        self.audio_queue = []
        self.processing = False
        
        # Medical terminology database
        self.medical_terms = set()
        self.medical_phrases = set()
        self.trained = False
        
        # Load existing training if available
        self.load_medical_database()
        
        self.setup_ui()
        
    def load_medical_database(self):
        """Load pre-trained medical terminology database"""
        db_file = "voxify_medical_db.pkl"
        
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.medical_terms = data.get('terms', set())
                    self.medical_phrases = data.get('phrases', set())
                    self.trained = True
            except Exception as e:
                print(f"Error loading database: {e}")
                self.trained = False
        else:
            self.trained = False
    
    def save_medical_database(self):
        """Save medical terminology database"""
        db_file = "voxify_medical_db.pkl"
        
        try:
            data = {
                'terms': self.medical_terms,
                'phrases': self.medical_phrases
            }
            with open(db_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving database: {e}")
            return False
    
    def setup_ui(self):
        # Main container with gradient effect
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Header section
        header_frame = ctk.CTkFrame(
            main_frame, 
            fg_color=("#2b2d42", "#1a1b26"),
            corner_radius=20,
            border_width=2,
            border_color=("#8d99ae", "#4a5568")
        )
        header_frame.pack(fill="x", pady=(0, 25))
        
        # Logo and title
        title_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_container.pack(pady=30)
        
        ctk.CTkLabel(
            title_container,
            text="V O X I F Y",
            font=ctk.CTkFont(size=48, weight="bold", family="Helvetica"),
            text_color=("#06ffa5", "#06ffa5")
        ).pack()
        
        ctk.CTkLabel(
            title_container,
            text="Medical Transcription Intelligence",
            font=ctk.CTkFont(size=14, family="Helvetica"),
            text_color=("#8d99ae", "#718096")
        ).pack(pady=(5, 0))
        
        # Training status indicator
        status_frame = ctk.CTkFrame(
            main_frame,
            fg_color=("#2b2d42", "#1a1b26"),
            corner_radius=15,
            border_width=2,
            border_color=("#8d99ae", "#4a5568")
        )
        status_frame.pack(fill="x", pady=(0, 20))
        
        status_inner = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_inner.pack(pady=20, padx=25)
        
        if self.trained:
            status_indicator = "●"
            status_color = "#06ffa5"
            status_text = f"Medical Database Active • {len(self.medical_terms)} terms • {len(self.medical_phrases)} phrases"
        else:
            status_indicator = "○"
            status_color = "#ef233c"
            status_text = "Medical Database Not Configured"
        
        status_row = ctk.CTkFrame(status_inner, fg_color="transparent")
        status_row.pack()
        
        ctk.CTkLabel(
            status_row,
            text=status_indicator,
            font=ctk.CTkFont(size=24),
            text_color=status_color
        ).pack(side="left", padx=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            status_row,
            text=status_text,
            font=ctk.CTkFont(size=13, family="Helvetica"),
            text_color=("#e0e1dd", "#a0aec0")
        )
        self.status_label.pack(side="left")
        
        if not self.trained:
            ctk.CTkButton(
                status_row,
                text="⚡ Configure Now",
                command=self.initial_training_setup,
                width=140,
                height=32,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=("#06ffa5", "#059669"),
                hover_color=("#05d98c", "#047857"),
                text_color="#000000",
                corner_radius=8
            ).pack(side="left", padx=(20, 0))
        
        # File processing section
        processing_frame = ctk.CTkFrame(
            main_frame,
            fg_color=("#2b2d42", "#1a1b26"),
            corner_radius=20,
            border_width=2,
            border_color=("#8d99ae", "#4a5568")
        )
        processing_frame.pack(fill="both", expand=True)
        
        # Section title
        ctk.CTkLabel(
            processing_frame,
            text="BATCH PROCESSING",
            font=ctk.CTkFont(size=16, weight="bold", family="Helvetica"),
            text_color=("#8d99ae", "#718096")
        ).pack(pady=(25, 15))
        
        # Queue display
        queue_container = ctk.CTkFrame(processing_frame, fg_color="transparent")
        queue_container.pack(fill="both", expand=True, padx=25, pady=(0, 20))
        
        queue_header = ctk.CTkFrame(queue_container, fg_color="transparent")
        queue_header.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(
            queue_header,
            text="Queue",
            font=ctk.CTkFont(size=13, family="Helvetica"),
            text_color=("#e0e1dd", "#a0aec0")
        ).pack(side="left")
        
        self.queue_count_label = ctk.CTkLabel(
            queue_header,
            text="0 files",
            font=ctk.CTkFont(size=13, weight="bold", family="Helvetica"),
            text_color=("#06ffa5", "#06ffa5")
        )
        self.queue_count_label.pack(side="right")
        
        # Queue textbox
        self.queue_textbox = ctk.CTkTextbox(
            queue_container,
            font=ctk.CTkFont(size=11, family="Courier"),
            wrap="none",
            fg_color=("#1a1b26", "#0d0e14"),
            corner_radius=12,
            border_width=1,
            border_color=("#4a5568", "#2d3748"),
            height=200
        )
        self.queue_textbox.pack(fill="both", expand=True)
        
        # Control buttons
        button_container = ctk.CTkFrame(processing_frame, fg_color="transparent")
        button_container.pack(pady=25)
        
        self.add_files_btn = ctk.CTkButton(
            button_container,
            text="Add Files",
            command=self.add_files,
            width=140,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#8d99ae", "#4a5568"),
            hover_color=("#6c757d", "#2d3748"),
            corner_radius=10
        )
        self.add_files_btn.pack(side="left", padx=8)
        
        self.add_folder_btn = ctk.CTkButton(
            button_container,
            text="Add Folder",
            command=self.add_folder,
            width=140,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#8d99ae", "#4a5568"),
            hover_color=("#6c757d", "#2d3748"),
            corner_radius=10
        )
        self.add_folder_btn.pack(side="left", padx=8)
        
        self.process_btn = ctk.CTkButton(
            button_container,
            text="Process All",
            command=self.start_processing,
            width=140,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#06ffa5", "#059669"),
            hover_color=("#05d98c", "#047857"),
            text_color="#000000",
            corner_radius=10,
            state="disabled"
        )
        self.process_btn.pack(side="left", padx=8)
        
        self.clear_btn = ctk.CTkButton(
            button_container,
            text="Clear",
            command=self.clear_queue,
            width=100,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#ef233c", "#dc2626"),
            hover_color=("#d90429", "#b91c1c"),
            corner_radius=10,
            state="disabled"
        )
        self.clear_btn.pack(side="left", padx=8)
        
        # Progress section
        progress_container = ctk.CTkFrame(processing_frame, fg_color="transparent")
        progress_container.pack(fill="x", padx=25, pady=(0, 25))
        
        self.progress_label = ctk.CTkLabel(
            progress_container,
            text="Ready",
            font=ctk.CTkFont(size=12, family="Helvetica"),
            text_color=("#8d99ae", "#718096")
        )
        self.progress_label.pack(pady=(10, 8))
        
        self.progress_bar = ctk.CTkProgressBar(
            progress_container,
            width=600,
            height=8,
            corner_radius=4,
            fg_color=("#1a1b26", "#0d0e14"),
            progress_color=("#06ffa5", "#059669")
        )
        self.progress_bar.pack()
        self.progress_bar.set(0)
        
        self.progress_detail = ctk.CTkLabel(
            progress_container,
            text="0 / 0",
            font=ctk.CTkFont(size=11, family="Courier"),
            text_color=("#6c757d", "#4a5568")
        )
        self.progress_detail.pack(pady=(8, 0))
    
    def initial_training_setup(self):
        """One-time setup: Load medical reference documents"""
        folder = filedialog.askdirectory(
            title="Select Folder with Medical Reference Documents (Word/Text files)"
        )
        
        if not folder:
            return
        
        # Show processing dialog
        progress_window = ctk.CTkToplevel(self)
        progress_window.title("Training in Progress")
        progress_window.geometry("400x150")
        progress_window.transient(self)
        progress_window.grab_set()
        
        ctk.CTkLabel(
            progress_window,
            text="Analyzing Medical Documents...",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=20)
        
        progress = ctk.CTkProgressBar(progress_window, width=350)
        progress.pack(pady=10)
        progress.set(0)
        
        status_label = ctk.CTkLabel(
            progress_window,
            text="Processing...",
            font=ctk.CTkFont(size=11)
        )
        status_label.pack(pady=10)
        
        def train_worker():
            try:
                # Find all text/doc files
                text_files = list(Path(folder).glob("*.txt"))
                text_files.extend(Path(folder).glob("*.docx"))
                text_files.extend(Path(folder).glob("*.doc"))
                
                if not text_files:
                    self.after(0, lambda: messagebox.showerror(
                        "Error", 
                        "No text or Word documents found in selected folder."
                    ))
                    progress_window.destroy()
                    return
                
                all_words = []
                all_bigrams = []
                all_trigrams = []
                
                for idx, file_path in enumerate(text_files):
                    # Update progress
                    prog = (idx + 1) / len(text_files)
                    self.after(0, lambda p=prog: progress.set(p))
                    self.after(0, lambda f=file_path.name: status_label.configure(
                        text=f"Processing: {f}"
                    ))
                    
                    try:
                        # Read file (handle both txt and docx)
                        if file_path.suffix == '.txt':
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read()
                        else:
                            # For .doc/.docx, try to read as text
                            try:
                                from docx import Document
                                doc = Document(file_path)
                                text = '\n'.join([para.text for para in doc.paragraphs])
                            except:
                                # If python-docx not available, skip
                                continue
                        
                        # Tokenize
                        text = text.lower()
                        words = re.findall(r'\b[a-z]+\b', text)
                        all_words.extend(words)
                        
                        # Extract n-grams
                        for i in range(len(words) - 1):
                            all_bigrams.append(f"{words[i]} {words[i+1]}")
                        
                        for i in range(len(words) - 2):
                            all_trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
                    
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                
                # Analyze frequency
                word_freq = Counter(all_words)
                bigram_freq = Counter(all_bigrams)
                trigram_freq = Counter(all_trigrams)
                
                # Common filler words to exclude
                filler_words = {
                    'um', 'uh', 'like', 'you', 'know', 'mean', 'just', 'very',
                    'really', 'quite', 'sort', 'kind', 'basically', 'actually',
                    'literally', 'right', 'okay', 'well', 'gonna', 'the', 'and',
                    'for', 'are', 'this', 'that', 'with', 'from', 'have', 'been'
                }
                
                # Extract medical terms (high frequency, not filler)
                self.medical_terms = {
                    word for word, count in word_freq.items()
                    if count >= 3 and len(word) > 3 and word not in filler_words
                }
                
                self.medical_phrases = {
                    phrase for phrase, count in bigram_freq.items() if count >= 2
                }
                
                self.medical_phrases.update({
                    phrase for phrase, count in trigram_freq.items() if count >= 2
                })
                
                # Save database
                if self.save_medical_database():
                    self.trained = True
                    
                    self.after(0, lambda: self.status_label.configure(
                        text=f"Medical Database Active • {len(self.medical_terms)} terms • {len(self.medical_phrases)} phrases"
                    ))
                    
                    self.after(0, progress_window.destroy)
                    self.after(0, lambda: messagebox.showinfo(
                        "Training Complete",
                        f"Successfully trained on {len(text_files)} documents!\n\n"
                        f"Medical Terms: {len(self.medical_terms)}\n"
                        f"Medical Phrases: {len(self.medical_phrases)}\n\n"
                        f"Database saved. You won't need to do this again."
                    ))
                else:
                    self.after(0, progress_window.destroy)
                    self.after(0, lambda: messagebox.showerror(
                        "Error",
                        "Failed to save medical database."
                    ))
            
            except Exception as e:
                self.after(0, progress_window.destroy)
                self.after(0, lambda: messagebox.showerror(
                    "Error",
                    f"Training failed: {str(e)}"
                ))
        
        thread = threading.Thread(target=train_worker, daemon=True)
        thread.start()
    
    def add_files(self):
        """Add audio files via file dialog"""
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg *.wma"),
                ("All Files", "*.*")
            ]
        )
        
        if files:
            self.add_to_queue(list(files))
    
    def add_folder(self):
        """Add all audio files from folder"""
        folder = filedialog.askdirectory(title="Select Folder with Audio Files")
        
        if folder:
            extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma']
            files = []
            
            for ext in extensions:
                files.extend(Path(folder).glob(f"*{ext}"))
            
            if files:
                self.add_to_queue([str(f) for f in files])
            else:
                messagebox.showwarning("No Files", "No audio files found in folder.")
    
    def add_to_queue(self, files):
        """Add files to processing queue"""
        added = 0
        
        for file_path in files:
            if file_path not in self.audio_queue:
                size = os.path.getsize(file_path) / (1024 * 1024)
                
                if size > 100:
                    messagebox.showwarning(
                        "File Too Large",
                        f"Skipping {os.path.basename(file_path)} (>100MB)"
                    )
                    continue
                
                self.audio_queue.append(file_path)
                added += 1
        
        if added > 0:
            self.update_queue_display()
            self.process_btn.configure(state="normal")
            self.clear_btn.configure(state="normal")
    
    def update_queue_display(self):
        """Update queue display"""
        self.queue_textbox.delete("1.0", "end")
        self.queue_count_label.configure(text=f"{len(self.audio_queue)} files")
        
        for i, file_path in enumerate(self.audio_queue, 1):
            name = os.path.basename(file_path)
            size = os.path.getsize(file_path) / (1024 * 1024)
            self.queue_textbox.insert("end", f"{i:2d}. {name:50s} {size:6.2f} MB\n")
    
    def clear_queue(self):
        """Clear queue"""
        if messagebox.askyesno("Confirm", "Clear all files from queue?"):
            self.audio_queue.clear()
            self.update_queue_display()
            self.process_btn.configure(state="disabled")
            self.clear_btn.configure(state="disabled")
    
    def clean_transcription(self, text):
        """Clean transcription using medical database"""
        if not self.trained:
            return text
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        cleaned = []
        
        filler_words = {
            'um', 'uh', 'like', 'you know', 'i mean', 'sort of', 'kind of',
            'basically', 'actually', 'literally', 'right', 'okay', 'well', 'just'
        }
        
        for sentence in sentences:
            words = sentence.lower().split()
            kept = []
            i = 0
            
            while i < len(words):
                word = words[i].strip('.,;!?')
                matched = False
                
                # Check trigrams
                if i < len(words) - 2:
                    tri = f"{word} {words[i+1].strip('.,;!?')} {words[i+2].strip('.,;!?')}"
                    if tri in self.medical_phrases:
                        kept.extend([word, words[i+1].strip('.,;!?'), words[i+2].strip('.,;!?')])
                        i += 3
                        matched = True
                        continue
                
                # Check bigrams
                if not matched and i < len(words) - 1:
                    bi = f"{word} {words[i+1].strip('.,;!?')}"
                    if bi in self.medical_terms or bi in self.medical_phrases:
                        kept.extend([word, words[i+1].strip('.,;!?')])
                        i += 2
                        matched = True
                        continue
                
                # Single word
                if not matched:
                    if word in self.medical_terms or word not in filler_words:
                        kept.append(word)
                    i += 1
            
            if kept:
                kept[0] = kept[0].capitalize()
                cleaned.append(' '.join(kept))
        
        return '. '.join(cleaned) + '.' if cleaned else text
    
    def start_processing(self):
        """Start batch processing"""
        if not self.audio_queue:
            messagebox.showerror("Error", "No files in queue.")
            return
        
        if not self.trained:
            if messagebox.askyesno(
                "No Training",
                "Medical database not configured. Transcriptions may contain filler words.\n\nContinue anyway?"
            ):
                pass
            else:
                return
        
        # Create output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"voxify_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.processing = True
        self.process_btn.configure(state="disabled")
        self.add_files_btn.configure(state="disabled")
        self.add_folder_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        
        def worker():
            total = len(self.audio_queue)
            
            # Load model
            self.update_progress("Loading AI model...", 0, total)
            if self.model is None:
                self.model = whisper.load_model("base")
            
            for idx, audio_file in enumerate(self.audio_queue):
                try:
                    filename = os.path.basename(audio_file)
                    name_only = os.path.splitext(filename)[0]
                    
                    self.update_progress(f"Transcribing: {filename}", idx, total)
                    
                    # Transcribe
                    result = self.model.transcribe(
                        audio_file,
                        language="en",
                        task="transcribe",
                        verbose=False
                    )
                    
                    raw = result["text"]
                    cleaned = self.clean_transcription(raw)
                    
                    # Save
                    output_file = os.path.join(output_dir, f"{name_only}.txt")
                    
                    content = f"""VOXIFY TRANSCRIPTION
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source: {filename}

{cleaned}

---
Medical AI Transcription by Voxify
"""
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                except Exception as e:
                    print(f"Error: {e}")
            
            self.update_progress("Complete", total, total)
            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Processed {total} files!\n\nSaved to: {output_dir}"
            ))
            
            self.after(0, self.reset_ui)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def update_progress(self, msg, current, total):
        """Update progress UI"""
        progress = current / total if total > 0 else 0
        
        self.after(0, lambda: self.progress_label.configure(text=msg))
        self.after(0, lambda: self.progress_bar.set(progress))
        self.after(0, lambda: self.progress_detail.configure(text=f"{current} / {total}"))
    
    def reset_ui(self):
        """Reset UI after processing"""
        self.processing = False
        self.audio_queue.clear()
        self.update_queue_display()
        self.process_btn.configure(state="disabled")
        self.add_files_btn.configure(state="normal")
        self.add_folder_btn.configure(state="normal")
        self.clear_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Ready")
        self.progress_detail.configure(text="0 / 0")


if __name__ == "__main__":
    app = Voxify()
    app.mainloop()
