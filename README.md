# Voxify
🎙️ AI-powered medical transcription software with intelligent filler word removal. Train once on your medical documents, then enjoy clean, professional transcriptions forever. 100% local, HIPAA-friendly, built with Whisper AI.



# Voxify

**Professional Medical Voice Transcription with AI-Powered Intelligence**

Voxify is a desktop application that transcribes medical audio recordings with intelligent filler word removal and medical terminology preservation. Built with a one-time training setup, it learns from your medical reference documents to produce clean, professional transcriptions.

## ✨ Features

- **🎯 Intelligent Medical Transcription** - AI-powered transcription using OpenAI's Whisper model
- **🧠 One-Time Training** - Learns medical terminology from your reference documents
- **🚀 Batch Processing** - Process multiple audio files efficiently
- **🧹 Smart Cleaning** - Automatically removes filler words while preserving medical terms
- **📁 Flexible Input** - Add individual files or entire folders
- **💾 Persistent Database** - Training is saved and reused automatically
- **🎨 Modern UI** - Elegant dark-themed interface built with CustomTkinter

## 🎬 Demo

![Voxify Interface](https://via.placeholder.com/800x600?text=Voxify+Interface)

## 📋 Requirements

- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU optional (CPU works fine for base model)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/voxify.git
cd voxify
```

2. **Install dependencies**
```bash
pip install customtkinter pillow openai-whisper torch torchaudio soundfile numpy scipy python-docx
```

> **Note:** Whisper installation may take a few minutes as it downloads the AI model.

## 🚀 Quick Start

1. **Run Voxify**
```bash
python transcriptor.py
```

2. **First-Time Setup** (One-time only)
   - Click **"Configure Now"** 
   - Select a folder containing medical reference documents (Word/Text files)
   - Wait for training to complete (~1-2 minutes)
   - Your medical database is now saved permanently!

3. **Transcribe Audio**
   - Click **"Add Files"** or **"Add Folder"** to queue audio files
   - Click **"Process All"** to start transcription
   - Find your transcriptions in the generated `voxify_output_[timestamp]` folder

## 📂 Supported Formats

**Audio Input:**
- MP3, WAV, M4A, FLAC, OGG, WMA

**Training Documents:**
- TXT, DOC, DOCX

## 🧠 How Training Works

Voxify analyzes your medical reference documents to:
- Extract frequently used medical terms
- Identify medical phrases (bigrams and trigrams)
- Build a custom vocabulary specific to your practice
- Save this database locally as `voxify_medical_db.pkl`

The training happens **once** and is reused for all future transcriptions.

## 💡 Usage Tips

1. **Better Training = Better Results**
   - Use 5-10 representative medical documents
   - Include various note types (H&P, progress notes, discharge summaries)
   - More diverse terminology improves accuracy

2. **Audio Quality Matters**
   - Clear recordings produce better transcriptions
   - Minimize background noise
   - Files under 100MB process faster

3. **Batch Processing**
   - Queue multiple files for hands-free processing
   - Perfect for end-of-day transcription batches

## 📊 Output Format

Each transcription is saved as a formatted text file:

```
VOXIFY TRANSCRIPTION
Generated: 2024-01-15 14:30:45
Source: patient_note_001.mp3

[Clean transcription with medical terminology preserved and filler words removed]

---
Medical AI Transcription by Voxify
```

## 🔒 Privacy & Security

- **100% Local Processing** - All transcription happens on your machine
- **No Cloud Uploads** - Your audio and documents never leave your computer
- **HIPAA Friendly** - No data transmitted to external servers

## ⚙️ Configuration

Voxify stores its configuration in:
- `voxify_medical_db.pkl` - Trained medical terminology database

To retrain with new documents, simply delete this file and run the configuration setup again.

## 🐛 Troubleshooting

**"No module named 'whisper'"**
```bash
pip install openai-whisper
```

**"No module named 'docx'"**
```bash
pip install python-docx
```

**"Model loading takes too long"**
- First run downloads the Whisper model (~140MB)
- Subsequent runs load from cache and are much faster

**"Training found no terms"**
- Ensure your documents contain medical text
- Check that files are readable (not corrupted PDFs)
- Try adding more diverse documents

## 🛣️ Roadmap

- [ ] Support for additional Whisper models (small, medium, large)
- [ ] Speaker diarization for multi-speaker recordings
- [ ] Export to DOCX format
- [ ] Custom filler word lists
- [ ] Real-time transcription mode
- [ ] Integration with EHR systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [OpenAI Whisper](https://github.com/openai/whisper) for transcription
- UI powered by [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for healthcare professionals**
