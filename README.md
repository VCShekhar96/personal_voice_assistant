# Voice Assistant

A Python-based voice assistant project that combines **speech recognition, text-to-speech, and command automation**.

## Overview

The project explores how voice input can be converted into actionable commands and how application responses can be returned through synthesized speech.

## Capabilities

The current project documentation describes support for:

- 🎙️ Speech recognition
- 🔊 Text-to-speech responses
- 🌐 Web navigation/search
- 🕒 Time and date queries
- 📧 Email automation
- 🧩 Extensible command handling

## Tech Stack

- Python
- SpeechRecognition
- pyttsx3
- webbrowser
- datetime
- os
- smtplib
- requests

## Getting Started

### Clone

```bash
git clone https://github.com/VCShekhar96/personal_voice_assistant.git
cd personal_voice_assistant
```

### Create a virtual environment

**Windows**

```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

Use the project's Python entry point:

```bash
python main.py
```

If the local implementation uses a different entry point, use that file instead.

## Security & Privacy

Voice assistants can process microphone input and may interact with external services.

- Keep API keys, passwords, email credentials, and tokens out of source control.
- Use environment variables or a local configuration file for secrets.
- Do not commit private recordings or personal data.
- Review external-service permissions before enabling automation.

## Future Improvements

- Add unit tests for command routing and integrations.
- Separate speech recognition, command handling, and response generation into modules.
- Add structured logging and error handling.
- Make integrations configurable rather than hard-coded.
- Add a clear configuration example for optional services.

## License

See the repository's license file for the authoritative licensing terms.
