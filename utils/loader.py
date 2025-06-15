from pypdf import PdfReader
import whisper
import tempfile
from moviepy import VideoFileClip

model = whisper.load_model("base")

def load_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def load_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file.read())
        tmp.flush()
        result = model.transcribe(tmp.name)
    return result["text"]

def load_video(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file.read())
        tmp.flush()
        clip = VideoFileClip(tmp.name)
        audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        clip.audio.write_audiofile(audio_path)
        result = model.transcribe(audio_path)
    return result["text"]