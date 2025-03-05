import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai 
import speech_recognition as sr
from pydub import AudioSegment
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any
import tempfile
#from dotenv import load_dotenv

from pydub import AudioSegment
print(AudioSegment.converter)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Récupération de la clé API depuis les variables d'environnement
load_dotenv() 
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    logger.error("GOOGLE_API_KEY n'est pas définie dans les variables d'environnement")
    raise ValueError("GOOGLE_API_KEY est requise")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Transcription et Traduction Audio",
    description="Une API pour transcrire des fichiers audio et traduire le texte",
    version="1.0.0"
)
UPLOAD_DIR="/tmp"

#os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configuration CORS
origins = [
    "http://localhost:5173",
    "https://multilingualvoice.vercel.app",
    "https://multilingualvoice.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration de Gemini
try:
    genai.configure(api_key=google_api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini configuré avec succès")
except Exception as e:
    logger.error(f"Erreur lors de la configuration de Gemini: {str(e)}")
    raise

async def convert_to_wav(audio_file_path: str) -> str:
    """
    Convertit un fichier audio en format WAV.
    
    Args:
        audio_file_path: Chemin vers le fichier audio à convertir
        
    Returns:
        Chemin vers le fichier WAV converti
    """
    try:
        # Création d'un fichier temporaire pour le WAV
        fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # Conversion du fichier
        audio = AudioSegment.from_file(audio_file_path)
        audio=audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        logger.info(f"Fichier converti avec succès: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Erreur lors de la conversion audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de conversion audio: {str(e)}")

async def transcribe_audio(audio_file: str) -> str:
    """
    Transcrit un fichier audio en texte.
    
    Args:
        audio_file: Chemin vers le fichier audio à transcrire
        
    Returns:
        Texte transcrit
    """
    try:
        print("ICI -------", audio_file)
        # Vérification et conversion si nécessaire
        if not audio_file.lower().endswith(".wav"):
            audio_file = await convert_to_wav(audio_file)

        # Transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        transcript = recognizer.recognize_google(audio_data,language="en-US")
        logger.info("Transcription réussie")
        return transcript
    except sr.UnknownValueError:
        logger.warning("Impossible de comprendre l'audio")
        return "Désolé, je n'ai pas pu comprendre l'audio."
    except sr.RequestError as e:
        logger.error(f"Erreur de requête Google Speech Recognition: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Erreur de service Google Speech Recognition: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de transcription: {str(e)}")
    finally:
        # Nettoyage des fichiers temporaires si nécessaire
        if audio_file.startswith(tempfile.gettempdir()):
            try:
                os.remove(audio_file)
                logger.debug(f"Fichier temporaire supprimé: {audio_file}")
            except:
                pass

async def translate_text(text: str, target_language: str = "fr") -> str:
    """
    Traduit un texte vers la langue cible en utilisant Gemini.
    
    Args:
        text: Texte à traduire
        target_language: Code de langue cible (défaut: français)
        
    Returns:
        Texte traduit
    """
    try:
        #prompt = f"""Translate this into language corresponding to {target_language}.

         #Translation :{text}"""
        prompt = f"""Don't reason ,you are just a translator so translate always transcription of what user say
         Transcription 1:You are handsome
         translation 1: Tu es beau

         Transcription 2: Do you speak French?
         translation 2: Est-ce que vous parlez Français?

         Transcription 3: {text}
         translation 3:
           
           """
        response = llm.generate_content(prompt)
        translation = response.text
        print(prompt+translation)
        #translation ="test"
        logger.info(f"Traduction réussie vers {target_language}")
        return translation
    except Exception as e:
        logger.error(f"Erreur lors de la traduction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de traduction: {str(e)}")

@app.post("/transcribe/", response_model=Dict[str, Any])
async def transcribe_endpoint(file: UploadFile = File(...), target_language: str = "fr"):
    try:
        # Création d'un fichier temporaire
        temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Traitement du fichier
        transcript = await transcribe_audio(temp_file_path)
        translation = await translate_text(transcript, target_language)
        #translation ="test"
        
        return {
            "success": True,
            "transcription": transcript, 
            "translation": translation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")





@app.get("/health")
async def health_check():
    """
    Endpoint pour vérifier l'état de l'API.
    
    Returns:
        Statut de l'API
    """
    return {"status": "OK", "service": "transcription-api"}

# Lancer le serveur FastAPI
if __name__ == "__main__":
    import uvicorn
    
    # Configuration du serveur
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Démarrage du serveur sur {host}:{port}")
    uvicorn.run(app, host=host, port=port)