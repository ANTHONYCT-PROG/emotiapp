from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import httpx # For making HTTP requests to the Gemini API
import os
import json
import logging
import io
import base64

# For audio processing and spectrogram generation
from scipy import signal # For spectrogram calculation
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting spectrogram
import soundfile as sf # To handle WAV files correctly

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Standalone AI Multimedia Assistant",
    description="FastAPI serving both frontend and backend for a chatbot with text/audio analysis.",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Jinja2 Templates Configuration ---
templates = Jinja2Templates(directory="templates")

# --- Helper function to call Gemini API ---
async def call_gemini_api(payload: dict) -> str:
    """
    Makes an asynchronous POST request to the Gemini API and returns the text response.
    """
    api_key = "AIzaSyAhLXoAZRr3OlOI35aWk4NCSU2PQ1kG7cQ" # API key will be provided by Canvas or environment. DO NOT ADD your key here.
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            gemini_response = await client.post(
                api_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            gemini_response.raise_for_status()
            result = gemini_response.json()

        if result and result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            return result["candidates"][0]["content"]["parts"][0].get("text", "No se encontró texto de respuesta.") # Spanish
        else:
            logging.error(f"Unexpected Gemini API response structure: {json.dumps(result, indent=2)}")
            return "Lo siento, no pude generar una respuesta en este momento. (Respuesta de la API inesperada)" # Spanish

    except httpx.HTTPStatusError as e:
        logging.error(f"Gemini API HTTP Error: Status {e.response.status_code} - Body: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error de la IA: {e.response.text}") # Spanish
    except httpx.RequestError as e:
        logging.error(f"Gemini API Request Error: {e}")
        raise HTTPException(status_code=503, detail=f"No se pudo conectar al servicio de IA: {e}") # Spanish
    except Exception as e:
        logging.error(f"An unexpected error occurred during Gemini API call: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al llamar a Gemini: {e}") # Spanish

# --- Root Endpoint: Serves the HTML Frontend ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the main HTML application page.
    """
    logging.info("Serving index.html")
    return templates.TemplateResponse("index.html", {"request": request})

# --- Text Chat and Analysis Endpoint (MODIFIED for Spanish) ---
@app.post("/chat-and-analyze-text")
async def chat_and_analyze_text(request: Request):
    """
    Receives a user message and history, gets a chatbot response in Spanish,
    and analyzes the user message for emotions/sentiment in Spanish using Gemini.
    """
    try:
        data = await request.json()
        user_message = data.get("message")
        chat_history = data.get("history", [])

        if not user_message:
            raise HTTPException(status_code=400, detail="El campo de mensaje es requerido.") # Spanish

        logging.info(f"Received text message: '{user_message}'")

        # 1. Get Chatbot Response
        # Instruction for Spanish response added to the prompt
        spanish_instruction = "Responde en español. "
        
        # Prepend the instruction to the last user message for direct effect
        # Or, if you want a system-wide instruction for the conversation:
        # full_conversation_for_chat.insert(0, {"role": "system", "parts": [{"text": "Responde siempre en español."}]})
        
        # It's more effective to apply the language instruction at the end of the current user's turn
        # or as part of a system instruction. Let's add it to the user's message itself to be clear.
        # However, for a general chat, it's better to keep the history clean and add a system instruction
        # or a meta-prompt. For simplicity and direct control, we'll try prepending to the user's message parts.
        # A more robust way for persistent language would be to send it as a system prompt.
        
        # Let's adjust by adding a "system" role at the beginning of the conversation for language.
        # This approach is better for consistent multilingual behavior.
        system_instruction_for_chat = [{"role": "user", "parts": [{"text": "A partir de ahora, responde siempre en español."}]}] # Initial explicit instruction
        
        # If chat_history is empty, start with the instruction. Otherwise, prepend if not already present.
        if not chat_history:
            full_conversation_for_chat = system_instruction_for_chat + [{"role": "user", "parts": [{"text": user_message}]}]
        else:
            # Check if Spanish instruction is already in history, if not, add it.
            # This is a simplification; a real app might manage this with session state.
            has_spanish_instruction = any(
                "siempre en español" in part.get("text", "")
                for entry in chat_history
                for part in entry.get("parts", [])
                if entry.get("role") == "user"
            )
            
            if not has_spanish_instruction:
                full_conversation_for_chat = system_instruction_for_chat + chat_history + [{"role": "user", "parts": [{"text": user_message}]}]
            else:
                full_conversation_for_chat = chat_history + [{"role": "user", "parts": [{"text": user_message}]}]


        chat_payload = {
            "contents": full_conversation_for_chat,
            "generationConfig": {
                "responseMimeType": "text/plain",
            }
        }
        ai_chat_response = await call_gemini_api(chat_payload)
        logging.info(f"AI chat response: {ai_chat_response[:50]}...")

        # 2. Analyze User Message for Emotions/Sentiment (in Spanish)
        analysis_prompt = (
            f"Analiza el siguiente mensaje de texto para determinar su sentimiento dominante (por ejemplo, positivo, negativo, neutral) "
            f"y cualquier emoción evidente (por ejemplo, alegría, tristeza, ira, sorpresa, miedo). "
            f"Proporciona un resumen breve y conciso del análisis, en español.\n\n" # Explicitly request Spanish
            f"Mensaje: \"{user_message}\""
        )
        analysis_payload = {
            "contents": [{"role": "user", "parts": [{"text": analysis_prompt}]}],
            "generationConfig": {
                "responseMimeType": "text/plain",
            }
        }
        text_analysis_result = await call_gemini_api(analysis_payload)
        logging.info(f"Text analysis result: {text_analysis_result[:50]}...")

        return JSONResponse({
            "bot_response": ai_chat_response,
            "text_analysis": text_analysis_result
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.exception("Error in /chat-and-analyze-text endpoint:")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}") # Spanish

# --- Audio Analysis Endpoint (MODIFIED for Spanish emotion names) ---
@app.post("/analyze-audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """
    Receives a WAV audio file, performs simulated emotion analysis using numpy/scipy,
    and generates a spectrogram image using scipy.signal and matplotlib.
    """
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos de audio WAV.") # Spanish

    try:
        audio_data_bytes = await audio_file.read()
        logging.info(f"Received audio file: {audio_file.filename}, size: {len(audio_data_bytes)} bytes")

        with io.BytesIO(audio_data_bytes) as audio_io:
            y, sr = sf.read(audio_io)

        if y.ndim > 1:
            y = y.mean(axis=1)

        min_samples_for_analysis = 256
        if len(y) < min_samples_for_analysis:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo de audio es demasiado corto para un análisis significativo. Por favor, sube una grabación con al menos {min_samples_for_analysis} muestras (por ejemplo, unos pocos milisegundos a tasas de muestreo estándar)." # Spanish
            )

        # --- 1. Simulated Multi-Emotion Analysis (Using NumPy/SciPy) ---
        y_normalized = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

        intensity_level = np.mean(np.abs(y_normalized))
        zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(y_normalized)))) / (2 * len(y_normalized))
        
        f, t, Zxx = signal.stft(y_normalized, fs=sr, nperseg=512, noverlap=256)
        magnitudes = np.abs(Zxx)
        
        if magnitudes.sum() > 0:
            spectral_centroid_num = np.sum(f[:, np.newaxis] * magnitudes, axis=0)
            spectral_centroid_den = np.sum(magnitudes, axis=0)
            spectral_centroid = np.mean(spectral_centroid_num / spectral_centroid_den)
        else:
            spectral_centroid = 0

        # Translate emotion names to Spanish
        emotions = {
            "Alegre": 0.0, # Joyful
            "Enfadado": 0.0, # Angry
            "Triste": 0.0, # Sad
            "Neutral": 0.0, # Neutral
            "Sorprendido": 0.0, # Surprised
        }

        emotions["Alegre"] = np.clip(intensity_level * 2, 0.0, 1.0)
        emotions["Enfadado"] = np.clip(intensity_level * 1.5 - 0.2, 0.0, 1.0)
        emotions["Sorprendido"] = np.clip(intensity_level * 2.5 * (zero_crossing_rate * 5), 0.0, 1.0)

        emotions["Triste"] = np.clip(0.8 - intensity_level * 3, 0.0, 1.0)
        emotions["Neutral"] = np.clip(1.0 - abs(intensity_level - 0.2) * 3, 0.0, 1.0)

        emotions["Alegre"] = np.clip(emotions["Alegre"] + spectral_centroid / (sr / 4) * 0.5, 0.0, 1.0)
        emotions["Triste"] = np.clip(emotions["Triste"] + (1 - spectral_centroid / (sr / 4)) * 0.5, 0.0, 1.0)

        for key in emotions:
            emotions[key] = float(max(0.0, min(1.0, emotions[key])))

        logging.info(f"Simulated emotions: {emotions}")

        # --- 2. Spectrogram Generation ---
        plt.figure(figsize=(10, 4))
        f, t, Sxx = signal.spectrogram(y_normalized, fs=sr, nperseg=1024, noverlap=512)
        
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        plt.ylabel('Frecuencia [Hz]') # Spanish label
        plt.xlabel('Tiempo [seg]') # Spanish label
        plt.title('Espectrograma (SciPy)') # Spanish title
        plt.colorbar(label='Intensidad [dB]') # Spanish label
        plt.ylim([0, sr / 2])
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close()

        spectrogram_b64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.info("Spectrogram image generated and Base64 encoded.")

        return JSONResponse({
            "simulated_emotions": emotions,
            "spectrogram_image_b64": f"data:image/png;base64,{spectrogram_b64}"
        })

    except sf.LibsndfileError as e:
        logging.error(f"Soundfile error processing audio: {e}")
        raise HTTPException(status_code=422, detail="No se pudo procesar el archivo WAV. Asegúrate de que tenga un formato WAV válido.") # Spanish
    except Exception as e:
        logging.exception("Error in /analyze-audio endpoint:")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar el audio: {e}") # Spanish
