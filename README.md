# EmotiApp - Asistente Emocional

**EmotiApp** es una aplicación web desarrollada con **FastAPI** y **HTML + Tailwind**, que permite analizar emociones y sentimientos a partir de texto y audio. Está pensada como una herramienta de apoyo emocional para estudiantes, psicólogos y tutores.

---

## 🚀 Funcionalidades

- 🤖 Chat interactivo con análisis emocional del texto (usando Gemini API)
- 🎙️ Análisis de emociones a partir de archivos de audio (WAV)
- 📊 Gráfica radar de emociones simuladas + espectrograma del audio
- 💬 Interfaz moderna con pestañas para navegación entre texto y audio

---

## 🛠️ Tecnologías usadas

- FastAPI
- Python
- HTML + Tailwind CSS
- Chart.js
- Lucide Icons
- Gemini API (Google AI)
- Matplotlib, NumPy, SciPy

---

## 📦 Instalación local (opcional)

```bash
git clone https://github.com/tu_usuario/emotiapp.git
cd emotiapp
pip install -r requirements.txt
uvicorn main:app --reload
