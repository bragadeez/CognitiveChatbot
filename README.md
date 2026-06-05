# Cognitive Chatbot: Personalized ML Education via VARK Adaptivity and RAG

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Publication](https://img.shields.io/badge/Springer-2025%20Accepted-orange.svg)](#)

Cognitive Chatbot is an intelligent, adaptive educational application designed to personalize Machine Learning tutoring based on a student's cognitive learning style. By leveraging the **VARK learning model** (Visual, Auditory, Reading/Writing, Kinesthetic), the system dynamically tailors its instructional formats (including mindmaps, curated videos, detailed text, and real-world analogies) to match the student's learning profile.

The system utilizes a Support Vector Machine (SVM) classifier to diagnose learning styles from a 15-question diagnostic questionnaire. Conversational questions are then routed through a **Retrieval-Augmented Generation (RAG)** pipeline powered by a local FAISS vector store and Groq-hosted LLaMA 3.3 (70B).

---

## Why the Project is Useful

Traditional chatbot tutors output uniform textual answers regardless of a student's optimal cognitive modality. Cognitive Chatbot bridges this gap by offering:
*   **VARK Personalization**: Classifies students into distinct learning modalities and shapes responses dynamically.
*   **Context Preservation**: Maintains multi-turn conversation memory (last 10 turns) across style changes.
*   **RAG over Authoritative Sources**: Grounds answers in course materials using vector embeddings, preventing AI hallucinations.
*   **Tactile User Interface**: Styled using a Wabi-Sabi paper design layout with custom SVG icons and a floating toast active-style notification.

---

## How Users Can Get Started

### Prerequisites
*   **Python**: Version 3.10 or higher.
*   **Node.js**: Version 18 or higher (with `npm`).
*   **API Key**: A valid Groq API Key ([Get one here](https://console.groq.com/)).

### 1. Installation
Clone the repository and navigate to the project root:
```bash
git clone https://github.com/bragadeez/CognitiveChatbot
cd CognitiveChatbot/Implementation
```

### 2. Configure Virtual Environment
Create and activate a Python virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python dependencies:
```bash
pip install -r backend/requirements.txt
```

### 3. Environment Variables
Copy the dotenv template:
```bash
# Windows (Cmd)
copy .env.example .env

# macOS / Linux / PowerShell
cp .env.example .env
```
Open `.env` and fill in your credentials:
```env
API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional Supabase variables (runs in-memory if empty)
SUPABASE_URL=
SUPABASE_ANON_KEY=
```
*Note: Diagnostic questionnaire models are pre-loaded from [models/](models/) and vector indices from [vectorstore/](vectorstore/) recursively.*

### 4. Client Build
Install the frontend node modules:
```bash
cd frontend
npm install
cd ..
```

### 5. Running Locally
Launch the backend and frontend in **two separate terminal windows** inside `Implementation/`:

*   **Terminal 1 (FastAPI Backend)**:
    ```bash
    # Activate virtual environment first
    uvicorn backend.main:app --reload --port 8000
    ```
*   **Terminal 2 (React Frontend)**:
    ```bash
    cd frontend
    npm run dev
    ```

Open your browser to **http://localhost:5173**. The client Vite proxy redirects requests from `/api/*` to the local FastAPI port `8000` automatically.

---

## Where Users Can Get Help

*   **API Docs**: When running the backend server locally, navigate to `http://localhost:8000/docs` to view the interactive OpenAPI/Swagger definitions.
*   **Issue Tracking**: Report bugs or suggest features on our repository's [GitHub Issues Tracker](https://github.com/bragadeez/CognitiveChatbot/issues).
*   **Supabase Schema Setup**: For configuring the Postgres database layer and cloud bucket storage, run the SQL script in [supabase_migration.sql](supabase_migration.sql).
*   **SVM Training**: Read the setup for retraining the VARK classifier model in [train_svm_model.py](train_svm_model.py).

---

## Who Maintains and Contributes

*   **Maintainer**: bragadeez
*   **Contributions**: We welcome bug fixes and features! Please read our [Contributing Guidelines](docs/CONTRIBUTING.md) to understand coding styles, git branch workflows, and PR expectations before opening a pull request.
*   **License**: Distributed under the MIT license. See [LICENSE](LICENSE) for details.
