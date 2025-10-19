Ikarus3D — Product Recommender

This project demonstrates a product recommender system built using:
- **Backend:** FastAPI (semantic search + Pinecone integration + GenAI & CV endpoints)
- **Frontend:** React (Vite) — interface for search, image upload, and generated descriptions
- **Modeling:** text+image embeddings, simple CV classifier (MLP / KNN), GenAI generator (Flan-T5)
- **Vector DB:** Pinecone (used for production-like vector storage; local file index also supported)
- **Notebooks:** Data Analytics + Model Training (with evaluations & comments)

Demonstration - ML based product Recommendation
![mlrecommender (1)](https://github.com/user-attachments/assets/2ab89e9e-19a0-4b4a-ba11-7279b8a3c1de)

Demonstration - CV based image search
![cvimg (1)](https://github.com/user-attachments/assets/8f6e9170-3c39-491d-82de-6e8448745ecf)

Demonstration - GenAi based creative description
<img width="1779" height="1158" alt="image" src="https://github.com/user-attachments/assets/b7640aa3-ffff-4357-abc9-8b29bb6c3774" />

Demonstration - Analytics 
![analyticsnew](https://github.com/user-attachments/assets/e848a450-7a81-44c0-a68c-1622efcc5a4c)


Quickstart (local dev)

> **Prereqs**: Python 3.10+, Node 16+, Git

### 1. Backend
```bash
# cd to backend
cd backend

# create virtualenv (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# start dev server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API endpoints :
GET /health — healthcheck
POST /recommend — query by text: {"prompt":"wooden chair","top_k":6}
POST /search/image?top_k=5 — multipart upload file
POST /gen/description — body: { "title":"...", "description":"...", "meta": {...} }
GET /analytics/summary — dataset analytics

### 2. Frontend
```bash
cd frontend
npm install
# set VITE_API_BASE in .env or export env var
npm run dev
```

### 3. Model Training Notebook and
### 4. Analytics notebook
Both Attached

### 5. Required Environment Variables
PINECONE_API_KEY=
PINECONE_ENV=
PINECONE_INDEX_NAME=
USE_PINECONE=true
VECTOR_DIM=512

### 6. requirements.txt (Dependencies - backend)
fastapi>=0.95.0
uvicorn[standard]>=0.22.0
python-multipart>=0.0.6
pydantic>=1.10.7
numpy>=1.24.0
pandas>=2.0.0
requests>=2.28.0
pillow>=9.0.0
scikit-learn>=1.2.0
joblib>=1.2.0
pinecone-client>=5.1.0
transformers>=4.35.0
torch>=2.0.0         # if you are using CPU-only, this will still install CPU wheel; remove if you use separate model host
tqdm>=4.65.0
ftfy>=6.1.1
safetensors>=0.3.0
accelerate>=0.20.0
typing-extensions>=4.5.0


### This project fulfills the following requirements:

Requirement	Implementation
1. Backend (FastAPI)	REST API with endpoints for recommendations, image-based search, NLP-based grouping, analytics, and GenAI description generation
2. Frontend (React + Vite)	Clean UI with product cards, search bar, image upload, and description generator
3. VectorDB (Pinecone)	Stores product embeddings (CLIP-based) for semantic retrieval
4. ML Models	KNN + MLP models trained on product embeddings for similarity and classification
5. NLP	CLIP text encoder + HuggingFace Transformers for text embeddings and prompt understanding
6. Computer Vision (CV)	CLIP image encoder + KNN/MLP classifier for image-category mapping
7. GenAI	FLAN-T5 (via HuggingFace Transformers) for generating creative product design descriptions
Integration Framework	LangChain used for GenAI pipeline orchestration and embedding-based query flow

### 🧩 Tech Stack

Backend

FastAPI — primary REST API framework
Uvicorn — ASGI server
LangChain — for GenAI orchestration (prompt templates + HuggingFaceHub integration)
Hugging Face Transformers — for text & generative models (FLAN-T5, CLIP)
scikit-learn — for KNN/MLP classifier and preprocessing
PyTorch — backend for model inference
Pinecone — VectorDB for semantic search
NumPy / Pandas / Joblib — data utilities

Frontend

React + Vite — modern frontend setup
Tailwind CSS — clean responsive UI
shadcn/ui components — cards, buttons, routing
Lucide React — icons

Data

Provided product dataset with fields:
title, brand, description, price, categories,
images, manufacturer, package dimensions,
country_of_origin, material, color, uniq_id


Preprocessed embeddings are stored in Pinecone and also as local backup (app/data/index.npz).

⚙️ Architecture
User → Frontend (React)
     ↕ REST API
Backend (FastAPI)
 ├── /recommend           (text-based recommendations)
 ├── /search/image        (image similarity search)
 ├── /gen/description     (creative GenAI descriptions)
 ├── /analytics/summary   (dataset insights)
 ├── /cv/classify         (optional CV product type classification)
 │
 ├── Models:
 │   ├── CLIP encoder (text+image)
 │   ├── KNN for embeddings
 │   ├── MLP for image category prediction
 │   └── FLAN-T5 (Generative text)
 │
 └── Vector Database:
      Pinecone → stores embeddings + metadata
