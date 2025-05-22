# AI-Powered Math Question Generator

An intelligent system for automatically generating high-quality, context-aware, university-level math exam questions. Designed to support both general subjects (e.g. Probability for Mathematics) and domain-specific topics (e.g. Statistics for Mechanical Engineering), the system assists instructors by saving time, increasing question diversity, and enhancing educational efficiency.

##  Features

- **Multi-domain support**: Tailors questions to subjects like engineering, mathematics, and more.
- **Multi-agent RAG architecture**: Retriever, Generator, and Validator agents work collaboratively using RAG (Retrieval-Augmented Generation).
- **LLM-powered generation**: Uses local large language models (like Mistral) for generation.
- **Instructor input support**: Accepts course materials and example questions to guide generation.
- **Real-time status**: WebSocket-based updates for generation progress.
- **Human-in-the-loop**: Instructors can review, approve, or edit generated questions.
- **Database integration**: Stores questions, logs, and vector embeddings in a structured DB.

## 🛠 Tech Stack

- **Python** – Backend logic and service layer
- **LangGraph** – Multi-agent orchestration for Retrieval-Augmented Generation (RAG)
- **Mistral AI** – Used via API key for question generation
- **PostgreSQL** – Relational database with `pgvector` extension for vector storage
- **WebSockets** – Real-time generation progress updates
- **Frontend** – Built with HTML5, CSS3, Bootstrap, and vanilla JavaScript

##  Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/MahmudzadaZiyarat/ai-math-question-generator
   cd ai-math-question-generator

2. **Create and activate a virtual environment**
   # Create virtual environment
   python -m venv math_venv

   # Activate virtual environment
   # On Windows:
   math_venv\Scripts\activate

   # On macOS/Linux:
   source math_venv/bin/activate


3. **Install python packages**  
   pip install -r requirements.txt


4. **Download required NLP models**  
   # Download spaCy English model
   python -m spacy download en_core_web_sm

# Download NLTK punkt tokenizer
   python -c "import nltk; nltk.download('punkt')"

