# Document Study Assistant

A Streamlit app for studying a selected PDF or PowerPoint presentation. It extracts document text, splits it into overlapping chunks, creates embeddings, and combines literal/keyword retrieval with semantic search to find passages for answers.

## Features

- Document-grounded questions with supporting source passages.
- Plain-language document summary.
- 5–15 multiple-choice quiz questions, each with four options and answer explanations.
- Per-user Streamlit sessions; one user's uploaded document is not shown to other users.

## Deploy a public app

The recommended low-cost setup is Streamlit Community Cloud plus the Gemini API. The app can be opened by anyone with its URL. Community Cloud apps may sleep when unused, and Gemini API availability, quotas, and pricing depend on your Google AI Studio project.

### 1. Keep private files out of the public repository

`.gitignore` excludes local documents, processed data, vector indexes, the virtual environment, and Streamlit secrets. Do not remove those rules or commit private documents or API keys. Only publish documents that are safe to share with the public.

### 2. Create and push a public GitHub repository

Create a new **public** repository on GitHub (for example, `document-study-assistant`). Do not add a README or other files when creating it. Then open PowerShell in this project folder and run:

```powershell
git init
git add .
git status
```

Review the status and confirm it does **not** list private documents, `.venv`, or secrets. Then:

```powershell
git commit -m "Prepare document study assistant for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_NAME/document-study-assistant.git
git push -u origin main
```

Replace `YOUR_GITHUB_NAME` with your GitHub username.

### 3. Create a Gemini API key

Create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey). Keep the key private. The public app will use this key, so set project quotas and review provider billing/usage limits before sharing the app widely.

### 4. Deploy with Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Select **Create app**, choose the public repository, branch `main`, and main file `app.py`.
3. In the app's **Advanced settings → Secrets**, enter:

   ```toml
   LLM_PROVIDER = "gemini"
   GEMINI_API_KEY = "paste-your-Gemini-API-key-here"
   GEMINI_MODEL = "gemini-3.5-flash-lite"
   ```

4. Deploy. Wait for the app status to become **Running**, then share the app URL.

The model name can be changed with `GEMINI_MODEL` if Google changes model availability. Do not put the API key in Python source, GitHub, or this file. For local use, add the same settings to `.streamlit/secrets.toml` (this file is ignored by Git).

Anyone with the public link can upload documents and make model requests using the configured API key. Streamlit keeps user sessions separate, but the app does not currently have sign-in, request limits, or persistent document storage. Do not upload sensitive documents. Document text used for generation is sent to Google Gemini; check Google's current data-use and privacy terms before deployment.

## Local development

Requirements: Python 3.10+ and Ollama with `llama3.2:3b`, or a Gemini API key.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
ollama serve
```

In another PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

For local Gemini API use, create `.streamlit/secrets.toml` with the deployment settings shown above. The default provider is local Ollama.

## Tests

```powershell
python -m pytest
```
