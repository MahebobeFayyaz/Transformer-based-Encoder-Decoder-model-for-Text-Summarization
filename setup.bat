@echo off
echo 🚀 Setting up summarizer environment...

:: Create virtual environment
python -m venv venv
call venv\Scripts\activate

:: Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

:: Run the Streamlit app
echo ✅ Setup complete. Launching Streamlit app...
streamlit run streamlit_app.py
