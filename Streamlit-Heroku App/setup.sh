mkdir -p ~/.streamlit/
echo "[general]
email = "nitishg43@gmail.com"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
enableCORS=false
port = $PORT
" > ~/.streamlit/config.toml