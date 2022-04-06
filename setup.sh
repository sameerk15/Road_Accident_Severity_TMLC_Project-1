mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "[server]\nheadless = true\nenableCORS=false\nport = \n
" > ~/.streamlit/config.toml