# update pip
pip install --upgrade pip setuptools

# setup streamlit
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"prs.watch@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml