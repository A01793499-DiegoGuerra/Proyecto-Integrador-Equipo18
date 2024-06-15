#!/bin/bash

# Enable debug mode
set -x

# Start Streamlit in the background
streamlit run app.py --server.headless true --server.port 8501 &

# Start Apache in the foreground
apache2ctl -D FOREGROUND