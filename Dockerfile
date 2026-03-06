# -------- Use official Python 3.11 image ---------------
FROM python:3.11.9

#------------ Environment variables ---------------------
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Use environment variable to set port dynamically Default to 7860 for HF Spaces
ENV STREAMLIT_PORT=7860  

# ----------- Set working directory inside container-----
WORKDIR /app

# ------------- Install dependencies ---------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------------- Copy all project files into container inside app/ ------------
COPY . .

# -------------------Expose port 7860 (HF Spaces default) ------------------
EXPOSE 7860

# -------------------- Run Streamlit -------------------------------------------
CMD ["sh", "-c", "streamlit run src/app.py --server.port=$STREAMLIT_PORT --server.address=0.0.0.0"]

# Map container port 7860 to host port 8501
# docker run -p 8501:7860 my-rag-app
# open http://localhost:8501 on laptop for local testing, and container listens on 7860 internally(HF spaces requires 7860 port)

