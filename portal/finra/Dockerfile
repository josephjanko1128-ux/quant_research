FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8503

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8503/finra_dashboard/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
  "--server.port=8503", \
  "--server.address=0.0.0.0", \
  "--server.headless=true", \
  "--server.baseUrlPath=/finra_dashboard", \
  "--browser.gatherUsageStats=false"]
