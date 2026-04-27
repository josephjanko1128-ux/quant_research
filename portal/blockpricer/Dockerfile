FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY block_trade_pricer.py .

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8502/blockpricer/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "block_trade_pricer.py", \
  "--server.port=8502", \
  "--server.address=0.0.0.0", \
  "--server.headless=true", \
  "--server.baseUrlPath=/blockpricer", \
  "--browser.gatherUsageStats=false"]
