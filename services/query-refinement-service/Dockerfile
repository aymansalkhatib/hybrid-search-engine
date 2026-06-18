# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY services/query-refinement-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake NLTK resources into the image so runtime is offline & reproducible
# (wordnet → synonym expansion; stopwords → skip non-content terms). Same CDN
# approach as the preprocessing-service: a truncated read fails the BUILD, not startup.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import io,os,urllib.request,zipfile; base='https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages'; [zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f'{base}/{s}/{p}.zip',timeout=180).read())).extractall(os.path.join(os.environ['NLTK_DATA'],s)) for s,p in (('corpora','stopwords'),('corpora','wordnet'))]"

# Validate in a fresh process so corrupt/missing data fails the build, and confirm
# the SymSpell frequency dictionary shipped inside the symspellpy wheel.
RUN python -c "from nltk.corpus import wordnet, stopwords; stopwords.words('english'); assert wordnet.synsets('dog'); import importlib.resources, os; p=importlib.resources.files('symspellpy')/'frequency_dictionary_en_82_765.txt'; assert os.path.exists(p), p" && echo 'refinement resources verified'

# Shared contracts + service code.
COPY shared/ ./shared/
COPY services/query-refinement-service/app/ ./app/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8005/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
