# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY services/preprocessing-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake NLTK resources into the image so runtime is offline & reproducible.
# Download each zip straight from jsdelivr's CDN, which serves these files in
# one shot (raw.githubusercontent truncates large files on slow links). A
# truncated read raises here and fails the BUILD, never the container startup.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import io,os,urllib.request,zipfile; base='https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages'; [zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f'{base}/{s}/{p}.zip',timeout=180).read())).extractall(os.path.join(os.environ['NLTK_DATA'],s)) for s,p in (('corpora','stopwords'),('corpora','wordnet'),('taggers','averaged_perceptron_tagger_eng'))]"

# Validate in a fresh process so corrupt/missing data fails the build.
RUN python -c "from nltk.corpus import wordnet, stopwords; from nltk import pos_tag; from nltk.stem import WordNetLemmatizer; stopwords.words('english'); assert WordNetLemmatizer().lemmatize('studies','n')=='study'; assert wordnet.synsets('dog'); assert pos_tag(['running'])" && echo 'NLTK resources verified'

# Shared contracts + service code.
COPY shared/ ./shared/
COPY services/preprocessing-service/app/ ./app/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
