# Local private Spark image only. Build rc-pi05-base:local first with
# Dockerfile.base.local; that base extends the preinstalled rc-lerobot:train.
FROM rc-pi05-base:local

WORKDIR /app
ENV RC_PI05_TOKENIZER_PATH=/opt/pali-tokenizer
COPY . /app
RUN pip install --no-cache-dir ./rc_sdk

CMD ["python", "-m", "rc_sdk", "--manifest", "manifest.json", \
     "--policy", "policy.pi05:Pi05Policy"]
