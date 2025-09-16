FROM public.ecr.aws/lambda/python:3.11-arm64

# Create model directory
RUN mkdir -p /opt/ml/model

# Set environment variables to store models in the container
ENV TRANSFORMERS_CACHE="/opt/ml/model"
ENV TORCH_HOME="/opt/ml/model"
ENV HF_HOME="/opt/ml/model"
ENV HUGGINGFACE_HUB_CACHE="/opt/ml/model"
ENV TRANSFORMERS_NO_ADVISORY_LOCKING="1"
ENV TRANSFORMERS_OFFLINE="1"
ENV HF_HUB_OFFLINE="1"
ENV HF_HUB_DISABLE_TELEMETRY="1"
ENV DISABLE_TELEMETRY="1"
ENV PYTORCH_NUM_THREADS="1"
ENV OMP_NUM_THREADS="1"
ENV MKL_NUM_THREADS="1"
ENV CLIP_MODEL_PATH="/opt/ml/clip"
ENV HF_HOME="/tmp/hf"
ENV TRANSFORMERS_CACHE="/tmp/hf"
ENV DNNL_PRIMITIVE_CACHE_CAPACITY="512"

# Copy requirements file
COPY lambda_requirements.txt ${LAMBDA_TASK_ROOT}

# Ensure latest pip tooling to get prebuilt wheels
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r ${LAMBDA_TASK_ROOT}/lambda_requirements.txt

# Pre-download only the vision model and image processor to reduce cold start
# Temporarily enable online mode for this step to populate the cache
# Download CLIP vision model and processor assets using huggingface_hub
RUN mkdir -p /opt/ml/clip && \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=0 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='openai/clip-vit-base-patch32',
    local_dir='/opt/ml/clip',
    local_dir_use_symlinks=False,
    allow_patterns=['config.json','preprocessor_config.json','merges.txt','vocab.json','*.safetensors','pytorch_model.bin']
)
print('Downloaded CLIP snapshot to /opt/ml/clip')
PY

# Copy function code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}
COPY s3_vectors_client.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ] 
