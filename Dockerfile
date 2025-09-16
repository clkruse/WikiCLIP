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

# Copy requirements file
COPY lambda_requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r ${LAMBDA_TASK_ROOT}/lambda_requirements.txt

# Pre-download only the vision model and image processor to reduce cold start
RUN python3 - <<'PY'
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('Prefetched CLIP vision model and image processor')
PY

# Copy function code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}
COPY s3_vectors_client.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ] 
