FROM public.ecr.aws/lambda/python:3.11-arm64

# Install system dependencies
RUN yum install -y gcc gcc-c++ python3-devel

# Create model directory
RUN mkdir -p /opt/ml/model

# Set environment variables to store models in the container
ENV TRANSFORMERS_CACHE="/opt/ml/model"
ENV TORCH_HOME="/opt/ml/model"
ENV HF_HOME="/opt/ml/model"
ENV HUGGINGFACE_HUB_CACHE="/opt/ml/model"

# Copy requirements file
COPY lambda_requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies with specific numpy version
RUN pip install -r ${LAMBDA_TASK_ROOT}/lambda_requirements.txt

# Pre-download the model during build
RUN python3 -c "from transformers import CLIPProcessor, CLIPModel; model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Copy function code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ] 
