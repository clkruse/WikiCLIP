#!/bin/bash

# Exit on error
set -e

# Configuration
AWS_REGION="us-east-1"  # Change this to your desired region
ECR_REPO_NAME="wikiclip-lambda"
LAMBDA_FUNCTION_NAME="wikiclip-image-matcher"
IMAGE_TAG="latest"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} || \
    aws ecr create-repository --repository-name ${ECR_REPO_NAME}

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build the Docker image specifically for ARM64
docker buildx create --use
docker buildx build --platform linux/arm64 \
    --tag ${ECR_REPO_NAME}:${IMAGE_TAG} \
    --tag ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG} \
    --load \
    .

# Push the image to ECR
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}

# Update Lambda function to use the new image
echo "Updating Lambda function to use the new image..."
aws lambda update-function-code \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --image-uri ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}

echo "Container successfully built and pushed to ECR"
echo "ECR Image URI: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
echo "Lambda function ${LAMBDA_FUNCTION_NAME} updated to use the new image" 
