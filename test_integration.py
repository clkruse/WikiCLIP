import pytest
from fastapi.testclient import TestClient
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from server import app
from image_matcher import ImageMatcher
import modal
from modal_clip import app as modal_app, ClipProcessor, logger
import asyncio
import torch
import logging

client = TestClient(app)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def event_loop_policy():
    """Create and configure the event loop policy for the test session."""
    return asyncio.get_event_loop_policy()

@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """Create an instance of the default event loop for the test session."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def setup_modal():
    """Setup Modal for testing."""
    try:
        logger.info("Initializing Modal...")
        
        # First, try to authenticate Modal
        try:
            modal.Image.debian_slim()
        except Exception as e:
            logger.error(f"Modal authentication failed. Please run 'modal token new': {e}")
            return None

        # Deploy the Modal app
        logger.info("Deploying Modal app...")
        modal_app.deploy()
        
        # Start the Modal app in the background
        logger.info("Starting Modal app...")
        async with modal_app.run():
            logger.info("Modal app started successfully!")
            yield modal_app
    except Exception as e:
        logger.error(f"Modal setup failed: {e}")
        yield None

@pytest.fixture
async def clip_processor(setup_modal):
    """Create a CLIP processor instance within Modal context."""
    if setup_modal is None:
        pytest.skip("Modal setup failed")
    
    try:
        logger.info("Initializing CLIP processor...")
        processor = ClipProcessor()
        await processor.__aenter__()  # Initialize the Modal context
        logger.info("CLIP processor initialized successfully!")
        yield processor
        await processor.__aexit__(None, None, None)  # Cleanup
    except Exception as e:
        logger.error(f"CLIP processor initialization failed: {e}")
        yield None

def create_test_image():
    """Create a test image and convert it to base64."""
    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

@pytest.mark.asyncio
async def test_find_similar_endpoint(setup_modal):
    """Test the /api/find-similar endpoint with a test image."""
    if setup_modal is None:
        pytest.skip("Modal setup failed")

    # Create test image
    img_str = create_test_image()
    
    # Make request to the endpoint
    response = client.post(
        "/api/find-similar",
        json={"image": f"data:image/jpeg;base64,{img_str}"}
    )
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    
    # Check each result has the expected fields
    for result in data["results"]:
        assert "title" in result
        assert "similarity" in result
        assert "url" in result
        assert isinstance(result["similarity"], float)
        assert 0 <= result["similarity"] <= 1

@pytest.mark.asyncio
async def test_image_matcher_modal(setup_modal, clip_processor):
    """Test that ImageMatcher correctly uses Modal CLIP processor."""
    if clip_processor is None:
        pytest.skip("ClipProcessor initialization failed")

    # Initialize ImageMatcher with the test processor
    matcher = ImageMatcher()
    matcher.processor = clip_processor
    
    # Create test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Get embedding
    embedding = await matcher.get_embedding(img_bytes)
    
    # Check embedding properties
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape[1] == 512  # CLIP embeddings are 512-dimensional
    assert torch.allclose(torch.norm(embedding), torch.tensor(1.0), atol=1e-6)  # Should be normalized

@pytest.mark.asyncio
async def test_similar_articles(setup_modal, clip_processor):
    """Test the full pipeline of finding similar articles."""
    if clip_processor is None:
        pytest.skip("ClipProcessor initialization failed")

    # Initialize ImageMatcher with the test processor
    matcher = ImageMatcher()
    matcher.processor = clip_processor
    
    # Create test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Get similar articles
    results = await matcher.get_similar_articles(img_bytes, limit=5)
    
    # Check results
    assert isinstance(results, list)
    assert len(results) <= 5
    
    for result in results:
        assert "article_id" in result
        assert "title" in result
        assert "url" in result
        assert "similarity" in result
        assert isinstance(result["similarity"], float)
        assert 0 <= result["similarity"] <= 1

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=auto", __file__]) 