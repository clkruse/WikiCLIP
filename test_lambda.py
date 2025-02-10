import base64
import json
from lambda_handler import lambda_handler
import requests
from pathlib import Path
import urllib.parse

# Hardcoded values
API_URL = "https://hvezpjcs21.execute-api.us-east-1.amazonaws.com/prod/wikiclip-image-matcher"
IMAGE_PATH = "./static/bkg.jpg"

def test_local(image_path):
    """Test the lambda handler locally"""
    print("Testing locally...")
    
    # Read and encode image
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create mock event
    event = {
        'body': json.dumps({
            'image': encoded_image,
            'limit': 5,
            'threshold': 0.5
        })
    }
    
    # Call handler directly
    result = lambda_handler(event, None)
    
    # Print results
    if result['statusCode'] == 200:
        results = json.loads(result['body'])['results']
        print("\nFound similar articles:")
        for article in results:
            print(f"- {article['title']} (similarity: {article['similarity']:.3f})")
            print(f"  URL: {article['url']}")
    else:
        print(f"Error: {result['body']}")

def test_deployed(api_url, image_path):
    """Test the deployed Lambda function via API Gateway"""
    print(f"\nTesting deployed function at {api_url}...")
    print("Making POST request...")
    
    # Read and encode image
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Make request to Lambda function
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            api_url,
            headers=headers,
            json={
                'image': encoded_image,
                'limit': 5,
                'threshold': 0.5
            }
        )
        
        # Print response details for debugging
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response Body: {response.text}")
        
        # Print results
        if response.status_code == 200:
            try:
                response_data = response.json()
                print("\nParsed JSON response:", json.dumps(response_data, indent=2))
                
                if isinstance(response_data, dict) and 'results' in response_data:
                    results = response_data['results']
                    print("\nFound similar articles:")
                    for article in results:
                        print(f"- {article['title']} (similarity: {article['similarity']:.3f})")
                        print(f"  URL: {article['url']}")
                else:
                    print("\nWarning: Response does not contain expected 'results' field")
                    print("Full response content:", response_data)
            except json.JSONDecodeError as e:
                print(f"\nError decoding JSON response: {e}")
                print("Raw response:", response.text)
        else:
            print(f"Error Response: {response.text}")
            print("\nTroubleshooting tips:")
            print("1. Verify the API URL is exactly as shown in API Gateway console")
            print("2. Make sure CORS is enabled in API Gateway")
            print("3. Make sure the API is deployed after making any changes")
            print("4. Check that the Lambda function has proper permissions")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    print("WikiCLIP Lambda Function Tester")
    print("==============================")
    
    # Verify image exists
    if not Path(IMAGE_PATH).exists():
        print(f"Error: Image not found at {IMAGE_PATH}")
        exit(1)
    
    # Test locally first
    print("\nRunning local test...")
    test_local(IMAGE_PATH)
    
    # Run deployed test
    print("\nRunning deployed test...")
    test_deployed(API_URL, IMAGE_PATH) 
