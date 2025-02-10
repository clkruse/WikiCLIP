from PIL import Image
import json
import base64
from io import BytesIO

# Create a simple test image (red square)
img = Image.new('RGB', (224, 224), color='red')

# Convert to base64
buffer = BytesIO()
img.save(buffer, format='JPEG')
img_str = base64.b64encode(buffer.getvalue()).decode()

# Create the test event
test_event = {
    "body": {
        "image": img_str,
        "limit": 15,
        "threshold": 0.5
    }
}

# Save to file
with open('test_event.json', 'w') as f:
    json.dump(test_event, f, indent=2)

print("Test event created in test_event.json")
print("You can now copy this into your Lambda test event.") 