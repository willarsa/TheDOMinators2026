import numpy as np
from backend.utils import preprocess_image
import PIL.Image as Image
import io

def test_scaling():
    # Create a dummy white image (255, 255, 255)
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    bytes_data = buf.getvalue()
    
    # Preprocess
    arr = preprocess_image(bytes_data)
    
    # Expected: (255 / 127.5) - 1.0 = 2.0 - 1.0 = 1.0
    print(f"White pixel scaling (255): {arr[0,0,0,0]}")
    assert np.isclose(arr[0,0,0,0], 1.0)
    
    # Create a dummy black image (0, 0, 0)
    img = Image.new('RGB', (224, 224), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    bytes_data = buf.getvalue()
    
    # Preprocess
    arr = preprocess_image(bytes_data)
    
    # Expected: (0 / 127.5) - 1.0 = -1.0
    print(f"Black pixel scaling (0): {arr[0,0,0,0]}")
    assert np.isclose(arr[0,0,0,0], -1.0)

    # Create a dummy grey image (127.5)
    # Note: JPEG might not be exact, so we'll use a tolerance
    img = Image.new('RGB', (224, 224), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format='PNG') # Use PNG for exactness
    bytes_data = buf.getvalue()
    
    arr = preprocess_image(bytes_data)
    print(f"Grey pixel scaling (127): {arr[0,0,0,0]}")
    # (127 / 127.5) - 1.0 approx -0.0039
    
    print("✅ Preprocessing scaling test passed!")

if __name__ == "__main__":
    test_scaling()
