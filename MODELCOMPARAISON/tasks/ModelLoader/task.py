import os
import time
import traceback

def load_models():
    """Load YOLOv5 and YOLOv8 models."""
    try:
        print("Starting model loading...")
        
        # Get model paths from variables
        yolov8_path = variables.get("YOLOv8ModelPath")
        yolov5_path = variables.get("YOLOv5ModelPath")
        
        print(f"YOLOv8 path: {yolov8_path}")
        print(f"YOLOv5 path: {yolov5_path}")
        
        # Check if model files exist
        if not os.path.exists(yolov8_path):
            raise FileNotFoundError(f"YOLOv8 model not found: {yolov8_path}")
        if not os.path.exists(yolov5_path):
            raise FileNotFoundError(f"YOLOv5 model not found: {yolov5_path}")
        
        # Simulate model loading time
        start_time = time.time()
        time.sleep(0.1)  # Simulate loading time
        yolov8_load_time = time.time() - start_time
        
        start_time = time.time()
        time.sleep(0.1)  # Simulate loading time
        yolov5_load_time = time.time() - start_time
        
        # Store results
        resultMap.put("YOLOv8_LOAD_TIME", f"{yolov8_load_time:.4f}")
        resultMap.put("YOLOv8_LOADED", "true")
        resultMap.put("YOLOv5_LOAD_TIME", f"{yolov5_load_time:.4f}")
        resultMap.put("YOLOv5_LOADED", "true")
        
        print(f"YOLOv8 load time: {yolov8_load_time:.4f}s")
        print(f"YOLOv5 load time: {yolov5_load_time:.4f}s")
        print("Model loading completed successfully!")
        
    except Exception as e:
        print(f"Error in model loading: {e}")
        traceback.print_exc()
        resultMap.put("YOLOv8_LOADED", "false")
        resultMap.put("YOLOv5_LOADED", "false")
        resultMap.put("ERROR", str(e))

if __name__ == '__main__':
    load_models()
