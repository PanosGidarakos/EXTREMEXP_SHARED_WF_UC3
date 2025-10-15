import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import pandas as pd
import os
import traceback

# Helper function to convert DataFrame to JSON
def df_to_json(df):
    """Convert DataFrame to JSON string."""
    if df.empty:
        return "[]"
    return df.to_json(orient='records')

def test_model(model, image_path, model_name):
    """
    Loads the image, converts it to RGB, and passes it to the model for inference.
    """
    try:
        print(f"Testing {model_name} with image: {image_path}")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or incorrect format: {image_path}")
        
        print(f"Image loaded successfully: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Measure inference time
        start_time = time.time()
        results = model(image_rgb)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # milliseconds
        
        print(f"{model_name} inference time: {inference_time:.2f} ms")
        print(f"{model_name} results type: {type(results)}")
        
        # Debug: Check if results contain detections
        if hasattr(results, 'boxes') and results.boxes is not None:
            print(f"{model_name} boxes found: {len(results.boxes)}")
        elif hasattr(results, '__len__'):
            print(f"{model_name} results length: {len(results)}")
        else:
            print(f"{model_name} no boxes attribute found")
        
        return results, inference_time
        
    except Exception as e:
        print(f"Error during {model_name} inference: {e}")
        traceback.print_exc()
        return None, 0

def extract_predictions(results, model_name):
    """Extract predictions from model results and convert to DataFrame."""
    try:
        print(f"Extracting predictions from {model_name} results...")
        
        if model_name == "YOLOv8":
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                if hasattr(boxes, 'data') and boxes.data is not None:
                    data = boxes.data.cpu().numpy()
                    columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
                    pred_df = pd.DataFrame(data, columns=columns)
                    pred_df['class'] = pred_df['class'].astype(int)
                    pred_df['name'] = pred_df['class'].apply(lambda x: results[0].names[x])
                    print(f"YOLOv8: Extracted {len(pred_df)} predictions")
                    return pred_df
                else:
                    print("YOLOv8: No boxes data found")
                    return pd.DataFrame()
            else:
                print("YOLOv8: No boxes found in results")
                return pd.DataFrame()
            
        elif model_name == "YOLOv5":
            try:
                if hasattr(results, 'pandas'):
                    pred_df = results.pandas().xyxy
                    if isinstance(pred_df, list):
                        pred_df = pred_df[0]
                    print(f"YOLOv5: Extracted {len(pred_df)} predictions via pandas")
                    return pred_df
                elif hasattr(results, 'boxes'):
                    boxes = results.boxes
                    if hasattr(boxes, 'data') and boxes.data is not None:
                        data = boxes.data.cpu().numpy()
                        pred_df = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
                        pred_df['class'] = pred_df['class'].astype(int)
                        pred_df['name'] = pred_df['class'].apply(lambda x: results.names[x])
                        print(f"YOLOv5: Extracted {len(pred_df)} predictions via boxes")
                        return pred_df
                    else:
                        print("YOLOv5: No boxes data found")
                        return pd.DataFrame()
                else:
                    print("YOLOv5: No pandas or boxes method found")
                    return pd.DataFrame()
            except Exception as e:
                print(f"Error extracting YOLOv5 predictions: {e}")
                traceback.print_exc()
                return pd.DataFrame()
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error extracting {model_name} predictions: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def run_inference():
    """Main inference function for multiple images."""
    try:
        print("Starting inference engine for multiple images...")
        
        # Get input parameters
        image_paths = variables.get("ImagePath")
        yolov8_model_path = variables.get("YOLOv8ModelPath")
        yolov5_model_path = variables.get("YOLOv5ModelPath")
        
        print(f"Image paths: {image_paths}")
        print(f"YOLOv8 model path: {yolov8_model_path}")
        print(f"YOLOv5 model path: {yolov5_model_path}")
        
        # Handle single image path (string) or multiple paths (list)
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        elif not isinstance(image_paths, list):
            image_paths = [str(image_paths)]
        
        print(f"Processing {len(image_paths)} images")
        
        # Check if files exist
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(yolov8_model_path):
            raise FileNotFoundError(f"YOLOv8 model not found: {yolov8_model_path}")
        if not os.path.exists(yolov5_model_path):
            raise FileNotFoundError(f"YOLOv5 model not found: {yolov5_model_path}")
        
        # Load models
        print("Loading YOLOv8 model...")
        yolov8_model = YOLO(yolov8_model_path)
        print("YOLOv8 model loaded successfully")
        
        print("Loading YOLOv5 model...")
        import yolov5
        import torch
        # Fix for PyTorch 2.6 weights_only issue - use weights_only=False
        import torch.serialization
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        
        yolov5_model = yolov5.load(yolov5_model_path)
        print("YOLOv5 model loaded successfully")
        
        # Initialize aggregated results
        all_pred_v8 = []
        all_pred_v5 = []
        total_time_v8 = 0
        total_time_v5 = 0
        
        # Process each image
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Run inference on both models for this image
            yolov8_results, time_v8 = test_model(yolov8_model, image_path, f"YOLOv8_img_{i+1}")
            yolov5_results, time_v5 = test_model(yolov5_model, image_path, f"YOLOv5_img_{i+1}")
            
            # Extract predictions
            pred_df_v8 = extract_predictions(yolov8_results, f"YOLOv8_img_{i+1}") if yolov8_results else pd.DataFrame()
            pred_df_v5 = extract_predictions(yolov5_results, f"YOLOv5_img_{i+1}") if yolov5_results else pd.DataFrame()
            
            # Add image identifier to predictions
            if not pred_df_v8.empty:
                pred_df_v8['image_path'] = image_path
                pred_df_v8['image_id'] = i
                all_pred_v8.append(pred_df_v8)
            
            if not pred_df_v5.empty:
                pred_df_v5['image_path'] = image_path
                pred_df_v5['image_id'] = i
                all_pred_v5.append(pred_df_v5)
            
            total_time_v8 += time_v8
            total_time_v5 += time_v5
            
            print(f"Image {i+1}: YOLOv8={len(pred_df_v8)} detections, YOLOv5={len(pred_df_v5)} detections")
        
        # Combine all predictions
        combined_pred_v8 = pd.concat(all_pred_v8, ignore_index=True) if all_pred_v8 else pd.DataFrame()
        combined_pred_v5 = pd.concat(all_pred_v5, ignore_index=True) if all_pred_v5 else pd.DataFrame()
        
        # Calculate aggregated statistics
        num_detections_v8 = len(combined_pred_v8)
        num_detections_v5 = len(combined_pred_v5)
        mean_conf_v8 = combined_pred_v8['confidence'].mean() if num_detections_v8 > 0 else 0
        mean_conf_v5 = combined_pred_v5['confidence'].mean() if num_detections_v5 > 0 else 0
        avg_time_v8 = total_time_v8 / len(image_paths) if len(image_paths) > 0 else 0
        avg_time_v5 = total_time_v5 / len(image_paths) if len(image_paths) > 0 else 0
        
        # Store results
        resultMap.put("YOLOv8_INFERENCE_TIME", f"{avg_time_v8:.2f}")
        resultMap.put("YOLOv5_INFERENCE_TIME", f"{avg_time_v5:.2f}")
        resultMap.put("YOLOv8_TOTAL_TIME", f"{total_time_v8:.2f}")
        resultMap.put("YOLOv5_TOTAL_TIME", f"{total_time_v5:.2f}")
        resultMap.put("YOLOv8_DETECTIONS", str(num_detections_v8))
        resultMap.put("YOLOv5_DETECTIONS", str(num_detections_v5))
        resultMap.put("YOLOv8_MEAN_CONFIDENCE", f"{mean_conf_v8:.3f}")
        resultMap.put("YOLOv5_MEAN_CONFIDENCE", f"{mean_conf_v5:.3f}")
        resultMap.put("NUM_IMAGES_PROCESSED", str(len(image_paths)))
        
        # Store prediction data as JSON metrics
        resultMap.put("YOLOv8_PREDICTIONS_JSON", df_to_json(combined_pred_v8))
        resultMap.put("YOLOv5_PREDICTIONS_JSON", df_to_json(combined_pred_v5))
        
        # Legacy metrics for compatibility
        resultMap.put("YOLOv8_DATA_SAVED", "true" if not combined_pred_v8.empty else "false")
        resultMap.put("YOLOv5_DATA_SAVED", "true" if not combined_pred_v5.empty else "false")
        
        print(f"YOLOv8 predictions stored as JSON: {len(combined_pred_v8)} total detections")
        print(f"YOLOv5 predictions stored as JSON: {len(combined_pred_v5)} total detections")
        
        print("Inference completed successfully!")
        print(f"Processed {len(image_paths)} images")
        print(f"YOLOv8: {num_detections_v8} total detections, {avg_time_v8:.2f}ms avg, conf: {mean_conf_v8:.3f}")
        print(f"YOLOv5: {num_detections_v5} total detections, {avg_time_v5:.2f}ms avg, conf: {mean_conf_v5:.3f}")
        
    except Exception as e:
        print(f"Critical error in run_inference: {e}")
        traceback.print_exc()
        # Set default values in case of error
        resultMap.put("YOLOv8_INFERENCE_TIME", "0.0")
        resultMap.put("YOLOv5_INFERENCE_TIME", "0.0")
        resultMap.put("YOLOv8_TOTAL_TIME", "0.0")
        resultMap.put("YOLOv5_TOTAL_TIME", "0.0")
        resultMap.put("YOLOv8_DETECTIONS", "0")
        resultMap.put("YOLOv5_DETECTIONS", "0")
        resultMap.put("YOLOv8_MEAN_CONFIDENCE", "0.0")
        resultMap.put("YOLOv5_MEAN_CONFIDENCE", "0.0")
        resultMap.put("NUM_IMAGES_PROCESSED", "0")
        resultMap.put("YOLOv8_PREDICTIONS_JSON", "[]")
        resultMap.put("YOLOv5_PREDICTIONS_JSON", "[]")
        resultMap.put("YOLOv8_DATA_SAVED", "false")
        resultMap.put("YOLOv5_DATA_SAVED", "false")

if __name__ == '__main__':
    run_inference()
