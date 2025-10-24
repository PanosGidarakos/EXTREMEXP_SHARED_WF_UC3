import os
[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import pandas as pd
import os
import traceback
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import random
import glob
import proactive_helper as ph


def df_to_csv_bytes(df):
    """Convert a DataFrame to CSV bytes."""
    df_to_bytes = df.to_csv(index=False).encode("utf-8")
    # Log the size in human-readable format
    return df_to_bytes


def apply_image_transformations(image, resolution_scale=1.0, blur_kernel=0, noise_level=0.0, 
                                 brightness=1.0, contrast=1.0, rotation=0):
    """
    Apply various transformations to test model robustness.
    
    Args:
        image: Input image (numpy array)
        resolution_scale: Scale factor for image resolution (0.25-2.0)
        blur_kernel: Gaussian blur kernel size (0-15, must be odd)
        noise_level: Gaussian noise standard deviation (0-50)
        brightness: Brightness multiplication factor (0.3-2.0)
        contrast: Contrast multiplication factor (0.3-2.0)
        rotation: Rotation angle in degrees (-45 to 45)
    
    Returns:
        Transformed image
    """
    transformed = image.copy()
    original_height, original_width = image.shape[:2]
    
    # 1. Resolution scaling
    if resolution_scale != 1.0:
        new_width = int(original_width * resolution_scale)
        new_height = int(original_height * resolution_scale)
        transformed = cv2.resize(transformed, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # Scale back to original size for model input
        transformed = cv2.resize(transformed, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        print(f"  Applied resolution scaling: {resolution_scale}x")
    
    # 2. Gaussian blur
    if blur_kernel > 0:
        # Ensure kernel size is odd
        kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        transformed = cv2.GaussianBlur(transformed, (kernel_size, kernel_size), 0)
        print(f"  Applied Gaussian blur: kernel={kernel_size}")
    
    # 3. Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, transformed.shape).astype(np.float32)
        transformed = transformed.astype(np.float32) + noise
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        print(f"  Applied Gaussian noise: level={noise_level}")
    
    # 4. Brightness adjustment
    if brightness != 1.0:
        transformed = transformed.astype(np.float32) * brightness
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        print(f"  Applied brightness: factor={brightness}")
    
    # 5. Contrast adjustment
    if contrast != 1.0:
        # Convert to float and apply contrast
        transformed = transformed.astype(np.float32)
        mean = transformed.mean()
        transformed = mean + (transformed - mean) * contrast
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        print(f"  Applied contrast: factor={contrast}")
    
    # 6. Rotation
    if rotation != 0:
        center = (original_width // 2, original_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        transformed = cv2.warpAffine(transformed, rotation_matrix, (original_width, original_height), 
                                     borderMode=cv2.BORDER_REFLECT)
        print(f"  Applied rotation: {rotation} degrees")
    
    return transformed


def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two boxes.
    Each box is a list [xmin, ymin, xmax, ymax].
    """
    try:
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
    except Exception as e:
        print(f"Error computing IoU: {e}")
        return 0.0


def load_ground_truth_data(gt_csv_path, image_height, image_width):
    """
    Loads ground truth annotations from a CSV file in YOLO format.
    Converts normalized YOLO format to pixel coordinates.
    """
    try:
        print(f"Loading ground truth from: {gt_csv_path}")
        if not os.path.exists(gt_csv_path):
            print(f"Ground truth file not found: {gt_csv_path}")
            return [], []
            
        # Read CSV without headers (YOLO format: class x_center y_center width height - normalized)
        gt_df = pd.read_csv(gt_csv_path, header=None, sep=' ')
        print(f"Ground truth data shape: {gt_df.shape}")
        
        gt_boxes = []
        gt_labels = []
        
        for idx, row in gt_df.iterrows():
            class_id = int(row[0])
            x_center = float(row[1]) * image_width
            y_center = float(row[2]) * image_height
            width = float(row[3]) * image_width
            height = float(row[4]) * image_height
            
            # Convert from YOLO format to xmin, ymin, xmax, ymax (pixel coordinates)
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_labels.append(class_id)
        
        print(f"Loaded {len(gt_boxes)} ground truth boxes")
        return gt_boxes, gt_labels
        
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        traceback.print_exc()
        return [], []


def evaluate_detections(pred_df, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Evaluates detections against ground truth annotations.
    Returns precision, recall arrays and average precision.
    """
    try:
        if pred_df.empty:
            print("No predictions to evaluate")
            return np.array([]), np.array([]), 0.0
        
        if not gt_boxes:
            print("No ground truth boxes to evaluate against")
            return np.array([]), np.array([]), 0.0
        
        # Sort predictions by confidence
        pred_df = pred_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        
        tp = []
        fp = []
        confidences = []
        matched_gt = [False] * len(gt_boxes)
        
        for idx, row in pred_df.iterrows():
            pred_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            pred_conf = row['confidence']
            pred_label = int(row['class'])
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find the best matching ground truth box
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if pred_label != gt_label:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Check if it's a true positive or false positive
            if best_iou >= iou_threshold and best_gt_idx != -1 and not matched_gt[best_gt_idx]:
                tp.append(1)
                fp.append(0)
                matched_gt[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
            
            confidences.append(pred_conf)
        
        tp = np.array(tp)
        fp = np.array(fp)
        confidences = np.array(confidences)
        
        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        recalls = tp_cumsum / len(gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate average precision
        if len(tp) > 0:
            avg_precision = average_precision_score(tp, confidences) if len(np.unique(tp)) > 1 else 0.0
        else:
            avg_precision = 0.0
        
        return precisions, recalls, avg_precision
        
    except Exception as e:
        print(f"Error in evaluate_detections: {e}")
        traceback.print_exc()
        return np.array([]), np.array([]), 0.0


def save_annotated_image(image_path, pred_df, output_path, model_name):
    """
    Draws detection boxes on the image and saves the annotated image.
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return False
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or incorrect format: {image_path}")
        
        if pred_df.empty:
            print(f"No detections to visualize for {model_name}")
            # Save image without annotations
            cv2.imwrite(output_path, image)
            return True
            
        print(f"Drawing {len(pred_df)} detections for {model_name}")
        
        for idx, row in pred_df.iterrows():
            try:
                xmin = int(row['xmin'])
                ymin = int(row['ymin'])
                xmax = int(row['xmax'])
                ymax = int(row['ymax'])
                confidence = row['confidence']
                label = row.get('name', f"class_{int(row['class'])}")
                
                # Draw bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Draw label with confidence
                text = f"{label} {confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw background for text
                cv2.rectangle(
                    image, 
                    (xmin, max(ymin - text_height - 4, 0)), 
                    (xmin + text_width, ymin), 
                    (0, 0, 0), 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    image, text, (xmin, ymin - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            except Exception as e:
                print(f"Error drawing detection {idx}: {e}")
                continue
        
        cv2.imwrite(output_path, image)
        print(f"{model_name} annotated image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving annotated image for {model_name}: {e}")
        traceback.print_exc()
        return False


def plot_precision_recall_curve(pr_data_v8, pr_data_v5, ap_v8, ap_v5, output_path):
    """
    Plots precision-recall curves for both models.
    """
    try:
        print("Generating precision-recall comparison plot...")
        
        plt.figure(figsize=(10, 7))
        
        if len(pr_data_v8['recall']) > 0:
            plt.plot(
                pr_data_v8['recall'], 
                pr_data_v8['precision'], 
                label=f'YOLOv8 (AP={ap_v8:.4f})', 
                linewidth=2,
                color='blue'
            )
        
        if len(pr_data_v5['recall']) > 0:
            plt.plot(
                pr_data_v5['recall'], 
                pr_data_v5['precision'], 
                label=f'YOLOv5 (AP={ap_v5:.4f})', 
                linewidth=2,
                color='red'
            )
        
        plt.xlabel('Recall', fontsize=13)
        plt.ylabel('Precision', fontsize=13)
        plt.title('Precision-Recall Curve Comparison', fontsize=15, fontweight='bold')
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Precision-recall plot saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating precision-recall plot: {e}")
        traceback.print_exc()
        return False


def run_model_comparison():
    """
    Main function that performs complete model comparison workflow:
    1. Load models
    2. Run inference
    3. Evaluate metrics
    4. Generate visualizations
    """
    try:
        print("=" * 80)
        print("STARTING MODEL COMPARISON WORKFLOW")
        print("=" * 80)
        
        # =====================================================================
        # 1. GET INPUT PARAMETERS
        # =====================================================================
        image_path_pattern = variables.get("ImagePath")
        yolov8_model_path = variables.get("YOLOv8ModelPath")
        yolov5_model_path = variables.get("YOLOv5ModelPath")
        ground_truth_path_pattern = variables.get("GroundTruthPath")
        iou_threshold_raw = int(variables.get("IoUThreshold"))
        iou_threshold = float(iou_threshold_raw) / 100.0
        
        # Find all available images and ground truth files
        image_dir = os.path.dirname(image_path_pattern)
        gt_dir = os.path.dirname(ground_truth_path_pattern)
        
        # Get all fire images (png or jpg)
        available_images = []
        for ext in ['png', 'jpg', 'jpeg']:
            available_images.extend(glob.glob(os.path.join(image_dir, f"fire.*.{ext}")))
        
        if not available_images:
            raise FileNotFoundError(f"No fire images found in {image_dir}")
        
        # Select a random image
        image_path = random.choice(available_images)
        
        # Get the corresponding ground truth file
        image_basename = os.path.splitext(os.path.basename(image_path))[0]  # e.g., "fire.1"
        ground_truth_path = os.path.join(gt_dir, f"{image_basename}.csv")
        
        print(f"\n🎲 Randomly selected:")
        print(f"  Image: {image_path}")
        print(f"  Ground Truth: {ground_truth_path}")
        
        # Image transformation parameters (convert integers to appropriate ranges)
        resolution_scale = float(variables.get("image_resolution_scale")) / 100.0  # 25-200 → 0.25-2.0
        blur_kernel = int(variables.get("blur_kernel_size"))  # 0-15
        noise_level = float(variables.get("noise_level"))  # 0-30
        brightness_factor = float(variables.get("brightness_factor")) / 100.0  # 30-200 → 0.3-2.0
        contrast_factor = float(variables.get("contrast_factor")) / 100.0  # 30-200 → 0.3-2.0
        rotation_angle = int(variables.get("rotation_angle"))  # -30 to 30
        
        print(f"\nInput Parameters:")
        print(f"  Image: {image_path}")
        print(f"  YOLOv8 Model: {yolov8_model_path}")
        print(f"  YOLOv5 Model: {yolov5_model_path}")
        print(f"  Ground Truth: {ground_truth_path}")
        print(f"  IoU Threshold: {iou_threshold}")
        print(f"\nImage Transformation Parameters:")
        print(f"  Resolution Scale: {resolution_scale}")
        print(f"  Blur Kernel Size: {blur_kernel}")
        print(f"  Noise Level: {noise_level}")
        print(f"  Brightness Factor: {brightness_factor}")
        print(f"  Contrast Factor: {contrast_factor}")
        print(f"  Rotation Angle: {rotation_angle}°")
        
        # Validate files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(yolov8_model_path):
            raise FileNotFoundError(f"YOLOv8 model not found: {yolov8_model_path}")
        if not os.path.exists(yolov5_model_path):
            raise FileNotFoundError(f"YOLOv5 model not found: {yolov5_model_path}")
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
        
        # =====================================================================
        # 2. LOAD MODELS
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 1: LOADING MODELS")
        print("=" * 80)
        
        print("\nLoading YOLOv8 model...")
        start_time = time.time()
        yolov8_model = YOLO(yolov8_model_path)
        yolov8_load_time = time.time() - start_time
        print(f"YOLOv8 loaded in {yolov8_load_time:.4f}s")
        
        print("\nLoading YOLOv5 model...")
        # Fix for PyTorch weights_only issue
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        
        start_time = time.time()
        import yolov5
        yolov5_model = yolov5.load(yolov5_model_path)
        yolov5_load_time = time.time() - start_time
        print(f"YOLOv5 loaded in {yolov5_load_time:.4f}s")
        
        # =====================================================================
        # 3. RUN INFERENCE
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 2: RUNNING INFERENCE")
        print("=" * 80)
        
        # Load and transform image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_height, image_width = original_image.shape[:2]
        print(f"\nOriginal image dimensions: {image_width}x{image_height}")
        
        # Apply transformations
        print("\nApplying image transformations...")
        transformed_image = apply_image_transformations(
            original_image,
            resolution_scale=resolution_scale,
            blur_kernel=blur_kernel,
            noise_level=noise_level,
            brightness=brightness_factor,
            contrast=contrast_factor,
            rotation=rotation_angle
        )
        
        # Save transformed image for reference
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        transformed_path = f"{image_basename}_transformed.jpg"
        cv2.imwrite(transformed_path, transformed_image)
        print(f"Transformed image saved to: {transformed_path}")
        
        image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        
        # YOLOv8 inference
        print("\nRunning YOLOv8 inference...")
        start_time = time.time()
        results_v8 = yolov8_model(image_rgb)
        yolov8_inference_time = (time.time() - start_time) * 1000  # milliseconds
        print(f"YOLOv8 inference time: {yolov8_inference_time:.2f} ms")
        
        # Extract YOLOv8 predictions
        pred_df_v8 = pd.DataFrame()
        if hasattr(results_v8[0], 'boxes') and results_v8[0].boxes is not None:
            boxes = results_v8[0].boxes
            if hasattr(boxes, 'data') and boxes.data is not None:
                data = boxes.data.cpu().numpy()
                pred_df_v8 = pd.DataFrame(data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
                pred_df_v8['class'] = pred_df_v8['class'].astype(int)
                pred_df_v8['name'] = pred_df_v8['class'].apply(lambda x: results_v8[0].names[x])
        print(f"YOLOv8 detections: {len(pred_df_v8)}")
        
        # YOLOv5 inference
        print("\nRunning YOLOv5 inference...")
        start_time = time.time()
        results_v5 = yolov5_model(image_rgb)
        yolov5_inference_time = (time.time() - start_time) * 1000  # milliseconds
        print(f"YOLOv5 inference time: {yolov5_inference_time:.2f} ms")
        
        # Extract YOLOv5 predictions
        pred_df_v5 = pd.DataFrame()
        if hasattr(results_v5, 'pandas'):
            pred_df_v5 = results_v5.pandas().xyxy[0]
        print(f"YOLOv5 detections: {len(pred_df_v5)}")
        
        # =====================================================================
        # 4. LOAD GROUND TRUTH AND EVALUATE
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 3: EVALUATION")
        print("=" * 80)
        
        # Load ground truth
        print("\nLoading ground truth...")
        gt_boxes, gt_labels = load_ground_truth_data(ground_truth_path, image_height, image_width)
        print(f"Ground truth boxes: {len(gt_boxes)}")
        
        # Evaluate YOLOv8
        print(f"\nEvaluating YOLOv8 (IoU threshold: {iou_threshold})...")
        precision_v8, recall_v8, ap_v8 = evaluate_detections(
            pred_df_v8, gt_boxes, gt_labels, iou_threshold
        )
        print(f"YOLOv8 Average Precision: {ap_v8:.4f}")
        print(f"YOLOv8 Precision points: {len(precision_v8)}")
        
        # Evaluate YOLOv5
        print(f"\nEvaluating YOLOv5 (IoU threshold: {iou_threshold})...")
        precision_v5, recall_v5, ap_v5 = evaluate_detections(
            pred_df_v5, gt_boxes, gt_labels, iou_threshold
        )
        print(f"YOLOv5 Average Precision: {ap_v5:.4f}")
        print(f"YOLOv5 Precision points: {len(precision_v5)}")
        
        # =====================================================================
        # 5. GENERATE VISUALIZATIONS
        # =====================================================================
        print("\n" + "=" * 80)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Save annotated images (using transformed image)
        print("\nSaving YOLOv8 annotated image...")
        yolov8_output = f"{image_basename}_YOLOv8_annotated.jpg"
        save_annotated_image(transformed_path, pred_df_v8, yolov8_output, "YOLOv8")
        
        print("\nSaving YOLOv5 annotated image...")
        yolov5_output = f"{image_basename}_YOLOv5_annotated.jpg"
        save_annotated_image(transformed_path, pred_df_v5, yolov5_output, "YOLOv5")
        
        # Save prediction data
        print("\nSaving prediction data...")
        pred_df_v8.to_csv(f"{image_basename}_YOLOv8_predictions.csv", index=False)
        pred_df_v5.to_csv(f"{image_basename}_YOLOv5_predictions.csv", index=False)
        
        # Plot precision-recall curves
        print("\nGenerating precision-recall curve...")
        pr_data_v8 = {'precision': precision_v8, 'recall': recall_v8}
        pr_data_v5 = {'precision': precision_v5, 'recall': recall_v5}
        pr_plot_path = f"{image_basename}_precision_recall_comparison.png"
        plot_precision_recall_curve(pr_data_v8, pr_data_v5, ap_v8, ap_v5, pr_plot_path)
        
        # =====================================================================
        # 6. STORE RESULTS
        # =====================================================================
        print("\n" + "=" * 80)
        print("STORING RESULTS")
        print("=" * 80)
        
        # Model loading times
        resultMap.put("YOLOv8_LOAD_TIME", f"{yolov8_load_time:.4f}")
        resultMap.put("YOLOv5_LOAD_TIME", f"{yolov5_load_time:.4f}")
        
        # Inference times
        resultMap.put("YOLOv8_INFERENCE_TIME", f"{yolov8_inference_time:.2f}")
        resultMap.put("YOLOv5_INFERENCE_TIME", f"{yolov5_inference_time:.2f}")
        
        # Detection counts
        resultMap.put("YOLOv8_DETECTIONS", str(len(pred_df_v8)))
        resultMap.put("YOLOv5_DETECTIONS", str(len(pred_df_v5)))
        
        # Mean confidences
        yolov8_mean_conf = pred_df_v8['confidence'].mean() if len(pred_df_v8) > 0 else 0.0
        yolov5_mean_conf = pred_df_v5['confidence'].mean() if len(pred_df_v5) > 0 else 0.0
        resultMap.put("YOLOv8_MEAN_CONFIDENCE", f"{yolov8_mean_conf:.4f}")
        resultMap.put("YOLOv5_MEAN_CONFIDENCE", f"{yolov5_mean_conf:.4f}")
        
        # Evaluation metrics
        resultMap.put("YOLOv8_AP", f"{ap_v8:.4f}")
        resultMap.put("YOLOv5_AP", f"{ap_v5:.4f}")
        resultMap.put("YOLOv8_PRECISION_POINTS", str(len(precision_v8)))
        resultMap.put("YOLOv5_PRECISION_POINTS", str(len(precision_v5)))
        resultMap.put("IOU_THRESHOLD", f"{iou_threshold:.2f}")
        
        # Image transformation parameters (store the actual float values used)
        resultMap.put("IMAGE_RESOLUTION_SCALE", f"{resolution_scale:.2f}")
        resultMap.put("BLUR_KERNEL_SIZE", str(blur_kernel))
        resultMap.put("NOISE_LEVEL", f"{noise_level:.2f}")
        resultMap.put("BRIGHTNESS_FACTOR", f"{brightness_factor:.2f}")
        resultMap.put("CONTRAST_FACTOR", f"{contrast_factor:.2f}")
        resultMap.put("ROTATION_ANGLE", str(rotation_angle))
        
        # Ground truth info
        resultMap.put("GROUND_TRUTH_BOXES", str(len(gt_boxes)))
        
        # Image info
        resultMap.put("IMAGE_WIDTH", str(image_width))
        resultMap.put("IMAGE_HEIGHT", str(image_height))
        resultMap.put("IMAGE_PATH", image_path)
        
        # Output files
        resultMap.put("YOLOv8_ANNOTATED_IMAGE", yolov8_output)
        resultMap.put("YOLOv5_ANNOTATED_IMAGE", yolov5_output)
        resultMap.put("PRECISION_RECALL_PLOT", pr_plot_path)
        
        # Status
        resultMap.put("COMPARISON_COMPLETED", "true")
        
        # =====================================================================
        # 8. PERFORMANCE ANALYSIS METRICS
        # =====================================================================
        
        # Calculate performance metrics for analysis
        # These will help identify which transformations are most/least advantageous
        
        # Detection efficiency (detections per ground truth)
        gt_count = len(gt_boxes)
        detection_efficiency_v8 = len(pred_df_v8) / max(gt_count, 1) if gt_count > 0 else 0
        detection_efficiency_v5 = len(pred_df_v5) / max(gt_count, 1) if gt_count > 0 else 0
        
        resultMap.put("YOLOv8_DETECTION_EFFICIENCY", f"{detection_efficiency_v8:.4f}")
        resultMap.put("YOLOv5_DETECTION_EFFICIENCY", f"{detection_efficiency_v5:.4f}")
        
        # Model comparison metrics
        detection_advantage_v8 = len(pred_df_v8) - len(pred_df_v5)
        confidence_advantage_v8 = yolov8_mean_conf - yolov5_mean_conf
        ap_advantage_v8 = ap_v8 - ap_v5
        
        resultMap.put("DETECTION_ADVANTAGE_YOLOv8", str(detection_advantage_v8))
        resultMap.put("CONFIDENCE_ADVANTAGE_YOLOv8", f"{confidence_advantage_v8:.4f}")
        resultMap.put("AP_ADVANTAGE_YOLOv8", f"{ap_advantage_v8:.4f}")
        
        # Transformation impact indicators
        transformation_count = 0
        if resolution_scale != 1.0:
            transformation_count += 1
        if blur_kernel > 0:
            transformation_count += 1
        if noise_level > 0:
            transformation_count += 1
        if brightness_factor != 1.0:
            transformation_count += 1
        if contrast_factor != 1.0:
            transformation_count += 1
        if rotation_angle != 0:
            transformation_count += 1
            
        resultMap.put("TRANSFORMATION_COUNT", str(transformation_count))
        
        # Robustness score (higher is better - combines multiple metrics)
        robustness_v8 = (ap_v8 * 0.4 + detection_efficiency_v8 * 0.3 + yolov8_mean_conf * 0.3)
        robustness_v5 = (ap_v5 * 0.4 + detection_efficiency_v5 * 0.3 + yolov5_mean_conf * 0.3)
        
        resultMap.put("YOLOv8_ROBUSTNESS_SCORE", f"{robustness_v8:.4f}")
        resultMap.put("YOLOv5_ROBUSTNESS_SCORE", f"{robustness_v5:.4f}")
        
        # Overall model advantage
        overall_advantage_v8 = robustness_v8 - robustness_v5
        resultMap.put("OVERALL_ADVANTAGE_YOLOv8", f"{overall_advantage_v8:.4f}")
        
        print("\n" + "-" * 80)
        print("PERFORMANCE ANALYSIS:")
        print("-" * 80)
        print(f"Detection Efficiency - YOLOv8: {detection_efficiency_v8:.4f}, YOLOv5: {detection_efficiency_v5:.4f}")
        print(f"Detection Advantage YOLOv8: {detection_advantage_v8}")
        print(f"Confidence Advantage YOLOv8: {confidence_advantage_v8:.4f}")
        print(f"AP Advantage YOLOv8: {ap_advantage_v8:.4f}")
        print(f"Transformation Count: {transformation_count}")
        print(f"Robustness Score - YOLOv8: {robustness_v8:.4f}, YOLOv5: {robustness_v5:.4f}")
        print(f"Overall Advantage YOLOv8: {overall_advantage_v8:.4f}")
        
        # =====================================================================
        # 7. PRINT SUMMARY
        # =====================================================================
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(f"\nImage: {image_path}")
        print(f"Image Size: {image_width}x{image_height}")
        print(f"Ground Truth Boxes: {len(gt_boxes)}")
        print(f"IoU Threshold: {iou_threshold:.2f}")
        
        print("\n" + "-" * 80)
        print("Image Transformations Applied:")
        print("-" * 80)
        print(f"  Resolution Scale: {resolution_scale}x")
        print(f"  Blur Kernel: {blur_kernel}")
        print(f"  Noise Level: {noise_level}")
        print(f"  Brightness: {brightness_factor}x")
        print(f"  Contrast: {contrast_factor}x")
        print(f"  Rotation: {rotation_angle}°")
        
        print("\n" + "-" * 80)
        print("YOLOv8 Results:")
        print("-" * 80)
        print(f"  Load Time: {yolov8_load_time:.4f}s")
        print(f"  Inference Time: {yolov8_inference_time:.2f}ms")
        print(f"  Detections: {len(pred_df_v8)}")
        print(f"  Mean Confidence: {yolov8_mean_conf:.4f}")
        print(f"  Average Precision: {ap_v8:.4f}")
        
        print("\n" + "-" * 80)
        print("YOLOv5 Results:")
        print("-" * 80)
        print(f"  Load Time: {yolov5_load_time:.4f}s")
        print(f"  Inference Time: {yolov5_inference_time:.2f}ms")
        print(f"  Detections: {len(pred_df_v5)}")
        print(f"  Mean Confidence: {yolov5_mean_conf:.4f}")
        print(f"  Average Precision: {ap_v5:.4f}")
        
        print("\n" + "-" * 80)
        print("Output Files:")
        print("-" * 80)
        print(f"  YOLOv8 Annotated: {yolov8_output}")
        print(f"  YOLOv5 Annotated: {yolov5_output}")
        print(f"  PR Curve Plot: {pr_plot_path}")
        # Read YOLOv8 annotated image and convert to bytes
        if os.path.exists(yolov8_output):
            with open(yolov8_output, "rb") as f:
                yolov8_bytes = f.read()
            print(f"YOLOv8 Annotated image loaded as {len(yolov8_bytes)} bytes")
        else:
            yolov8_bytes = None
            print(f"⚠️ YOLOv8 annotated image not found at: {yolov8_output}")

        if os.path.exists(yolov5_output):
            with open(yolov5_output, "rb") as f:
                yolov5_output = f.read()
            print(f"YOLOv5 Annotated image loaded as {len(yolov5_output)} bytes")
        else:
            yolov5_output = None
            print(f"⚠️ YOLOv5 annotated image not found at: {yolov5_output}")

                
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("pred_df_v5:", pred_df_v5)
        
        # Read original image and ground truth for dataset
        with open(image_path, "rb") as f:
            original_image_bytes = f.read()
        print(f"Original image loaded: {os.path.basename(image_path)} ({len(original_image_bytes)} bytes)")
        
        with open(ground_truth_path, "rb") as f:
            ground_truth_bytes = f.read()
        print(f"Ground truth loaded: {os.path.basename(ground_truth_path)} ({len(ground_truth_bytes)} bytes)")
        
        # Read transformed image
        with open(transformed_path, "rb") as f:
            transformed_image_bytes = f.read()
        print(f"Transformed image loaded: {os.path.basename(transformed_path)} ({len(transformed_image_bytes)} bytes)")
        
        # Read PR plot
        with open(pr_plot_path, "rb") as f:
            pr_plot_bytes = f.read()
        print(f"PR plot loaded: {os.path.basename(pr_plot_path)} ({len(pr_plot_bytes)} bytes)")

        ph.save_datasets(variables, resultMap, "Crypto_desktop_samples",
                         [df_to_csv_bytes(pred_df_v5), df_to_csv_bytes(pred_df_v8), yolov8_bytes, yolov5_output, original_image_bytes, ground_truth_bytes, transformed_image_bytes, pr_plot_bytes],
                         ['pred_df_v5.csv', 'pred_df_v8.csv', 'yolov8_annotated.png', 'yolov5_annotated.png', 'original_image.jpg', 'ground_truth.csv', 'transformed_image.jpg', 'precision_recall_plot.png'])

        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print("ERROR IN MODEL COMPARISON")
        print("=" * 80)
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Store error state
        resultMap.put("COMPARISON_COMPLETED", "false")
        resultMap.put("ERROR", str(e))
        resultMap.put("YOLOv8_LOAD_TIME", "0.0")
        resultMap.put("YOLOv5_LOAD_TIME", "0.0")
        resultMap.put("YOLOv8_INFERENCE_TIME", "0.0")
        resultMap.put("YOLOv5_INFERENCE_TIME", "0.0")
        resultMap.put("YOLOv8_DETECTIONS", "0")
        resultMap.put("YOLOv5_DETECTIONS", "0")
        resultMap.put("YOLOv8_MEAN_CONFIDENCE", "0.0")
        resultMap.put("YOLOv5_MEAN_CONFIDENCE", "0.0")
        resultMap.put("YOLOv8_AP", "0.0")
        resultMap.put("YOLOv5_AP", "0.0")


if __name__ == '__main__':
    run_model_comparison()

