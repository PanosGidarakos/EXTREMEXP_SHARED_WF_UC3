import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
import traceback

# Helper function to convert DataFrame to JSON
def df_to_json(df):
    """Convert DataFrame to JSON string."""
    if df.empty:
        return "[]"
    return df.to_json(orient='records')

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
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    except Exception as e:
        print(f"Error computing IoU: {e}")
        return 0.0

def evaluate_detections(pred_df, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Evaluates detections against ground truth annotations.
    """
    try:
        if pred_df.empty:
            print("No predictions to evaluate")
            return np.array([]), np.array([]), 0.0
        
        pred_df = pred_df.sort_values(by='confidence', ascending=False).reset_index(drop=True)
        predictions = []
        tp = []
        matched = [False] * len(gt_boxes)
        
        for idx, row in pred_df.iterrows():
            pred_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            pred_conf = row['confidence']
            pred_label = row['class']
            best_iou = 0
            best_gt_idx = -1
            
            # Find the best match among ground truth boxes
            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if pred_label != gt_label:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
            if best_iou >= iou_threshold and best_gt_idx != -1 and not matched[best_gt_idx]:
                tp.append(1)
                matched[best_gt_idx] = True
            else:
                tp.append(0)
            predictions.append(pred_conf)
            
        tp = np.array(tp)
        predictions = np.array(predictions)
        
        if len(tp) == 0:
            print("No true positives found")
            return np.array([]), np.array([]), 0.0
            
        precision, recall, thresholds = precision_recall_curve(tp, predictions)
        avg_precision = average_precision_score(tp, predictions)
        return precision, recall, avg_precision
        
    except Exception as e:
        print(f"Error in evaluate_detections: {e}")
        traceback.print_exc()
        return np.array([]), np.array([]), 0.0

def load_ground_truth_data(gt_csv_path):
    """
    Loads ground truth annotations from a CSV file in YOLO format.
    """
    try:
        print(f"Loading ground truth from: {gt_csv_path}")
        if not os.path.exists(gt_csv_path):
            print(f"Ground truth file not found: {gt_csv_path}")
            return [], []
            
        # Read CSV without headers (YOLO format)
        gt_df = pd.read_csv(gt_csv_path, header=None, sep=' ')
        print(f"Ground truth data shape: {gt_df.shape}")
        print(f"Ground truth columns: {gt_df.columns.tolist()}")
        
        # YOLO format: class x_center y_center width height (normalized)
        # Convert to xmin, ymin, xmax, ymax format
        gt_boxes = []
        gt_labels = []
        
        for idx, row in gt_df.iterrows():
            class_id = int(row[0])
            x_center = float(row[1])
            y_center = float(row[2])
            width = float(row[3])
            height = float(row[4])
            
            # Convert from YOLO format to xmin, ymin, xmax, ymax
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

def run_evaluation():
    """Main evaluation function."""
    try:
        print("Starting evaluation metrics calculation...")
        
        # Get input parameters
        gt_csv_paths = variables.get("GroundTruthPath")
        # IoU threshold comes as integer (e.g., 50 for 0.5) from the workflow
        iou_threshold_raw = variables.get("IoUThreshold")
        iou_threshold = float(iou_threshold_raw) / 100.0
        
        print(f"Ground truth paths: {gt_csv_paths}")
        print(f"IoU threshold raw: {iou_threshold_raw}, converted: {iou_threshold}")
        
        # Handle single ground truth path (string) or multiple paths (list)
        if isinstance(gt_csv_paths, str):
            gt_csv_paths = [gt_csv_paths]
        elif not isinstance(gt_csv_paths, list):
            gt_csv_paths = [str(gt_csv_paths)]
        
        print(f"Processing {len(gt_csv_paths)} ground truth files")
        
        # Get prediction data from previous task metrics
        # Note: In a real workflow, these would come from the previous task's metrics
        # For now, we'll try to load from CSV files as fallback
        try:
            pred_df_v8 = pd.read_csv("fire_YOLOv8_data.csv") if os.path.exists("fire_YOLOv8_data.csv") else pd.DataFrame()
            pred_df_v5 = pd.read_csv("fire_YOLOv5_data.csv") if os.path.exists("fire_YOLOv5_data.csv") else pd.DataFrame()
            print(f"YOLOv8 predictions loaded: {len(pred_df_v8)} rows")
            print(f"YOLOv5 predictions loaded: {len(pred_df_v5)} rows")
        except Exception as e:
            print(f"Error loading prediction data: {e}")
            traceback.print_exc()
            pred_df_v8 = pd.DataFrame()
            pred_df_v5 = pd.DataFrame()
        
        # Load and combine ground truth data from all files
        all_gt_boxes = []
        all_gt_labels = []
        
        for i, gt_csv_path in enumerate(gt_csv_paths):
            print(f"Loading ground truth from: {gt_csv_path}")
            gt_boxes, gt_labels = load_ground_truth_data(gt_csv_path)
            
            if gt_boxes:
                # Add image identifier to ground truth
                for j, box in enumerate(gt_boxes):
                    all_gt_boxes.append(box)
                    all_gt_labels.append(gt_labels[j] if j < len(gt_labels) else 0)
                print(f"Loaded {len(gt_boxes)} ground truth boxes from {gt_csv_path}")
            else:
                print(f"No ground truth data found in {gt_csv_path}")
        
        if not all_gt_boxes:
            print("No ground truth data found in any file, skipping evaluation")
            resultMap.put("EVALUATION_COMPLETED", "false")
            resultMap.put("ERROR", "No ground truth data")
            return
        
        # Calculate evaluation metrics
        print("Calculating YOLOv8 metrics...")
        precision_v8, recall_v8, ap_v8 = evaluate_detections(pred_df_v8, all_gt_boxes, all_gt_labels, iou_threshold)
        
        print("Calculating YOLOv5 metrics...")
        precision_v5, recall_v5, ap_v5 = evaluate_detections(pred_df_v5, all_gt_boxes, all_gt_labels, iou_threshold)
        
        # Store results
        resultMap.put("YOLOv8_AP", f"{ap_v8:.4f}")
        resultMap.put("YOLOv5_AP", f"{ap_v5:.4f}")
        resultMap.put("YOLOv8_PRECISION_POINTS", str(len(precision_v8)))
        resultMap.put("YOLOv5_PRECISION_POINTS", str(len(precision_v5)))
        resultMap.put("YOLOv8_RECALL_POINTS", str(len(recall_v8)))
        resultMap.put("YOLOv5_RECALL_POINTS", str(len(recall_v5)))
        resultMap.put("EVALUATION_COMPLETED", "true")
        resultMap.put("IOU_THRESHOLD", str(iou_threshold))
        
        # Store precision-recall data as JSON metrics
        if len(precision_v8) > 0:
            pr_data_v8 = pd.DataFrame({'precision': precision_v8, 'recall': recall_v8})
            resultMap.put("YOLOv8_PRECISION_RECALL_JSON", df_to_json(pr_data_v8))
            resultMap.put("YOLOv8_PR_DATA_SAVED", "true")
            print("YOLOv8 precision-recall data stored as JSON")
        else:
            resultMap.put("YOLOv8_PRECISION_RECALL_JSON", "[]")
            resultMap.put("YOLOv8_PR_DATA_SAVED", "false")
            print("No YOLOv8 precision-recall data to store")
            
        if len(precision_v5) > 0:
            pr_data_v5 = pd.DataFrame({'precision': precision_v5, 'recall': recall_v5})
            resultMap.put("YOLOv5_PRECISION_RECALL_JSON", df_to_json(pr_data_v5))
            resultMap.put("YOLOv5_PR_DATA_SAVED", "true")
            print("YOLOv5 precision-recall data stored as JSON")
        else:
            resultMap.put("YOLOv5_PRECISION_RECALL_JSON", "[]")
            resultMap.put("YOLOv5_PR_DATA_SAVED", "false")
            print("No YOLOv5 precision-recall data to store")
        
        print("Evaluation completed successfully!")
        print(f"YOLOv8 Average Precision (AP): {ap_v8:.4f}")
        print(f"YOLOv5 Average Precision (AP): {ap_v5:.4f}")
        
    except Exception as e:
        print(f"Critical error in run_evaluation: {e}")
        traceback.print_exc()
        resultMap.put("EVALUATION_COMPLETED", "false")
        resultMap.put("ERROR", str(e))

if __name__ == '__main__':
    run_evaluation()
