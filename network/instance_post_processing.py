# post_processing.py

import numpy as np
import cv2
from skimage.measure import label, regionprops
from scipy.stats import mode

def post_process(segmentation_mask, classification_logits, seg_threshold=0.5):
    """
    Post-process the outputs of the semantic segmentation and classification decoders.
    
    Args:
        segmentation_mask (np.ndarray): A 2D array of shape (H, W) with soft segmentation scores (0-1) for nuclei.
        classification_logits (np.ndarray): A 3D array of shape (H, W, num_classes) representing per-pixel logits or probabilities.
            It is assumed that class 0 corresponds to background.
        seg_threshold (float): Threshold to convert soft segmentation to binary mask.
    
    Returns:
        instance_masks (list of np.ndarray): A list of binary masks for each detected nucleus instance.
        instance_types (list of int): A list of predicted nucleus type (class index) for each instance.
    """
    # Step 1: Convert the soft segmentation mask to a binary mask.
    binary_mask = (segmentation_mask >= seg_threshold).astype(np.uint8)
    
    # Optional: perform morphological operations to remove noise and fill holes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Step 2: Label connected components (individual nuclei).
    labeled_mask = label(binary_mask)
    props = regionprops(labeled_mask)
    
    instance_masks = []
    instance_types = []
    
    # Prepare a classification map by taking argmax over classification logits per pixel.
    # Assuming classification_logits is either logits or probabilities.
    cls_map = np.argmax(classification_logits, axis=-1)  # shape: (H, W)
    
    # Step 3: For each connected component, determine the nucleus type by majority vote.
    for prop in props:
        # Create a mask for the current nucleus instance.
        instance_mask = (labeled_mask == prop.label).astype(np.uint8)
        instance_masks.append(instance_mask)
        
        # Get the pixel indices of this instance.
        coords = prop.coords  # array of shape (num_pixels, 2)
        
        # Extract the classification labels for these pixels.
        instance_cls = cls_map[coords[:, 0], coords[:, 1]]
        
        # Exclude background pixels if needed (assuming 0 is background).
        instance_cls = instance_cls[instance_cls != 0]
        
        if len(instance_cls) == 0:
            # If all pixels are background, assign background (0).
            nucleus_type = 0
        else:
            # Majority vote (mode) for the nucleus type.
            nucleus_type = mode(instance_cls, nan_policy='omit').mode[0]
        
        instance_types.append(nucleus_type)
    
    return instance_masks, instance_types


# Example usage:
if __name__ == '__main__':
    # Dummy data: 512x512 segmentation mask and classification logits for 4 classes.
    H, W = 512, 512
    num_classes = 4  # e.g., 0: background, 1: type A, 2: type B, 3: type C
    # Random soft segmentation mask
    seg_mask = np.random.rand(H, W)
    # Random classification logits (for simplicity, use probabilities here)
    cls_logits = np.random.rand(H, W, num_classes)
    
    # Run post-processing
    masks, types = post_process(seg_mask, cls_logits, seg_threshold=0.5)
    
    print("Detected instances:", len(masks))
    print("Assigned nucleus types:", types)
