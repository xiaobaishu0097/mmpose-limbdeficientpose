# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob  # Import glob for finding files
import os
import re  # Import regular expressions
import shutil
import time
from typing import Any, Optional

# ...existing code...
import cv2
import json_tricks as json
import mmcv
import numpy as np
from rich.progress import track  # Import track

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def bbox_xyxy2xywh(bbox_xywh):
    """
    Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
      bbox_xywh: A list or tuple representing the bounding box in [x, y, w, h] format.

    Returns:
      A list representing the bounding box in [x1, y1, x2, y2] format.
    """
    x, y, w, h = bbox_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


def process_one_image(
    img,
    pose_estimator,
    detector: Optional[Any] = None,
    detection_results: Optional[np.ndarray] = None,
):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    if detection_results is not None:
        bboxes = detection_results.astype(np.float32)
    else:
        # predict bbox
        assert detector is not None, "Please provide a detector model."
        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
        )
        bboxes = bboxes[
            np.logical_and(
                pred_instance.labels == 0,
                pred_instance.scores > 0.3,
            )
        ]
        bboxes = bboxes[nms(bboxes, 0.3), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order="rgb")
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    # if there is no instance detected, return None
    return data_samples.get("pred_instances", None)


# Generate COCO format annotations
def generate_coco_pose_annotations(
    input_images,
    pred_instances_list,
    output_dir="./coco_dataset",
    original_gt_annos_list=None,  # Accept original GT annotations
    pose_masks_data=None,  # Add parameter for pose masks
):
    """Generate COCO format annotations for pose estimation.

    Args:
        input_images: List of input image paths or image data
        pred_instances_list: List of predicted instances containing keypoints for each image
        output_dir: Output directory for COCO dataset
        original_gt_annos_list: List of lists, where each inner list contains
                                 the original ground truth annotation dicts
                                 (including 'bbox', 'segmentation', 'track_id')
                                 for the corresponding image.
        pose_masks_data (dict, optional): A dictionary mapping track_id to its
                                          corresponding pose mask (list of 0s/1s).
                                          Defaults to None.
    Returns:
        dict: COCO format annotations
    """
    # Initialize pose_masks_data if None
    if pose_masks_data is None:
        pose_masks_data = {}

    # Get keypoint information from ldpose config
    try:
        # Import the configuration
        import os
        import sys

        project_root = os.path.abspath(os.path.dirname(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from mmengine.config import Config

        cfg = Config.fromfile("./configs/_base_/datasets/ldpose.py")
        dataset_info = cfg.dataset_info

        keypoint_names = list(dataset_info["keypoint_info"].keys())
        skeleton = []
        # Handle different possible formats of skeleton_info
        if isinstance(dataset_info["skeleton_info"], list):
            for connection in dataset_info["skeleton_info"]:
                if isinstance(connection, list) and len(connection) >= 2:
                    skeleton.append([connection[0], connection[1]])
                elif isinstance(connection, dict) and "link" in connection:
                    skeleton.append(connection["link"])
                elif hasattr(connection, "link"):
                    skeleton.append(connection.link)
        elif isinstance(dataset_info["skeleton_info"], dict):
            for connection_id, connection in dataset_info["skeleton_info"].items():
                if isinstance(connection, dict) and "link" in connection:
                    skeleton.append(connection["link"])
                elif isinstance(connection, list) and len(connection) >= 2:
                    skeleton.append(connection)
    except (ImportError, KeyError, FileNotFoundError):  # Added FileNotFoundError
        # Fallback if config import fails
        print(
            "Warning: Failed to load keypoint/skeleton info from config. Using fallback."
        )
        keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "head_top",
            "neck",
            "hip",
            "spine",
            "left_big_toe",
            "right_big_toe",
            "left_small_toe",
            "right_small_toe",
        ]
        skeleton = []

    coco_json = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": keypoint_names,
                "skeleton": skeleton,
            }
        ],
    }

    # Make sure input_images and pred_instances_list are lists
    if not isinstance(input_images, list):
        input_images = [input_images]
    if not isinstance(pred_instances_list, list):
        pred_instances_list = [pred_instances_list]
    if original_gt_annos_list is not None and not isinstance(
        original_gt_annos_list, list
    ):
        original_gt_annos_list = [
            original_gt_annos_list
        ]  # Ensure it's a list if provided

    # Make sure we have matching number of images and predictions
    assert len(input_images) == len(
        pred_instances_list
    ), "Number of images and predictions must match"
    if original_gt_annos_list is not None:
        assert len(input_images) == len(
            original_gt_annos_list
        ), "Number of images and original GT annotations must match"

    # Create COCO dataset directory structure
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    image_id = 0  # Start image_id from 0, will be incremented before use
    ann_id = 1

    # Process each image
    for img_idx, (input_image, pred_instances) in enumerate(
        zip(input_images, pred_instances_list)
    ):
        image_id = img_idx + 1  # Ensure image_id is frame index + 1

        # Process image info (Always add image entry)
        if isinstance(input_image, str):
            file_name = os.path.basename(input_image)
            img = cv2.imread(input_image)
            if img is None:
                print(f"Warning: Failed to read image {input_image}. Skipping.")
                image_id += 1  # Increment image_id even if skipped
                continue
            # Copy image to COCO dataset directory
            dest_img_path = os.path.join(images_dir, file_name)
            cv2.imwrite(dest_img_path, img)
        else:
            # Assuming input_image is numpy array if not string
            file_name = f"{img_idx:012d}.jpg"  # COCO-style filename format
            dest_img_path = os.path.join(images_dir, file_name)
            cv2.imwrite(dest_img_path, input_image)
            img = input_image

        height, width = img.shape[:2]

        # Add image information in COCO format using the correct image_id
        coco_json["images"].append(
            {
                "id": image_id,  # Use frame index + 1
                "file_name": file_name,
                "height": height,
                "width": width,
                "license": 1,  # Example license
                "date_captured": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime()
                ),  # Example date
            }
        )

        # Get corresponding original GT annotations for this image index, if available
        current_original_gt_annos = None
        gt_source_was_empty = False  # Flag for the specific condition

        if original_gt_annos_list is not None and img_idx < len(original_gt_annos_list):
            current_original_gt_annos = original_gt_annos_list[img_idx]
            # Check the specific condition: GT list exists, is empty, and no prediction resulted
            if (
                isinstance(current_original_gt_annos, list)
                and not current_original_gt_annos
                # and pred_instances is None
            ):
                gt_source_was_empty = True

        # --- Annotation Generation Logic ---

        if gt_source_was_empty:
            # Condition met: GT was provided but empty for this frame, and no prediction exists.
            # Do NOT generate any annotations for this image_id.
            # print(f"Image ID {image_id}: GT source was empty. No annotations generated.")
            pass  # Explicitly do nothing for annotations

        elif pred_instances is None:
            # No pose prediction, AND the above condition (gt_source_was_empty) is FALSE.
            # This means either GT wasn't provided, OR GT was provided AND was NOT empty,
            # but pose estimation still resulted in None (e.g., skipped due to segmentation, or failed).
            # If GT annotations *are* available here, create placeholders.
            if (
                current_original_gt_annos
            ):  # Check if GT annos exist for placeholder creation
                # print(f"Image ID {image_id}: No pose prediction, but non-empty GT annotations exist. Creating placeholder annotations.")
                for gt_anno in current_original_gt_annos:
                    # ... (logic to extract bbox, seg, track_id from gt_anno) ...
                    bbox_coco = [0.0, 0.0, 10.0, 10.0]
                    segmentation_coco = []
                    area = 100.0
                    track_id = gt_anno.get("track_id")

                    if "bbox" in gt_anno and gt_anno["bbox"] is not None:
                        try:
                            bbox_xyxy = gt_anno["bbox"]
                            if not isinstance(bbox_xyxy, np.ndarray):
                                bbox_xyxy = np.array(bbox_xyxy)
                            if bbox_xyxy.shape == (4,):
                                bbox_coco = bbox_xyxy2xywh(bbox_xyxy)
                                area = bbox_coco[2] * bbox_coco[3]
                        except Exception as e:
                            print(
                                f"Warning: Error converting GT bbox for placeholder annotation (Image ID: {image_id}): {e}"
                            )
                    if (
                        "segmentation" in gt_anno
                        and gt_anno["segmentation"] is not None
                    ):
                        segmentation_coco = gt_anno["segmentation"]

                    coco_json["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,  # Use frame index + 1
                            "category_id": 1,
                            "keypoints": (
                                [0.0] * len(keypoint_names) * 3
                            ),  # Zero keypoints
                            "score": 0.0,  # Zero score
                            "bbox": bbox_coco,
                            "area": area,
                            "iscrowd": 0,
                            "num_keypoints": 0,  # Zero keypoints
                            "segmentation": segmentation_coco,
                            "track_id": track_id,
                        }
                    )
                    ann_id += 1
                # else:
                # No prediction, and no GT annos (or GT wasn't provided). Do nothing.
                # print(f"Image ID {image_id}: No pose prediction and no corresponding GT annotations. Skipping annotation generation.")
                pass

        else:
            # pred_instances is NOT None. Generate annotations based on predictions.
            # print(f"Image ID {image_id}: Processing {len(pred_instances.keypoints)} predicted instances.")
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores

            # Check if keypoints or keypoint_scores are unexpectedly None
            if keypoints is None or keypoint_scores is None:
                print(
                    f"Warning: Image ID {image_id}: pred_instances exist but keypoints ({type(keypoints)}) or keypoint_scores ({type(keypoint_scores)}) is None. Skipping annotation generation for this image."
                )
                continue  # Skip to the next image_id

            # Ensure keypoints is iterable and has a length
            try:
                num_instances = len(keypoints)
            except TypeError:
                print(
                    f"Warning: Image ID {image_id}: pred_instances.keypoints is not iterable ({type(keypoints)}). Skipping annotation generation for this image."
                )
                continue  # Skip to the next image_id

            # Process each predicted pose instance
            for instance_idx in range(num_instances):
                # Get keypoints for this instance
                # Add checks for the content of keypoints/scores per instance
                try:
                    instance_keypoints = keypoints[instance_idx]
                    instance_scores = keypoint_scores[instance_idx]
                    if instance_keypoints is None or instance_scores is None:
                        print(
                            f"Warning: Image ID {image_id}, Instance {instance_idx}: instance_keypoints or instance_scores is None. Skipping this instance."
                        )
                        continue  # Skip to the next instance
                except IndexError:
                    print(
                        f"Warning: Image ID {image_id}: IndexError accessing instance {instance_idx} in keypoints/scores. Skipping this instance."
                    )
                    continue  # Skip to the next instance

                # --- Determine bbox, segmentation, and track_id ---
                bbox_coco = [0.0, 0.0, 10.0, 10.0]  # Default bbox [x,y,w,h]
                segmentation_coco = []  # Default empty segmentation
                area = 100.0  # Default area
                track_id = None  # Default track_id
                gt_anno_found = False

                # Try to get bbox, segmentation, and track_id from original ground truth
                if current_original_gt_annos is not None and instance_idx < len(
                    current_original_gt_annos
                ):
                    # ... (existing logic to get GT data and set gt_anno_found = True) ...
                    gt_anno = current_original_gt_annos[instance_idx]
                    track_id = gt_anno.get("track_id")  # Get track_id from GT

                    if "bbox" in gt_anno and gt_anno["bbox"] is not None:
                        try:
                            bbox_xyxy = gt_anno["bbox"]
                            if not isinstance(bbox_xyxy, np.ndarray):
                                bbox_xyxy = np.array(bbox_xyxy)
                            if bbox_xyxy.shape == (4,):
                                bbox_coco = bbox_xyxy2xywh(bbox_xyxy)
                                area = bbox_coco[2] * bbox_coco[3]
                                gt_anno_found = True  # Mark that we used GT bbox
                            # ... (else print warning) ...
                        except Exception as e:
                            # ... (print warning) ...
                            pass  # Keep default bbox

                    if (
                        "segmentation" in gt_anno
                        and gt_anno["segmentation"] is not None
                    ):
                        segmentation_coco = gt_anno["segmentation"]

                # Initialize lists for COCO keypoints and valid keypoints
                valid_kpts = []
                coco_keypoints = []
                num_keypoints = 0

                # --- Get pose mask for the current track_id ---
                current_pose_mask = None
                if track_id is not None and track_id in pose_masks_data:
                    # ... (existing logic to get current_pose_mask) ...
                    current_pose_mask = pose_masks_data[track_id]
                    # ... (validation of mask format) ...

                # Convert to COCO format keypoints, applying pose mask logic
                # Check if instance_keypoints is iterable before looping
                try:
                    iterator = enumerate(instance_keypoints)
                except TypeError:
                    print(
                        f"Warning: Image ID {image_id}, Instance {instance_idx}: instance_keypoints is not iterable ({type(instance_keypoints)}). Skipping keypoint processing for this instance."
                    )
                    # Keep coco_keypoints empty, num_keypoints=0
                else:
                    for kpt_idx, kpt in iterator:
                        # ... (existing logic to calculate x, y, score, v) ...
                        # ... (handle potential errors accessing kpt[0], kpt[1]) ...
                        try:
                            x, y = kpt[0], kpt[1]
                            score = instance_scores[
                                kpt_idx
                            ]  # Check instance_scores length?
                            # Visibility flag v: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                            v_score_based = (
                                2 if score > 0.6 else (1 if score >= 0.3 else 0)
                            )

                            # Apply pose mask if available and valid
                            if current_pose_mask is not None:
                                # Add check for kpt_idx bounds in mask
                                if kpt_idx < len(current_pose_mask):
                                    mask_value = current_pose_mask[kpt_idx]
                                    if mask_value == 0:
                                        v = 0
                                    else:
                                        v = v_score_based
                                else:
                                    print(
                                        f"Warning: Image ID {image_id}, Instance {instance_idx}: kpt_idx {kpt_idx} out of bounds for pose mask length {len(current_pose_mask)}. Using score-based visibility."
                                    )
                                    v = v_score_based
                            else:
                                v = v_score_based

                            if v > 0:
                                valid_kpts.append((float(x), float(y)))
                                num_keypoints += 1
                            coco_keypoints.extend([float(x), float(y), int(v)])
                        except (IndexError, TypeError) as e_kpt:
                            print(
                                f"Warning: Image ID {image_id}, Instance {instance_idx}, Kpt {kpt_idx}: Error processing keypoint data ({kpt}, score index {kpt_idx}): {e_kpt}. Skipping this keypoint."
                            )

                # --- Recalculate BBox from valid keypoints if GT wasn't used ---
                # This block is now definitely after instance_keypoints loop and uses valid_kpts
                if not gt_anno_found:
                    if valid_kpts:  # Use the populated valid_kpts list
                        try:
                            x_coords = [k[0] for k in valid_kpts]
                            y_coords = [k[1] for k in valid_kpts]
                            x_min, y_min, x_max, y_max = (
                                min(x_coords),
                                min(y_coords),
                                max(x_coords),
                                max(y_coords),
                            )
                            bbox_coco = [
                                float(x_min),
                                float(y_min),
                                float(x_max - x_min),
                                float(y_max - y_min),
                            ]
                            area = bbox_coco[2] * bbox_coco[3]
                            segmentation_coco = (
                                []
                            )  # Ensure segmentation is empty if calculated
                        except Exception as e_bbox:
                            print(
                                f"Warning: Image ID {image_id}, Instance {instance_idx}: Error calculating bbox from keypoints: {e_bbox}. Using default bbox."
                            )
                            bbox_coco = [
                                0.0,
                                0.0,
                                10.0,
                                10.0,
                            ]  # Reset to default on error
                            area = 100.0
                            segmentation_coco = []

                    # else: If no valid_kpts, bbox_coco remains default

                # Calculate overall score as average of keypoint scores
                # Check instance_scores type and length before calculating mean
                score = 0.0
                try:
                    if (
                        isinstance(instance_scores, (np.ndarray, list))
                        and len(instance_scores) > 0
                    ):
                        score = float(np.mean(instance_scores))
                except Exception as e_score:
                    print(
                        f"Warning: Image ID {image_id}, Instance {instance_idx}: Error calculating mean score from instance_scores ({type(instance_scores)}): {e_score}"
                    )

                # Add annotation in COCO format
                coco_json["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,  # Use frame index + 1
                        "category_id": 1,
                        "keypoints": coco_keypoints,
                        "score": score,
                        "bbox": bbox_coco,
                        "area": area,
                        "iscrowd": 0,
                        "num_keypoints": num_keypoints,
                        "segmentation": segmentation_coco,
                        "track_id": track_id,
                    }
                )
                ann_id += 1

        # No need to increment image_id here, handled by loop

    # Add license information (standard in COCO)
    coco_json["licenses"] = [{"id": 1, "name": "Attribution-NonCommercial", "url": ""}]

    # Add dataset information
    coco_json["info"] = {
        "description": "Generated pose dataset in COCO format",
        "url": "",
        "version": "1.0",
        "year": int(time.strftime("%Y", time.localtime())),
        "contributor": "mmpose-demo.py",  # Updated contributor
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    # Save to file
    annotation_file = os.path.join(annotations_dir, "person_keypoints.json")
    # Use json_tricks for potentially handling numpy types if they sneak in
    try:
        with open(annotation_file, "w") as f:
            json.dump(coco_json, f, indent=2)
    except TypeError as e:
        print(
            f"Warning: Encountered TypeError during JSON dump: {e}. Attempting standard json."
        )
        import json as std_json

        with open(annotation_file, "w") as f:
            std_json.dump(coco_json, f, indent=2)

    print(f"COCO format dataset saved to {output_dir}")
    print(f"  - Images: {len(coco_json['images'])}")
    print(f"  - Annotations: {len(coco_json['annotations'])}")
    print(f"  - Categories: {len(coco_json['categories'])}")
    return coco_json


def bbox_xywh_to_xyxy(bbox_xywh):
    """
    Converts bounding box format from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
      bbox_xywh: A list or tuple representing the bounding box in [x, y, w, h] format.

    Returns:
      A list representing the bounding box in [x1, y1, x2, y2] format.
    """
    x, y, w, h = bbox_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]


def post_process_limb_scores(keypoint_scores):
    """
    Applies post-processing logic to keypoint confidence scores based on limb groups.

    Args:
        keypoint_scores (np.ndarray): Array of keypoint scores for a single instance.

    Returns:
        np.ndarray: Modified keypoint scores.
    """
    if keypoint_scores is None or len(keypoint_scores) == 0:
        return keypoint_scores

    modified_scores = keypoint_scores.copy()
    groups = [
        [7, 9, 17, 19],  # Group 1
        [8, 10, 18, 20],  # Group 2
        [13, 15, 21, 23],  # Group 3
        [14, 16, 22, 24],  # Group 4
    ]

    for group_idx, group in enumerate(groups):
        # Ensure all indices are within the bounds of the scores array
        if any(idx >= len(modified_scores) for idx in group):
            print(
                f"Warning: Group {group_idx+1} indices {group} out of bounds for scores length {len(modified_scores)}. Skipping group."
            )
            continue

        group_scores = modified_scores[group]

        # Skip if group scores are empty (shouldn't happen with check above, but safety)
        if len(group_scores) == 0:
            continue

        max_idx_in_group = np.argmax(group_scores)
        max_original_idx = group[max_idx_in_group]

        # Apply logic based on the group and the index of the max score
        if group_idx == 0:  # Group [7, 9, 17, 19]
            idx_7, idx_9, idx_17, idx_19 = group[0], group[1], group[2], group[3]
            if max_original_idx == idx_7:
                modified_scores[idx_17] = 0
                if modified_scores[idx_9] < modified_scores[idx_19]:
                    modified_scores[idx_9] = 0
                else:
                    modified_scores[idx_19] = 0
            elif max_original_idx == idx_17:
                modified_scores[idx_7] = 0
                modified_scores[idx_9] = 0
                modified_scores[idx_19] = 0
            elif max_original_idx == idx_9:
                modified_scores[idx_17] = 0
                modified_scores[idx_19] = 0
            elif max_original_idx == idx_19:
                modified_scores[idx_9] = 0
                modified_scores[idx_17] = 0

        elif group_idx == 1:  # Group [8, 10, 18, 20] - Apply analogous logic
            idx_8, idx_10, idx_18, idx_20 = group[0], group[1], group[2], group[3]
            if max_original_idx == idx_8:
                modified_scores[idx_18] = 0
                if modified_scores[idx_10] < modified_scores[idx_20]:
                    modified_scores[idx_10] = 0
                else:
                    modified_scores[idx_20] = 0
            elif max_original_idx == idx_18:
                modified_scores[idx_8] = 0
                modified_scores[idx_10] = 0
                modified_scores[idx_20] = 0
            elif max_original_idx == idx_10:
                modified_scores[idx_18] = 0
                modified_scores[idx_20] = 0
            elif max_original_idx == idx_20:
                modified_scores[idx_10] = 0
                modified_scores[idx_18] = 0

        elif group_idx == 2:  # Group [13, 15, 21, 23] - Apply analogous logic
            idx_13, idx_15, idx_21, idx_23 = group[0], group[1], group[2], group[3]
            if max_original_idx == idx_13:
                modified_scores[idx_21] = 0
                if modified_scores[idx_15] < modified_scores[idx_23]:
                    modified_scores[idx_15] = 0
                else:
                    modified_scores[idx_23] = 0
            elif max_original_idx == idx_21:
                modified_scores[idx_13] = 0
                modified_scores[idx_15] = 0
                modified_scores[idx_23] = 0
            elif max_original_idx == idx_15:
                modified_scores[idx_21] = 0
                modified_scores[idx_23] = 0
            elif max_original_idx == idx_23:
                modified_scores[idx_15] = 0
                modified_scores[idx_21] = 0

        elif group_idx == 3:  # Group [14, 16, 22, 24] - Apply analogous logic
            idx_14, idx_16, idx_22, idx_24 = group[0], group[1], group[2], group[3]
            if max_original_idx == idx_14:
                modified_scores[idx_22] = 0
                if modified_scores[idx_16] < modified_scores[idx_24]:
                    modified_scores[idx_16] = 0
                else:
                    modified_scores[idx_24] = 0
            elif max_original_idx == idx_22:
                modified_scores[idx_14] = 0
                modified_scores[idx_16] = 0
                modified_scores[idx_24] = 0
            elif max_original_idx == idx_16:
                modified_scores[idx_22] = 0
                modified_scores[idx_24] = 0
            elif max_original_idx == idx_24:
                modified_scores[idx_16] = 0
                modified_scores[idx_22] = 0

    return modified_scores


def process_single_video(
    video_path: str,
    args: argparse.Namespace,
    detector: Optional[Any],
    pose_estimator: Any,
):
    """Processes a single video file, dynamically looking for its ground truth and pose mask."""  # Updated docstring
    video_name = os.path.basename(video_path)
    video_base_name = os.path.splitext(video_name)[0]
    print(f"\n--- Processing Video: {video_name} ---")

    # --- Dynamic Ground Truth and Pose Mask Loading ---
    gt_base_dir = "/home/hemingdu/Data/LimbDeficientPose-Video-Dataset/annotations/"
    gt_dir_path = os.path.join(
        gt_base_dir, video_base_name
    )  # Directory for this video's annotations
    gt_file_path = os.path.join(gt_dir_path, "instances_default.json")
    pose_mask_file_path = os.path.join(
        gt_dir_path, "pose_mask.json"
    )  # Path for pose_mask.json

    local_detection_ground_truths = None
    local_image_ids_with_segmentation = None
    local_has_track_id = False
    use_gt_for_this_video = False
    local_pose_masks = {}  # Dictionary to store pose masks {track_id: mask_list}
    # --- Add mappings for filename lookup ---
    filename_to_image_id = {}
    filename_to_gt_annos = {}
    # ---------------------------------------

    print(f"Checking for ground truth file: {gt_file_path}")
    try:
        with open(gt_file_path, "r") as f:
            data = json.load(f)

        # --- Create image_id to filename map AND filename to image_id map ---
        image_id_to_filename = {}
        if "images" in data and isinstance(data["images"], list):
            for img_info in data["images"]:
                if (
                    isinstance(img_info, dict)
                    and "id" in img_info
                    and "file_name" in img_info
                ):
                    img_id = img_info["id"]
                    fname = img_info["file_name"]
                    image_id_to_filename[img_id] = fname
                    filename_to_image_id[fname] = img_id  # Add reverse mapping
            print(
                f"  Built map for {len(image_id_to_filename)} image IDs to filenames."
            )
        # -----------------------------------------

        # Parse the loaded data and populate filename_to_gt_annos
        parsed_detection_ground_truths = (
            {}
        )  # Keep this for potential other uses if needed
        parsed_image_ids_with_segmentation = set()
        parsed_has_track_id = False
        raw_annotations = data.get("annotations", [])
        processed_annos = 0
        skipped_annos_bbox = 0
        skipped_annos_no_image_id = 0  # Track annotations missing image_id
        skipped_annos_no_filename = (
            0  # Track annotations where filename couldn't be found
        )
        skipped_annos_frame_parse_fail = (
            0  # Track annotations where frame number parsing failed
        )

        for item in raw_annotations:
            image_id = item.get("image_id")
            if image_id is None:
                skipped_annos_no_image_id += 1
                continue

            # --- Find filename using image_id --- # Modified section
            file_name = image_id_to_filename.get(image_id)
            if file_name is None:
                skipped_annos_no_filename += 1
                continue  # Cannot process annotation without a filename mapping
            # --- Frame number extraction (remains the same, but less critical now) ---
            frame_number = None
            if file_name:
                try:
                    # Use regex to find numbers in the filename base (more robust)
                    # Handles "frame_1.jpg", "frame_001.jpg", "video_frame_123.png" etc.
                    base = os.path.splitext(file_name)[0]  # Remove extension
                    match = re.search(
                        r"(\d+)$", base
                    )  # Find digits at the end of the base name
                    if match:
                        frame_number = int(match.group(1))
                    else:
                        # Fallback: try splitting by '_' if regex fails
                        num_part = base.split("_")[-1]
                        if num_part.isdigit():
                            frame_number = int(num_part)
                        else:
                            skipped_annos_frame_parse_fail += 1
                            # print(f"Warning: Could not parse frame number from filename '{file_name}' for image_id {image_id}")
                except Exception as e:
                    skipped_annos_frame_parse_fail += 1
                    # print(f"Warning: Error parsing frame number from filename '{file_name}' for image_id {image_id}: {e}")
            else:
                skipped_annos_no_filename += 1
                # print(f"Warning: Could not find filename for image_id {image_id} in GT images list.")
            # ------------------------------------

            # Populate filename_to_gt_annos instead of parsed_detection_ground_truths by image_id
            if file_name not in filename_to_gt_annos:
                filename_to_gt_annos[file_name] = []

            bbox_xywh = item.get("bbox")
            bbox_xyxy = None
            if bbox_xywh and len(bbox_xywh) == 4:
                try:
                    bbox_xyxy = bbox_xywh_to_xyxy(bbox_xywh)
                except Exception:
                    bbox_xywh = None  # Mark as invalid on conversion error
            else:
                skipped_annos_bbox += 1
                bbox_xywh = None  # Mark as invalid/missing

            segmentation = item.get("segmentation")
            track_id = item.get("track_id")
            if track_id is not None:
                parsed_has_track_id = True

            # Only add if bbox was valid
            if bbox_xyxy is not None:
                # Add to filename_to_gt_annos
                filename_to_gt_annos[file_name].append(
                    {
                        "bbox": np.array(bbox_xyxy, dtype=np.float32),
                        "segmentation": segmentation,
                        "track_id": track_id,
                        "image_id": image_id,  # Store image_id for later use
                        # "frame_number": frame_number, # Keep if needed
                        # "file_name": file_name, # Redundant key
                    }
                )
                processed_annos += 1
                if segmentation and segmentation != []:
                    parsed_image_ids_with_segmentation.add(
                        image_id
                    )  # Still track by image_id
            # If bbox was invalid, we don't add the annotation

        if processed_annos > 0:  # Check if we actually processed any valid annotations
            # local_detection_ground_truths = parsed_detection_ground_truths # No longer the primary source
            local_image_ids_with_segmentation = parsed_image_ids_with_segmentation
            local_has_track_id = parsed_has_track_id
            use_gt_for_this_video = True
            print(
                f"Successfully loaded and processed ground truth for {video_name} from {gt_file_path}."
            )
            print(
                f"  Loaded GT for {len(filename_to_gt_annos)} filenames from {processed_annos} valid annotations."
            )
            if skipped_annos_bbox > 0:
                print(
                    f"    Skipped {skipped_annos_bbox} annotations due to missing/invalid bbox during loading."
                )
            if skipped_annos_no_image_id > 0:
                print(
                    f"    Skipped {skipped_annos_no_image_id} annotations due to missing 'image_id'."
                )
            if skipped_annos_no_filename > 0:
                print(
                    f"    Skipped {skipped_annos_no_filename} annotations because their image_id was not found in the 'images' list."
                )
            if skipped_annos_frame_parse_fail > 0:
                print(
                    f"    Failed to parse frame number from filename for {skipped_annos_frame_parse_fail} annotations."
                )
            print(
                f"  Found segmentation data for {len(local_image_ids_with_segmentation)} images."
            )
            if local_has_track_id:
                print("  Ground truth contains track_id information.")
        else:
            print(
                f"Ground truth file {gt_file_path} loaded but contained no annotations with valid bboxes. Using detector."
            )
            use_gt_for_this_video = False

    except FileNotFoundError:
        print(f"Ground truth file not found for {video_name}. Using detector.")
        use_gt_for_this_video = False
    except json.JSONDecodeError as e:
        print(
            f"Error decoding JSON from ground truth file {gt_file_path}: {e}. Using detector."
        )
        use_gt_for_this_video = False
    except Exception as e:
        print(
            f"Error loading or processing ground truth file {gt_file_path}: {e}. Using detector."
        )
        use_gt_for_this_video = False

    # --- Load Pose Mask if GT was loaded and has track_id ---
    if use_gt_for_this_video and local_has_track_id:
        print(f"Checking for pose mask file: {pose_mask_file_path}")
        try:
            with open(pose_mask_file_path, "r") as f_mask:
                pose_mask_list = json.load(f_mask)
            if isinstance(pose_mask_list, list):
                loaded_masks = 0
                skipped_masks_no_tid = 0
                skipped_masks_invalid = 0
                for mask_item in pose_mask_list:
                    if (
                        isinstance(mask_item, dict)
                        and "track_id" in mask_item
                        and "pose_mask" in mask_item
                    ):
                        tid = mask_item["track_id"]
                        mask = mask_item["pose_mask"]
                        # Basic validation (can be enhanced)
                        if isinstance(mask, list) and all(
                            isinstance(x, int) for x in mask
                        ):
                            local_pose_masks[tid] = mask
                            loaded_masks += 1
                        else:
                            print(
                                f"Warning: Invalid pose_mask format for track_id {tid} in {pose_mask_file_path}. Skipping."
                            )
                            skipped_masks_invalid += 1
                    else:
                        skipped_masks_no_tid += 1
                print(
                    f"  Successfully loaded {loaded_masks} pose masks from {pose_mask_file_path}."
                )
                if skipped_masks_no_tid > 0:
                    print(
                        f"    Skipped {skipped_masks_no_tid} items lacking 'track_id' or 'pose_mask'."
                    )
                if skipped_masks_invalid > 0:
                    print(
                        f"    Skipped {skipped_masks_invalid} items with invalid mask format."
                    )
            else:
                print(
                    f"Warning: Expected a list in {pose_mask_file_path}, but got {type(pose_mask_list)}. Pose masks not loaded."
                )
        except FileNotFoundError:
            print(
                f"  Pose mask file not found: {pose_mask_file_path}. Proceeding without pose masks."
            )
        except json.JSONDecodeError as e_mask:
            print(
                f"  Error decoding JSON from pose mask file {pose_mask_file_path}: {e_mask}. Pose masks not loaded."
            )
        except Exception as e_mask:
            print(
                f"  Error loading pose mask file {pose_mask_file_path}: {e_mask}. Pose masks not loaded."
            )
    elif use_gt_for_this_video and not local_has_track_id:
        print(
            "  Ground truth loaded but lacks track_id information. Pose masks cannot be applied."
        )

    # --- End Dynamic Loading ---

    # read the frames from the video
    print(f"Reading video file: {video_path}")
    # ... (rest of video reading logic remains the same) ...
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Failed to open video file: {video_path}. Skipping.")
        return
    frames = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has approximately {frame_count} frames.")
    read_start_time = time.time()
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    read_end_time = time.time()
    print(
        f"Read {len(frames)} frames from video in {read_end_time - read_start_time:.2f} seconds."
    )
    if not frames:
        print("Error: No frames read from video. Skipping.")
        return

    # Create a unique temporary directory for this video's frames
    temp_dir = os.path.join("./tmp_frames/", video_base_name)  # Unique temp dir
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Using temporary directory for frames: {temp_dir}")

    # Lists to store image paths, predictions, and ORIGINAL GT annotations for this video
    image_paths = []
    all_pred_instances = []
    all_original_gt_annos = (
        []
    )  # This will store GT annos ONLY if use_gt_for_this_video is True

    print(f"Processing {len(frames)} video frames for {video_name}...")
    process_start_time = time.time()
    for i, frame in enumerate(track(frames, description=f"Processing {video_name}")):
        # --- Generate filename matching GT format --- # Modified section
        # current_image_id = i + 1 # No longer assume this
        frame_filename = f"{video_name}_{i:08d}.jpg"  # Use the GT filename format
        frame_path = os.path.join(temp_dir, frame_filename)

        cv2.imwrite(frame_path, frame)
        image_paths.append(frame_path)

        # --- Determine ORIGINAL ground truth annotations for this frame using filename --- # Modified section
        original_gt_annos_for_frame = []
        current_image_id_from_gt = None  # Store the image_id if found
        gt_exists_for_frame = False  # Flag to check if GT was found for this filename

        if use_gt_for_this_video:
            # Fetch annotations using the generated frame_filename
            if frame_filename in filename_to_gt_annos:
                original_gt_annos_for_frame = filename_to_gt_annos[frame_filename]
                gt_exists_for_frame = True  # GT file has an entry for this frame
                # Check if the list is actually empty
                if original_gt_annos_for_frame:
                    current_image_id_from_gt = original_gt_annos_for_frame[0].get(
                        "image_id"
                    )
            # else: gt_exists_for_frame remains False, original_gt_annos_for_frame is empty

        # Store the original annos (or empty list) - needed for COCO generation later
        all_original_gt_annos.append(original_gt_annos_for_frame)

        # --- NEW LOGIC: Skip pose estimation if using GT and GT is empty for this frame ---
        if (
            use_gt_for_this_video
            and gt_exists_for_frame
            and not original_gt_annos_for_frame
        ):
            # print(f"Frame {i} (Filename {frame_filename}): GT exists but is empty. Skipping pose estimation.")
            all_pred_instances.append(None)  # Store None for prediction
            continue  # Skip the rest of the loop for this frame

        # --- Skipping logic based on segmentation (only if GT is used for this video) ---
        # This logic now only runs if the frame wasn't skipped above
        should_skip_segmentation = False
        if use_gt_for_this_video and local_image_ids_with_segmentation is not None:
            if (
                current_image_id_from_gt is not None
                and current_image_id_from_gt not in local_image_ids_with_segmentation
            ):
                # print(f"Frame {i} (Filename {frame_filename}, Image ID {current_image_id_from_gt}): GT found but lacks segmentation. Skipping pose estimation.")
                should_skip_segmentation = True
                # elif current_image_id_from_gt is None and gt_exists_for_frame:
                # This case means GT existed for the frame, but we couldn't get an image_id (shouldn't happen with current loading)
                # print(f"Warning: Frame {i} (Filename {frame_filename}): GT found but image_id missing in annotation. Skipping pose estimation.")
                # should_skip_segmentation = True # Or handle differently? For now, skip.
                # elif not gt_exists_for_frame:
                # GT wasn't found for this frame, segmentation check doesn't apply based on GT
                pass

        if should_skip_segmentation:
            all_pred_instances.append(None)
            continue  # Skip the rest of the loop for this frame

        # --- Determine BBoxes to use for Pose Estimation ---
        # This logic only runs if the frame wasn't skipped by the empty GT check or segmentation check
        bboxes_to_use = None
        detector_to_use = detector  # Default to using the detector passed in

        if (
            use_gt_for_this_video and gt_exists_for_frame
        ):  # We know original_gt_annos_for_frame is NOT empty here
            detector_to_use = None  # Don't use detector if using GT for this frame
            valid_gt_annos_for_frame = (
                original_gt_annos_for_frame  # Use the list fetched earlier
            )

            # Extract valid bboxes [x1, y1, x2, y2]
            original_valid_gt_bboxes = np.array(
                [anno["bbox"] for anno in valid_gt_annos_for_frame],
                dtype=np.float32,
            )

            # Decide whether to merge based on track_id
            if local_has_track_id:
                # Group annotations by track_id
                grouped_annos = {}
                annos_without_tid = []
                for anno in valid_gt_annos_for_frame:
                    tid = anno.get("track_id")
                    if tid is not None:
                        if tid not in grouped_annos:
                            grouped_annos[tid] = []
                        grouped_annos[tid].append(anno)
                    else:
                        annos_without_tid.append(anno)

                merged_bboxes_list = []
                merged_track_ids = set()  # Keep track of merged track IDs
                if grouped_annos:
                    for tid, annos_in_group in grouped_annos.items():
                        if not annos_in_group:
                            continue
                        bboxes_in_group = np.array([a["bbox"] for a in annos_in_group])
                        min_x1, min_y1 = np.min(bboxes_in_group[:, :2], axis=0)
                        max_x2, max_y2 = np.max(bboxes_in_group[:, 2:], axis=0)
                        merged_bboxes_list.append([min_x1, min_y1, max_x2, max_y2])
                        merged_track_ids.add(
                            tid
                        )  # Record the track ID associated with the merged bbox

                # Add bboxes for annotations that didn't have a track_id
                for anno in annos_without_tid:
                    merged_bboxes_list.append(anno["bbox"])

                if merged_bboxes_list:
                    bboxes_to_use = np.array(merged_bboxes_list, dtype=np.float32)
                else:
                    bboxes_to_use = np.array([]).reshape(0, 4)
            else:
                bboxes_to_use = original_valid_gt_bboxes

            if bboxes_to_use is None or bboxes_to_use.shape[0] == 0:
                bboxes_to_use = np.array([]).reshape(0, 4)
        # else: (Not using GT, or GT file didn't contain this frame)
        # detector_to_use remains detector
        # bboxes_to_use remains None

        # --- Process image: Run Pose Estimation ---
        # This only runs if not skipped above, and if either detector is present or bboxes_to_use is non-empty
        pred_instances = None  # Default to None
        if detector_to_use is not None or (
            bboxes_to_use is not None and bboxes_to_use.shape[0] > 0
        ):
            pred_instances = process_one_image(
                frame_path,
                pose_estimator,
                detector=detector_to_use,
                detection_results=bboxes_to_use,
            )
        # else:
        # print(f"Frame {i}: Skipping process_one_image as no detector and no GT bboxes provided.")

        # --- Apply Keypoint Score Post-processing ---
        if pred_instances is not None:
            try:
                # Check if keypoint_scores attribute exists and is iterable
                if (
                    hasattr(pred_instances, "keypoint_scores")
                    and pred_instances.keypoint_scores is not None
                ):
                    num_instances = len(pred_instances.keypoint_scores)
                    for instance_idx in range(num_instances):
                        original_scores = pred_instances.keypoint_scores[instance_idx]
                        # Ensure scores are numpy array for processing
                        if not isinstance(original_scores, np.ndarray):
                            original_scores = np.array(original_scores)

                        modified_scores = post_process_limb_scores(original_scores)
                        # Update the scores in the pred_instances object
                        pred_instances.keypoint_scores[instance_idx] = modified_scores
                else:
                    print(
                        f"Warning: pred_instances for frame {i} lacks 'keypoint_scores' or it is None. Skipping post-processing."
                    )
            except Exception as e:
                print(f"Error during keypoint post-processing for frame {i}: {e}")
        # --- End Post-processing ---

        all_pred_instances.append(pred_instances)  # Store the result (could be None)

    process_end_time = time.time()
    print(
        f"Finished processing {len(frames)} frames for {video_name} in {process_end_time - process_start_time:.2f} seconds."
    )

    # ... (Assertion remains the same) ...
    assert (
        len(image_paths) == len(all_pred_instances) == len(all_original_gt_annos)
    ), f"Mismatch in list lengths for {video_name}: images={len(image_paths)}, preds={len(all_pred_instances)}, gt_annos={len(all_original_gt_annos)}"

    # Generate COCO format dataset for this video
    print(f"Generating COCO format dataset for {video_name}...")
    # ... (COCO generation logic remains the same, using all_original_gt_annos) ...
    generate_start_time = time.time()
    coco_dataset_dir = os.path.join(args.output, f"coco_dataset_{video_base_name}")
    os.makedirs(coco_dataset_dir, exist_ok=True)
    coco_annotations = generate_coco_pose_annotations(
        image_paths,
        all_pred_instances,
        output_dir=coco_dataset_dir,
        original_gt_annos_list=all_original_gt_annos,  # Pass the potentially empty list
        pose_masks_data=local_pose_masks,  # Pass the loaded pose masks
    )
    generate_end_time = time.time()
    print(
        f"COCO dataset generation for {video_name} finished in {generate_end_time - generate_start_time:.2f} seconds."
    )

    # Clean up temporary files for this video
    # ... (Cleanup logic remains the same) ...
    try:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} for {video_name} has been removed")
    except OSError as e:
        print(f"Error removing temporary directory {temp_dir}: {e}")

    print(f"--- Finished processing video: {video_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video(s) and generate COCO format pose dataset, dynamically finding ground truth."  # Updated description
    )
    # --- Input Arguments (Mutually Exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", type=str, help="Path to a single input video file"
    )
    input_group.add_argument(
        "--input-dir", type=str, help="Path to a directory containing input video files"
    )
    # --- Other Arguments ---
    parser.add_argument(
        "--pose-estimator-config",
        # ... (existing default and help) ...
        type=str,
        default="/home/hemingdu/Code/mmpose-limbdeficientpose/configs/body_2d_keypoint/topdown_heatmap/coco/ViTPose-base_8xb64-210e_para-256x192.py",
        help="Path to pose estimator config",
    )
    parser.add_argument(
        "--pose-estimator-checkpoint",
        # ... (existing default and help) ...
        type=str,
        default="/home/hemingdu/Code/mmpose-limbdeficientpose/work_dirs/best_coco_AP_epoch_250.pth",
        help="Path to pose estimator checkpoint",
    )
    # REMOVED: --detection-ground-truth argument
    # parser.add_argument(
    #     "--detection-ground-truth",
    #     # ...
    # )
    parser.add_argument(
        "--output",
        # ... (existing default and help) ...
        type=str,
        default="./work_dirs",
        help="Base output directory for COCO datasets",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default=None,
        help="Optional prefix for video filenames. Only process videos starting with this prefix.",
    )
    args = parser.parse_args()

    # --- Determine list of videos to process ---
    # ... (Video finding logic remains the same) ...
    videos_to_process = []
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            exit(1)
        print(f"Searching for videos in directory: {args.input_dir}")
        supported_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        for ext in supported_extensions:
            videos_to_process.extend(glob.glob(os.path.join(args.input_dir, ext)))
        if not videos_to_process:
            print(
                f"Error: No supported video files ({', '.join(supported_extensions)}) found in {args.input_dir}"
            )
            exit(1)
        print(f"Found {len(videos_to_process)} video file(s) initially.")

        # Filter by prefix if provided
        if args.video_prefix:
            videos_to_process = [
                v
                for v in videos_to_process
                if os.path.basename(v).startswith(args.video_prefix)
            ]
            if not videos_to_process:
                print(
                    f"Error: No videos found matching prefix '{args.video_prefix}' in {args.input_dir}"
                )
                exit(1)
            print(
                f"Filtered down to {len(videos_to_process)} video file(s) matching prefix '{args.video_prefix}'."
            )
        else:
            print(f"Processing all {len(videos_to_process)} found video file(s).")

        videos_to_process.sort()
    elif args.video:
        if not os.path.isfile(args.video):
            print(f"Error: Video file not found: {args.video}")
            exit(1)
        videos_to_process.append(args.video)

    # --- Initialize models (once) ---
    # Initialize detector - it might be needed if GT is not found for some videos
    detector = None
    if not has_mmdet:
        print(
            "MMDetection is not installed, required for detection if ground truth is not found."
        )
        # Decide if exit or continue: let's continue, maybe all videos have GT
        # exit(1)
    else:
        print(
            "Initializing detector (will be used if ground truth is not found for a video)..."
        )
        try:
            detector = init_detector(
                "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
                "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
                device="cuda:0",
            )
            detector.cfg = adapt_mmdet_pipeline(detector.cfg)
            print("Detector initialized.")
        except Exception as e:
            print(
                f"Warning: Error initializing detector: {e}. Detection will not be available."
            )
            detector = None  # Ensure detector is None if init fails

    # build pose estimator
    print("Initializing pose estimator...")
    # ... (Pose estimator initialization remains the same) ...
    try:
        pose_estimator = init_pose_estimator(
            args.pose_estimator_config,
            args.pose_estimator_checkpoint,
            device="cuda:0",
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
        )
        print("Pose estimator initialized.")
    except Exception as e:
        print(f"Error initializing pose estimator: {e}")
        exit(1)

    # --- REMOVED Global detection ground truth loading ---

    # --- Process each video ---
    overall_start_time = time.time()
    for video_file in videos_to_process:
        process_single_video(
            video_path=video_file,
            args=args,
            detector=detector,
            pose_estimator=pose_estimator,
        )

    overall_end_time = time.time()
    print(
        f"\nTotal processing time for {len(videos_to_process)} video(s): {overall_end_time - overall_start_time:.2f} seconds."
    )
    print("All processing complete.")
