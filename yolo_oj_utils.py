import numpy as np
import cv2

def filter_outputs(layer_output, confidence):
    """ Pick the most probable class in each box and then filter it by confidence.
        layer_output : Output from a YOLO output layer
        confidence : confidence threshold (float)
    """
    box_xywh = np.array(layer_output[:, :4])
    box_confidence = np.array(layer_output[:, 4]).reshape(layer_output.shape[0], 1)
    box_class_probs = np.array(layer_output[:, 5:])

    box_scores = box_confidence * box_class_probs
    box_class = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)

    # pick up boxes with box_class_scores that are higher than confidence
    filtering_mask = box_class_scores >= confidence
    class_filtered = box_class[filtering_mask]
    score_filtered = box_class_scores[filtering_mask]
    xywh_filtered = box_xywh[np.nonzero(filtering_mask)]

    return (xywh_filtered, score_filtered, class_filtered)

def iou(box1, box2):
    """ Caculate IoU between box1 and box2
        box1/box2 : (x1, y1, x2, y2), where x1 and y1 are coordinates of upper left corner,
                    x2 and y2 are of lower right corner
        return: IoU
    """

    # get the area of intersection
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # get the area of union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    # get iou
    iou = inter_area / union_area

    return iou


def yolo_non_max_supression(boxes, scores, confidence_threshold, iou_threshold):
    """ Apply Non-max supression.
        boxes : Array of coordinates of boxes (x1, y1, x2, y2)
        scores : Array of confidence scores with respect to boxes
        score_threshold : Threshold of the score to keep
        iou_threshold : Threshold of IoU to keep

        Return : Indices of boxes and scores to be kept
    """

    # Sort scores in descending order
    sorted_idx = np.argsort(scores)[::-1]

    remove = []
    for i in np.arange(len(scores)):
        # if the score is already removed, skip it
        if i in remove:
            continue
        # if the score is blow the confidence, add it to the remove list
        if scores[sorted_idx[i]] < confidence_threshold:
            remove.append(i)
            continue

        for j in np.arange(i+1, len(scores)): # start the search from the next score
            if j in remove:
                continue
            if scores[sorted_idx[j]] < confidence_threshold:
                remove.append(j)
                continue

            # calculate IoU of two boxes.
            # If IoU is more than the threshold, add the box with the lower score to the remove list
            overlap = iou(boxes[sorted_idx[i]], boxes[sorted_idx[j]])
            if overlap > iou_threshold:
                remove.append(j)

    sorted_idx = np.delete(sorted_idx, remove)
    return sorted(sorted_idx)

def rescale_box_coord(boxes, width, height):
    """ Rescale bounding boxes to fit the original image, and calculate the coordinates
        of the top left corner and the bottom right corner.
        boxes : Array of (x,y,w,h) of the box
        width : Width of the original image
        height : Height of the original image
    """
    boxes_orig = boxes * np.array([width, height, width, height])
    boxes_orig[:, 0] -= boxes_orig[:, 2] / 2
    boxes_orig[:, 1] -= boxes_orig[:, 3] / 2

    # make an array of box coordinates.
    # boxes_coord = array of [[x1, y1, x2, y2], ...]: where (x1, y1) = upper left, (x2, y2) = lower right
    boxes_coord = boxes_orig
    # set x2 = x1 + w
    boxes_coord[:, 2] = boxes_orig[:, 0] + boxes_orig[:, 2]
    # set y2 = y1 + h
    boxes_coord[:, 3] = boxes_orig[:, 1] + boxes_orig[:, 3]

    return boxes_coord

def draw_boxes(image, boxes_coord, nms_idx, scores, classes, labels, colors):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    border_thickness = 2
    text_thickness = 1

    text_all = []
    for i in nms_idx:
        color = tuple([int(c) for c in colors[classes[i]]])
        text = "{}: {:.4f}".format(labels[classes[i]], scores[i])
        text_all.append(text)

        (pt1_x, pt1_y) = (int(boxes_coord[i, 0]), int(boxes_coord[i, 1]))
        (pt2_x, pt2_y) = (int(boxes_coord[i, 2]), int(boxes_coord[i, 3]))
        cv2.rectangle(image, (pt1_x, pt1_y), (pt2_x, pt2_y), color, border_thickness)

        (t_w, t_h), _ = cv2.getTextSize(text, font, fontScale=font_scale, thickness=text_thickness)
        text_offset_x = 7
        text_offset_y = 7
        (text_box_x1, text_box_y1) = (pt1_x, pt1_y - (t_h + text_offset_y))
        (test_box_x2, text_box_y2) = ((pt1_x + t_w + text_offset_x), pt1_y)

        cv2.rectangle(image, (text_box_x1, text_box_y1), (test_box_x2, text_box_y2), color, cv2.FILLED)

        cv2.putText(image, text, (pt1_x + text_offset_x, pt1_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (255, 255, 255), text_thickness)

    return(image, text_all)


def yolo_object_detection(image_filename, net, confidence, threshold, labels, colors):
    """ Apply YOLO object detection on a image_file.
        image_filename : Input image file to read
        net : YOLO v3 network object
        confidence : Confidence threshold (specified in command line)
        threshold : IoU threshold for NMS (specified in command line)
        labels : Class labels specified in coco.names
        colors : Colors assigned to the classes
    """

    # read image file
    # image is an array of image data (row, column, channel)
    image = cv2.imread(image_filename)

    (H, W) = image.shape[:2]

    # preprocess image data with rescaling and resizing to fit YOLO input shape
    # OpenCV assumes BGR images: we have to convert to RGB, with swapRB=True
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # set a new input to the network
    net.setInput(blob)

    # get YOLOv3's output layer names
    ln = net.getLayerNames()
    ln_out = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # perform object detection
    layerOutputs = net.forward(ln_out)


    # Get the result from outputs, and filter them by confidence
    boxes = []
    scores = []
    classes = []
    for output in layerOutputs: # There are three output layers in YOLO v3
        # Filter outputs by confidence
        (xywh_filterd, score_filtered, class_filtered) = filter_outputs(output, confidence)

        boxes.append(xywh_filterd)
        scores.append(score_filtered)
        classes.append(class_filtered)

    # Change shapes of arrays so that all boxes from any output layers are stored together
    boxes = np.vstack([r for r in boxes])
    scores = np.concatenate([r for r in scores], axis=None)
    classes = np.concatenate([r for r in classes], axis=None)

    # Apply Non-max supression
    boxes_coord = rescale_box_coord(boxes, W, H)
    nms_idx = yolo_non_max_supression(boxes_coord, scores, confidence, threshold)

    # Draw boxes on the image
    image, text_list = draw_boxes(image, boxes_coord, nms_idx, scores, classes, labels, colors)
    print("{} : {}".format(image_filename, text_list), flush=True)
    cv2.imshow("{}".format(image_filename, image), image)
    cv2.waitKey(0)
