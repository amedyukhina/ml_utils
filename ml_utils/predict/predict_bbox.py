import pandas as pd
import torch
import torchvision
from am_utils.utils import walk_dir
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from ..dataset.dataset_object_inference import DatasetObjectInference
from ..transforms.bbox import get_test_transform
from ..utils.utils import collate_fn
from ..utils.utils import remove_overlapping_boxes, get_boxes_above_threshold


def get_df_of_file_list(input_dir, id_name='image_id'):
    """
    List files in given folder and generate a dataframe for the data loader.

    Parameters
    ----------
    input_dir : str
        Input directory
    id_name : str, optional
        Column name to specify image ID.
        Default is 'image_id'

    Returns
    -------
    pd.DataFrame
        Dataframe with a list of input files.

    """
    files = walk_dir(input_dir)
    files = [fn[len(input_dir) + 1:] for fn in files]
    df = pd.DataFrame({id_name: files})
    return df


def load_detection_model(model_fn, num_classes=2, device=None):
    """
    Load the object detection model from a given file.

    Parameters
    ----------
    model_fn : str
        Model filename with the full path.
    num_classes : int, optional
        Number of classes in the object detection model.
        Default is 2 (one class + background).
    device : torch.device
        Device to send the model to ('cpu' or 'cuda').
        If None, the device will be detected automatically.
        Default is None.

    Returns
    -------
    model:
        Torch model with loaded weights.

    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_fn))
    model.eval()

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model


def detect_bboxes(input_dir, model_fn, batch_size=2, maxsize=None,
                  detection_threshold=0.5, overlap_threshold=0.1, id_name='image_id'):
    """
    Detect object bounding boxes in all image in give directory and return dataframe with the results.

    Parameters
    ----------
    input_dir : str
        Input directory.
    model_fn :  str
        Model filename with the full path.
    batch_size : int, optional
        Batch size for predictions.
        Default is 2.
    maxsize : int, optional
        Pad the input image to a square with this size.
        Default is None.
    detection_threshold : float, optional
        Threshold (between 0 and 1) for the confidence of the bounding boxes.
        Bounding boxes with a confidence score lower than `detection_threshold` will not be included.
        Default is 0.5.
    overlap_threshold : float, optional
        Maximum allowed intersection-over-union (IOU) score for two bounding boxes.
        If two boxes overlap with a higher score, the box with a lower confidence score will be removed
    id_name : str, optional
        Column name to specify image ID.
        Default is 'image_id'

    Returns
    -------
    pd.DataFrame
        Dataframe with detected bounding box coordinates.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_detection_model(model_fn, device=device)
    df = get_df_of_file_list(input_dir)
    loader_kwargs = dict(batch_size=batch_size,
                         shuffle=False,
                         num_workers=batch_size,
                         drop_last=False)

    ds = DatasetObjectInference(df, input_dir,
                                get_test_transform(),
                                maxsize=maxsize)
    dl = DataLoader(ds, collate_fn=collate_fn, **loader_kwargs)
    results = pd.DataFrame()
    for images, image_ids in tqdm(dl):

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i in range(len(outputs)):
            bboxes, scores = get_boxes_above_threshold(outputs[i], detection_threshold)
            bboxes, scores = remove_overlapping_boxes(bboxes, scores,
                                                      overlap_threshold, return_full=True)
            bboxes = bboxes[scores > 0].data.cpu().numpy()
            scores = scores[scores > 0].data.cpu().numpy()

            image_id = image_ids[i]
            cur_results = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])
            cur_results['scores'] = scores
            cur_results[id_name] = image_id
            results = pd.concat([results, cur_results], ignore_index=True)
    return results
