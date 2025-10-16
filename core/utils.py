import json

import torch
import cv2
import io
import numpy as np

from typing import *
from PIL import Image
from base64 import b64decode
from torch.utils.data import default_collate
import torchvision.transforms.functional as TF


def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    """
    Partition a list by size.
    When indivisible, the last group contains fewer items than the target size.

    Examples:
        - data: [1,2,3,4,5]
        - size: 2
        - return: [[1,2], [3,4], [5]]
    """
    return [data[i:i + size] for i in range(0, len(data), size)]


def partition_by_groups(data: List[Any], groups: int) -> List[List[Any]]:
    """
    Partition a list by groups.
    When indivisible, some groups may have more items than others.

    Examples:
        - data: [1,2,3,4,5]
        - groups: 2
        - return: [[1,3,5], [2,4]]
    """
    return [data[i::groups] for i in range(groups)]


def shift_list(data: List[Any], n: int):
    """
    Rotate a list by n elements.

    Examples:
        - data: [1,2,3,4,5]
        - n: 3
        - return: [4,5,1,2,3]
    """
    return data[n:] + data[:n]


def custom_collate(batch: List[dict]):
    base_size = batch[0]['image'].shape[1:]
    for sample in batch[1:]:
        size = sample['image'].shape[1:]
        if base_size != size:
            print(f"Found diff resolution in one batch size, index file: {sample['index_file']}, key: {sample['key']}.")
            sample['image'] = TF.resize(sample['image'], size=base_size, interpolation=TF.InterpolationMode.BILINEAR)
    return default_collate(batch)


def normalized_tensor_to_uint8_array(tensor: torch.Tensor):
    if tensor.ndim == 4:
        return (((tensor.permute(0, 2, 3, 1) + 1) / 2.) * 255).to('cpu').numpy().astype('uint8').copy()
    elif tensor.ndim == 3:
        return (((tensor.permute(1, 2, 0) + 1) / 2.) * 255).to('cpu').numpy().astype('uint8').copy()
    else:
        raise NotImplementedError


def im2bytes(im: np.ndarray, dtype='png'):
    im_pil = Image.fromarray(im)
    io_bytes = io.BytesIO()
    im_pil.save(io_bytes, dtype)
    img_bytes = io_bytes.getvalue()
    return img_bytes


def calculate_fit_size(height, width, output_size: int):
    long_side = max(height, width)
    short_side = min(height, width)
    if height > width:
        fit_size = (output_size, int(output_size / long_side * short_side))
    else:
        fit_size = (int(output_size / long_side * short_side), output_size)
    return fit_size


def fit_image(image, output_size):
    fit_size = calculate_fit_size(image.shape[0], image.shape[1], output_size)
    image = Image.fromarray(image)
    fitted = image.resize((fit_size[1], fit_size[0]), Image.Resampling.BILINEAR)
    fitted = np.array(fitted)
    return fitted, fit_size


def paste_to_canvas(image: np.ndarray, canvas_size: int):
    assert (image.shape[2] == 3)
    long_side = max(image.shape[0], image.shape[1])
    assert long_side <= canvas_size
    offset_h = (canvas_size - image.shape[0]) // 2
    offset_w = (canvas_size - image.shape[1]) // 2
    canvas = np.zeros([canvas_size, canvas_size, 3], dtype=image.dtype)
    mask = np.zeros([canvas_size, canvas_size], dtype=image.dtype)
    canvas[offset_h: (offset_h + image.shape[0]), offset_w: (offset_w + image.shape[1]), :] = image
    mask[offset_h: (offset_h + image.shape[0]), offset_w: (offset_w + image.shape[1])] = 1
    return canvas, mask


def decode_image(image_data):
    if isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        image_bytes = b64decode(image_data)

    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
            image = image.convert("RGBA")
            white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
            white.paste(image, mask=image.split()[3])
            image = white
        else:
            image = image.convert("RGB")
    return image


def expand_batch_dim(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x, np.ndarray):
        return x.reshape([1, *x.shape])
    elif isinstance(x, torch.Tensor):
        return x.unsqueeze(0)
    elif isinstance(x, (np.float16, np.float32, np.float64, np.int64)):
        return x.reshape(1)
    elif isinstance(x, (int, float)):
        return np.array([x])
    else:
        raise NotImplementedError


def pack_data(data_list: List[Union[np.ndarray, torch.Tensor]]) -> Union[np.ndarray, torch.Tensor]:
    data_list = [expand_batch_dim(d) for d in data_list]
    if isinstance(data_list[0], np.ndarray):
        return np.concatenate(data_list, axis=0)
    elif isinstance(data_list[0], torch.Tensor):
        return torch.cat(data_list, dim=0)
    else:
        raise NotImplementedError


def unzip_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    def check(lst):
        return all(i == lst[0] for i in lst)
    keys = batch.keys()
    assert check([len(v) for v in batch.values()])
    values = [[d[0] for d in v.split(1, 0)] if isinstance(v, torch.Tensor) else v for v in batch.values()]
    values = [l for l in zip(*values)]
    unzipped = [dict((k, v) for k, v in zip(keys, val)) for val in values]
    return unzipped


def zip_batches(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(data_list) == 0:
        return {}
    keys = data_list[0].keys()
    batch_dict = {}
    for k in keys:
        item_list = [d[k] for d in data_list]
        if isinstance(item_list[0], str):
            batch_dict[k] = item_list
        else:
            batch_dict[k] = pack_data(item_list)
    return batch_dict

def dumps(data_dict: Dict[str, Any]) -> str:
    for k in data_dict.keys():
        if isinstance(data_dict[k], np.ndarray):
            data_dict[k] = data_dict[k].tolist()
        elif isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k].detach().cpu().numpy()
        elif isinstance(data_dict[k], str):
            data_dict[k] = data_dict[k]
        else:
            raise NotImplementedError
    return json.dumps(data_dict)
