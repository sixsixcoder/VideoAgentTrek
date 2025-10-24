import math


def round_by_factor(number: int, factor: int) -> int:
    """返回最接近 number 的且能被 factor 整除的整数"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """返回大于等于 number 的且能被 factor 整除的整数"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """返回小于等于 number 的且能被 factor 整除的整数"""
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280, max_long_side=8192):
    """缩放后图片满足以下条件:
    1. 长宽能被 factor 整除
    2. pixels 总数被限制在 [min_pixels, max_pixels] 内
    3. 最长边限制在 max_long_side 内
    4. 保证其长宽比基本不变
    """
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def update_image_size_(image_ele: dict, min_tokens=1, max_tokens=12800, merge_base=2, patch_size=14):
    """根据 min_tokens, max_tokens 更新 image_ele 的尺寸信息

    Args:
        image_ele (dict):
            - image_ele["image"]: str 图片路径
            - image_ele["height"]: int 图片原始高度
            - image_ele["width"]: int 图片原始宽度

    Returns:
        更新后的 image_ele, 新增如下 key-value pair
        dict:
            - image_ele["resized_height"]: int 输入到模型的真实高度
            - image_ele["resized_width"]: int 输入到模型的真实宽度
            - image_ele["seq_len"]: int 输入到模型所占的序列长度
    """
    height, width = image_ele["height"], image_ele["width"]
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=merge_base * patch_size,
        min_pixels=pixels_per_token * min_tokens,
        max_pixels=pixels_per_token * max_tokens,
        max_long_side=50000,
    )
    image_ele.update(
        {
            "resized_height": resized_height,
            "resized_width": resized_width,
            "seq_len": resized_height * resized_width // pixels_per_token + 2,
        }
    )
    return image_ele