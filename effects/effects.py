import cv2
import numpy as np
from PIL import Image, ImageFilter
import scipy.ndimage as ndimage

# 常量定义
DEFAULT_MEAN = 0  # 高斯噪声的默认均值
DEFAULT_STD_DEV = 50  # 高斯噪声的默认标准差
DEFAULT_SCALE = 0.3  # 散粒噪声的默认比例因子
DEFAULT_PROBABILITY = 0.05  # 脉冲噪声的默认概率
DEFAULT_KERNEL_SIZE = 25  # 模糊核的默认大小
DEFAULT_BLUR_RADIUS = 3  # 磨砂玻璃模糊的默认半径
DEFAULT_ZOOM_FACTOR = 1.5  # 变焦模糊的默认因子
DEFAULT_DENSITY = 0.0005  # 雪效果的默认密度
DEFAULT_BRIGHTNESS = 1.5  # 雪效果的默认亮度
DEFAULT_LAYERS = 3  # 雪效果的默认层数
DEFAULT_ANGLE = 45  # 雪效果的默认角度
DEFAULT_INTENSITY = 0.5  # 霜冻和雾效果的默认强度
DEFAULT_BRIGHTNESS_FACTOR = 2.0  # 亮度调整的默认因子
DEFAULT_CONTRAST_FACTOR = 2.0  # 对比度调整的默认因子
DEFAULT_ALPHA = 200  # 弹性变形的默认强度
DEFAULT_SIGMA = 20  # 弹性变形的高斯滤波默认标准差
DEFAULT_PIXEL_SIZE = 20  # 像素化效果的默认大小
DEFAULT_JPEG_QUALITY = 10  # JPEG压缩的默认质量


def add_gaussian_noise(image, mean=DEFAULT_MEAN, std_dev=DEFAULT_STD_DEV):
    """
    添加高斯噪声。

    参数:
    image (numpy.ndarray): 输入图像。
    mean (float): 噪声均值。
    std_dev (float): 噪声标准差。

    返回:
    numpy.ndarray: 添加噪声后的图像。
    """
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_shot_noise(image, scale=DEFAULT_SCALE):
    """
    添加散粒噪声。

    参数:
    image (numpy.ndarray): 输入图像。
    scale (float): 噪声比例因子。

    返回:
    numpy.ndarray: 添加噪声后的图像。
    """
    noise = np.random.poisson(image * scale).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_impulse_noise(image, probability=DEFAULT_PROBABILITY):
    """
    添加脉冲噪声（盐和胡椒噪声）。

    参数:
    image (numpy.ndarray): 输入图像。
    probability (float): 噪声概率。

    返回:
    numpy.ndarray: 添加噪声后的图像。
    """
    noisy_image = image.copy()
    num_salt = int(probability * image.size * 0.5)
    num_pepper = int(probability * image.size * 0.5)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255
    pepper_coords = [
        np.random.randint(0, i - 1, num_pepper) for i in image.shape
    ]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    return noisy_image


def add_defocus_blur(image, kernel_size=DEFAULT_KERNEL_SIZE):
    """
    添加散焦模糊。

    参数:
    image (numpy.ndarray): 输入图像。
    kernel_size (int): 模糊核大小。

    返回:
    numpy.ndarray: 添加模糊后的图像。
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_frosted_glass_blur(image, blur_radius=DEFAULT_BLUR_RADIUS):
    """
    添加磨砂玻璃模糊。

    参数:
    image (numpy.ndarray): 输入图像。
    blur_radius (int): 模糊半径。

    返回:
    numpy.ndarray: 添加模糊后的图像。
    """
    pil_image = Image.fromarray(image)
    return np.array(pil_image.filter(ImageFilter.GaussianBlur(blur_radius)))


def add_motion_blur(image, kernel_size=DEFAULT_KERNEL_SIZE):
    """
    添加运动模糊。

    参数:
    image (numpy.ndarray): 输入图像。
    kernel_size (int): 模糊核大小。

    返回:
    numpy.ndarray: 添加模糊后的图像。
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)


def add_zoom_blur(image, zoom_factor=DEFAULT_ZOOM_FACTOR):
    """
    添加变焦模糊。

    参数:
    image (numpy.ndarray): 输入图像。
    zoom_factor (float): 变焦因子。

    返回:
    numpy.ndarray: 添加模糊后的图像。
    """
    height, width, _ = image.shape
    zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)
    zoomed_image = zoomed_image[(zoomed_image.shape[0] - height) //
                                2:(zoomed_image.shape[0] + height) // 2,
                                (zoomed_image.shape[1] - width) //
                                2:(zoomed_image.shape[1] + width) // 2]
    return cv2.GaussianBlur(zoomed_image, (15, 15), 0)


def add_snow(image,
             density=DEFAULT_DENSITY,
             brightness=DEFAULT_BRIGHTNESS,
             layers=DEFAULT_LAYERS,
             angle=DEFAULT_ANGLE):
    """
    添加雪效果。

    参数:
    image (numpy.ndarray): 输入图像。
    density (float): 雪密度。
    brightness (float): 雪亮度。
    layers (int): 雪层数。
    angle (float): 雪角度。

    返回:
    numpy.ndarray: 添加雪效果后的图像。
    """
    height, width, _ = image.shape
    snow_layer = np.zeros((height, width, 3), dtype=np.float32)

    for layer in range(layers):
        layer_density = density * (0.5 + layer * 0.3)
        num_snowflakes = int(layer_density * height * width)
        layer_snow = np.zeros((height, width, 3), dtype=np.float32)

        for _ in range(num_snowflakes):
            is_cluster = np.random.rand() > 0.6

            if is_cluster:
                cluster_size = np.random.randint(20, 50)
                mask = np.zeros((cluster_size, cluster_size), dtype=np.uint8)
                num_points = np.random.randint(6, 10)
                points = np.array([[
                    np.random.randint(0, cluster_size),
                    np.random.randint(0, cluster_size)
                ] for _ in range(num_points)])
                points = points.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [points], (255, 255, 255))
                mask = cv2.GaussianBlur(
                    mask,
                    (cluster_size // 2 * 2 + 1, cluster_size // 2 * 2 + 1), 0)
                alpha = np.random.uniform(0.3, 0.7)
                cluster = np.stack([mask] * 3, axis=-1).astype(
                    np.float32) * alpha
                x = np.random.randint(0, width - cluster_size)
                y = np.random.randint(0, height - cluster_size)
                layer_snow[y:y + cluster_size, x:x + cluster_size] += cluster
            else:
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                length = np.random.randint(20, 60 + layer * 20)
                snow_angle = angle + np.random.uniform(-10, 10)
                thickness = np.random.randint(2, 8)
                alpha = np.random.uniform(0.5, 0.8)
                start_thickness = max(1, thickness - np.random.randint(1, 3))
                end_thickness = thickness
                thickness_variation = np.linspace(start_thickness,
                                                  end_thickness,
                                                  num=length)
                x2 = int(x + length * np.cos(np.radians(snow_angle)))
                y2 = int(y + length * np.sin(np.radians(snow_angle)))

                for i in range(length - 1):
                    x_start = int(x + i * (x2 - x) / length)
                    y_start = int(y + i * (y2 - y) / length)
                    x_end = int(x + (i + 1) * (x2 - x) / length)
                    y_end = int(y + (i + 1) * (y2 - y) / length)
                    cv2.line(layer_snow, (x_start, y_start), (x_end, y_end),
                             (255 * alpha, 255 * alpha, 255 * alpha),
                             int(thickness_variation[i]))

        blur_kernel_size = 5 + 2 * layer
        layer_snow = cv2.GaussianBlur(layer_snow,
                                      (blur_kernel_size, blur_kernel_size), 0)
        snow_layer += layer_snow.astype(np.float32) / float(layer + 1)

    snow_layer = np.clip(snow_layer, 0, 255).astype(np.uint8)
    brightened_image = cv2.convertScaleAbs(image, alpha=1.1, beta=30)
    snow_image = cv2.addWeighted(brightened_image, 1, snow_layer, brightness,
                                 0)

    return snow_image


def add_frost(image, intensity=DEFAULT_INTENSITY):
    """
    添加霜冻效果。

    参数:
    image (numpy.ndarray): 输入图像。
    intensity (float): 霜冻强度。

    返回:
    numpy.ndarray: 添加霜冻效果后的图像。
    """
    frost_layer = np.random.normal(200, 50, image.shape).astype(np.float32)
    frosty_image = cv2.addWeighted(image.astype(np.float32), 1 - intensity,
                                   frost_layer, intensity, 0)
    return np.clip(frosty_image, 0, 255).astype(np.uint8)


def add_fog(image, intensity=DEFAULT_INTENSITY):
    """
    添加雾效果。

    参数:
    image (numpy.ndarray): 输入图像。
    intensity (float): 雾强度。

    返回:
    numpy.ndarray: 添加雾效果后的图像。
    """
    height, width, _ = image.shape
    fog_layer = np.full((height, width, 3), 200, dtype=np.uint8)
    foggy_image = cv2.addWeighted(image, 1 - intensity, fog_layer, intensity,
                                  0)
    return foggy_image


def adjust_brightness(image, factor=DEFAULT_BRIGHTNESS_FACTOR):
    """
    调整亮度。

    参数:
    image (numpy.ndarray): 输入图像。
    factor (float): 亮度因子。

    返回:
    numpy.ndarray: 调整亮度后的图像。
    """
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image, factor=DEFAULT_CONTRAST_FACTOR):
    """
    调整对比度。

    参数:
    image (numpy.ndarray): 输入图像。
    factor (float): 对比度因子。

    返回:
    numpy.ndarray: 调整对比度后的图像。
    """
    mean_value = np.mean(image)
    return np.clip((image - mean_value) * factor + mean_value, 0,
                   255).astype(np.uint8)


def add_elastic_transform(image, alpha=DEFAULT_ALPHA, sigma=DEFAULT_SIGMA):
    """
    添加弹性变形。

    参数:
    image (numpy.ndarray): 输入图像。
    alpha (float): 变形强度。
    sigma (float): 高斯滤波标准差。

    返回:
    numpy.ndarray: 添加变形后的图像。
    """
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    random_array = random_state.rand(*shape) * 2 - 1
    dx = ndimage.gaussian_filter(random_array, sigma, mode="constant",
                                 cval=0).astype(np.float32) * float(alpha)
    dy = ndimage.gaussian_filter(random_array, sigma, mode="constant",
                                 cval=0).astype(np.float32) * float(alpha)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, shape[0] - 1).astype(np.float32)
    distorted_image = cv2.remap(image,
                                map_x,
                                map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    return distorted_image


def add_pixelation(image, pixel_size=DEFAULT_PIXEL_SIZE):
    """
    添加像素化效果。

    参数:
    image (numpy.ndarray): 输入图像。
    pixel_size (int): 像素大小。

    返回:
    numpy.ndarray: 添加像素化效果后的图像。
    """
    height, width, _ = image.shape
    temp = cv2.resize(image, (width // pixel_size, height // pixel_size),
                      interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def add_jpeg_compression(image, quality=DEFAULT_JPEG_QUALITY):
    """
    添加JPEG压缩效果。

    参数:
    image (numpy.ndarray): 输入图像。
    quality (int): JPEG压缩质量。

    返回:
    numpy.ndarray: 添加压缩效果后的图像。
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg',
                                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                    encode_param)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
