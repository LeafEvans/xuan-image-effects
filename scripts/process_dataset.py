import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import shutil
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from effects.effects import (
    add_gaussian_noise, add_shot_noise, add_impulse_noise, add_defocus_blur,
    add_frosted_glass_blur, add_motion_blur, add_zoom_blur, add_snow,
    add_frost, add_fog, adjust_brightness, adjust_contrast,
    add_elastic_transform, add_pixelation, add_jpeg_compression)


def process_image(image_path, output_path, effects):
    """
    处理单张图像，应用指定的效果并保存结果。

    参数:
    - image_path: 输入图像的文件路径
    - output_path: 输出图像的文件路径
    - effects: 要应用的效果函数列表

    功能:
    读取输入图像，依次应用效果列表中的每个效果函数，
    然后将处理后的图像保存到指定的输出路径。
    """
    image = cv2.imread(image_path)  # 读取图像
    for effect in effects:
        image = effect(image)  # 应用每个效果
    cv2.imwrite(output_path, image)  # 保存处理后的图像


def copy_labels(input_label_dir, output_label_dir):
    """
    复制标签文件到指定的输出目录。

    参数:
    - input_label_dir: 输入标签的目录路径
    - output_label_dir: 输出标签的目录路径

    功能:
    遍历输入标签目录中的所有标签文件，并将其复制到输出目录。
    """
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)  # 如果输出目录不存在，则创建

    for filename in os.listdir(input_label_dir):
        if filename.endswith(('.txt', '.xml', '.json')):  # 支持多种标签格式
            input_label_path = os.path.join(input_label_dir, filename)
            output_label_path = os.path.join(output_label_dir, filename)
            shutil.copy(input_label_path, output_label_path)  # 复制标签文件


def visualize_comparison(original_image, processed_images, effect_names, output_dir, dpi=300):
    """
    生成处理前后的对比图，带有增强的美观设置。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置全局样式
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False
    })

    # 单个效果对比
    for processed_image, effect_name in zip(processed_images, effect_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
        fig.patch.set_facecolor('white')
        
        # 设置边框样式
        rect_original = {'linewidth': 3, 'edgecolor': '#FFD700', 'facecolor': 'none', 'alpha': 0.8}
        rect_processed = {'linewidth': 3, 'edgecolor': '#4682B4', 'facecolor': 'none', 'alpha': 0.8}

        # 原始图像
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.add_patch(Rectangle((0, 0), 1, 1, transform=ax1.transAxes, **rect_original))
        ax1.set_title('Original', fontsize=22, pad=20)
        ax1.axis('off')

        # 处理后图像
        ax2.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes, **rect_processed))
        ax2.set_title(effect_name, fontsize=22, pad=20)
        ax2.axis('off')

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_dir, f'{effect_name}.png'), 
                   bbox_inches='tight', dpi=dpi, facecolor='white')
        plt.close(fig)

    # 总对比图
    num_cols = 4
    num_rows = (len(processed_images) + num_cols) // num_cols
    fig = plt.figure(figsize=(20, 5 * num_rows), dpi=dpi, facecolor='white')
    gs = fig.add_gridspec(num_rows, num_cols, hspace=0.3, wspace=0.2)
    
    # 原始图像
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                          edgecolor='#FFD700', linewidth=3, facecolor='none', alpha=0.8))
    ax.set_title('Original', fontsize=24, pad=20)
    ax.axis('off')

    # 处理后图像
    for i, (processed_image, effect_name) in enumerate(zip(processed_images, effect_names), start=1):
        row, col = divmod(i, num_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             edgecolor='#4682B4', linewidth=3, facecolor='none', alpha=0.8))
        ax.set_title(effect_name, fontsize=24, pad=20)
        ax.axis('off')
    
    plt.savefig(os.path.join(output_dir, 'comparison.png'), 
                bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close(fig)


def process_dataset(input_image_dir, input_label_dir, output_base_dir, effects_dict):
    """
    处理整个数据集，应用指定的效果并组织输出目录。

    参数:
    - input_image_dir: 输入图像的目录路径
    - input_label_dir: 输入标签的目录路径
    - output_base_dir: 输出的基础目录路径
    - effects_dict: 包含效果名称和对应效果函数列表的字典

    功能:
    为每种效果创建单独的输出目录，处理图像并复制标签。
    """
    image_files = [
        f for f in os.listdir(input_image_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ]
    random_image_file = random.choice(image_files)
    original_image_path = os.path.join(input_image_dir, random_image_file)
    original_image = cv2.imread(original_image_path)

    processed_images = []
    effect_names = []

    for effect_name, effect_funcs in effects_dict.items():
        # 为每种效果创建单独的输出目录
        output_image_dir = os.path.join(output_base_dir, effect_name, 'images')
        output_label_dir = os.path.join(output_base_dir, effect_name, 'labels')

        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)  # 创建图像输出目录

        for filename in image_files:
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # 处理图像文件
                input_image_path = os.path.join(input_image_dir, filename)
                output_image_path = os.path.join(output_image_dir, filename)
                process_image(input_image_path, output_image_path, effect_funcs)  # 处理图像

        # 复制标签文件到对应的效果目录
        copy_labels(input_label_dir, output_label_dir)

        # 处理随机选择的图像并保存处理后的图像
        processed_image = original_image.copy()
        for effect in effect_funcs:
            processed_image = effect(processed_image)
        processed_images.append(processed_image)
        effect_names.append(effect_name)

    # 生成对比图
    visualize_comparison(original_image, processed_images, effect_names, 'visualizations')


if __name__ == "__main__":
    # 定义输入和输出的目录路径
    input_image_directory = 'data/raw/images'
    input_label_directory = 'data/raw/labels'
    output_base_directory = 'data/processed'

    # 定义每种效果及其对应的函数
    effects_dict = {
        'Gaussian': [add_gaussian_noise],
        'Shot': [add_shot_noise],
        'Impulse': [add_impulse_noise],
        'Defocus': [add_defocus_blur],
        'Frosted Glass': [add_frosted_glass_blur],
        'Motion': [add_motion_blur],
        'Zoom': [add_zoom_blur],
        'Snow': [add_snow],
        'Frost': [add_frost],
        'Fog': [add_fog],
        'Brightness': [adjust_brightness],
        'Contrast': [adjust_contrast],
        'Elastic': [add_elastic_transform],
        'Pixelation': [add_pixelation],
        'JPEG': [add_jpeg_compression]
    }

    # 处理数据集
    process_dataset(input_image_directory, input_label_directory, output_base_directory, effects_dict)
