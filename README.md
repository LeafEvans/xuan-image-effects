# 轩影幻境 (XuanImageEffects)

## 简介

**轩影幻境**是一个图像处理工具，支持对图像数据集应用多种效果，生成对比图，并便捷地管理处理流程。

## 目录结构

```
data/
├── raw/
│   ├── images/               # 原始图像
│   └── labels/               # 原始标签
└── processed/
    ├── Gaussian/             # 添加高斯噪声效果后的数据
    │   ├── images/           # 高斯噪声图像
    │   └── labels/           # 对应的标签
    ├── Shot/                 # 添加散粒噪声效果后的数据
    │   ├── images/           # 散粒噪声图像
    │   └── labels/           # 对应的标签
    ├── Impulse/              # 添加脉冲噪声效果后的数据
    │   ├── images/
    │   └── labels/
    └── ...                   # 其他效果的处理结果
effects/
    └── effects.py             # 图像效果的具体实现
scripts/
    └── process_dataset.py     # 数据处理脚本
visualizations/                # 存储可视化效果对比结果
requirements.txt               # 依赖文件
LICENSE                        # 许可证
README.md                      # 项目说明
setup.py                       # 安装脚本
```

## 系统要求

- **Python**: 需要 Python 3.6 或更高版本。
- **操作系统**: 支持 Windows、macOS 和 Linux。

## 安装

1. 克隆项目代码：

   ```bash
   git clone https://github.com/LeafEvans/XuanImageEffects.git
   cd XuanImageEffects
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

1. **准备数据**：将原始图像和标签文件放入 `data/raw` 目录：
   - 图像文件放在 `data/raw/images/`。
   - 标签文件放在 `data/raw/labels/`。

2. **运行数据处理脚本**：

   ```bash
   python scripts/process_dataset.py
   ```

3. **查看处理结果**：
   - 每种效果的处理结果会保存在 `data/processed/{效果名称}/` 目录下。
   - 可视化效果对比图存储在 `visualizations/` 目录。

## 数据集格式要求

- **图像**: 支持 `.jpg`、`.png`、`.jpeg` 格式。
- **标签**: 支持 `.txt`、`.xml`、`.json` 格式。

## 支持的图像效果

- 高斯噪声
- 散粒噪声
- 脉冲噪声
- 散焦模糊
- 磨砂玻璃模糊
- 运动模糊
- 变焦模糊
- 雪效果
- 霜冻效果
- 雾效果
- 亮度调整
- 对比度调整
- 弹性变形
- 像素化
- JPEG压缩

## 贡献

特别感谢 Zhou Zixuan (.zZ) 提供的算法支持。

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源，您可以自由使用和分发。

---

如果还有其他需求，请随时告知！
