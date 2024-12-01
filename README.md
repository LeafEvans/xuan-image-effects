# 轩影幻境 (XuanImageEffects)

## 简介
轩影幻境是一个用于处理图像数据集的项目，应用多种图像效果并生成对比图。

## 目录结构
```
data/
├── raw/
│   ├── images/
│   └── labels/
└── processed/
    ├── images/
    └── labels/
effects/
    └── effects.py
scripts/
    └── process_dataset.py
visualizations/
requirements.txt
LICENSE
README.md
setup.py
```

## 系统要求
- **Python**: 需要 Python 3.6 或更高版本。
- **操作系统**: 支持 Windows, macOS, 和 Linux。

## 安装
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
1. **准备数据**：将原始图像和标签放入 `data/raw` 目录。
   - 图像文件放在 `data/raw/images/`。
   - 标签文件放在 `data/raw/labels/`。
2. 运行数据处理脚本：
   ```bash
   python scripts/process_dataset.py
   ```
3. 处理后的图像和标签将保存在 `data/processed` 目录。 
4. 可视化对比图保存在 `visualizations` 目录。

## 数据集格式要求
- **图像**: 支持 `.jpg`, `.png`, `.jpeg` 格式。
- **标签**: 支持 `.txt`, `.xml`, `.json` 格式。

## 效果列表
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
感谢 Zhou Zixuan (.zZ) 提供的算法支持。

## 许可证
本项目采用 [Apache License 2.0](LICENSE)。
