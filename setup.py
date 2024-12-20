# Copyright 2024 LeafEvans, Sichuan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""项目安装配置模块.

本模块配置了项目的基本信息、依赖项和安装要求等。
"""

from setuptools import setup, find_packages

# 配置项目的基本信息和依赖项
setup(
    name="XuanImageEffects",  # 项目名称
    version="0.1.0",  # 项目版本
    description="A project for applying various image effects to datasets.",  # 项目描述
    author=".zZ, LeafEvans",  # 作者
    author_email="2669846676@qq.com, leafevans@foxmail.com",  # 作者邮箱
    packages=find_packages(),  # 自动查找项目中的包
    install_requires=[
        "opencv-python",  # 图像处理库
        "numpy",  # 数值计算库
        "Pillow",  # 图像处理库
        "matplotlib",  # 可视化库
        "scipy",  # 科学计算库
    ],
    classifiers=[
        "Programming Language :: Python :: 3",  # 支持的编程语言
        "License :: OSI Approved :: Apache Software License",  # 许可证
        "Operating System :: OS Independent",  # 操作系统兼容性
    ],
    python_requires=">=3.6",  # 支持的 Python 版本
)
