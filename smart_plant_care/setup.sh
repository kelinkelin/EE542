#!/bin/bash
# 项目环境设置脚本

echo "=========================================="
echo "智能植物护理系统 - 环境设置"
echo "=========================================="
echo ""

# 检查Python版本
echo "检查Python版本..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ 错误: 未找到Python3"
    echo "请先安装Python 3.8或更高版本"
    exit 1
fi

# 创建虚拟环境
echo ""
echo "创建虚拟环境..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ 创建虚拟环境失败"
    exit 1
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo ""
echo "安装项目依赖..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 安装依赖失败"
    exit 1
fi

# 检查GPU
echo ""
echo "检查GPU可用性..."
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('未检测到GPU，将使用CPU')"

# 创建必要的目录
echo ""
echo "创建项目目录..."
mkdir -p data models logs docs/images videos

# 创建.gitkeep文件
touch data/.gitkeep models/.gitkeep logs/.gitkeep

echo ""
echo "=========================================="
echo "✅ 环境设置完成!"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 激活虚拟环境: source venv/bin/activate"
echo "  2. 运行基线对比: python run_baseline_comparison.py"
echo "  3. 训练PPO模型: python src/agents/train_ppo.py --timesteps 1000000"
echo ""
echo "查看README.md了解更多信息"
echo "=========================================="

