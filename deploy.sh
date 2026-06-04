#!/bin/bash
# ======================================================
# TriClassSentiment 一键部署脚本
# 支持 Alibaba Cloud Linux 3 / RHEL 8+ 自动安装 Python 3.12
# 用法: chmod +x deploy.sh && sudo ./deploy.sh
# ======================================================

set -e  # 遇到错误立即退出

# ------------------ 配置变量 ------------------
APP_DIR="/opt/TriClassSentiment"
APP_USER="trisentiment"
PYTHON_VERSION="3.12.4"
PYTHON_BIN="/usr/local/bin/python${PYTHON_VERSION%.*}"   # 例如 /usr/local/bin/python3.12
PIP_BIN="/usr/local/bin/pip${PYTHON_VERSION%.*}"

# ------------------ 1. 安装系统编译依赖 ------------------
echo "=== 1. 安装系统编译依赖及基础工具 ==="
yum install -y gcc gcc-c++ make git nginx curl \
    openssl-devel bzip2-devel libffi-devel \
    zlib-devel readline-devel sqlite-devel \
    wget

# ------------------ 2. 编译安装 Python 3.12（如果不存在）------------------
if command -v python3.12 &> /dev/null; then
    echo "=== Python 3.12 已安装，跳过编译 ==="
else
    echo "=== 2. 编译安装 Python ${PYTHON_VERSION}（约需 10-20 分钟）==="
    cd /usr/local/src
    # 下载源码（若已存在则跳过下载）
    if [ ! -f "Python-${PYTHON_VERSION}.tgz" ]; then
        wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
    fi
    tar -xzf Python-${PYTHON_VERSION}.tgz
    cd Python-${PYTHON_VERSION}
    ./configure --enable-optimizations --prefix=/usr/local
    make -j $(nproc)
    make altinstall   # 使用 altinstall 避免覆盖系统 python3 命令
    echo "✓ Python ${PYTHON_VERSION} 安装完成"
fi

# 验证安装
if ! command -v python3.12 &> /dev/null; then
    echo "错误: Python 3.12 安装失败"
    exit 1
fi
echo "Python 3.12 路径: $(which python3.12)"

# ------------------ 3. 创建应用用户 ------------------
echo "=== 3. 创建应用用户（无登录权限）==="
id -u $APP_USER &>/dev/null || useradd -r -s /sbin/nologin $APP_USER

# ------------------ 4. 部署应用代码 ------------------
echo "=== 4. 准备应用目录 ==="
mkdir -p $APP_DIR
# 如果当前目录下有 Web 文件夹，则复制过去；否则提示用户
if [ -d "./Web" ]; then
    cp -r ./Web $APP_DIR/
    echo "已复制当前目录下的 Web 文件夹到 $APP_DIR"
else
    echo "警告: 当前目录未找到 Web 文件夹，请确保已将代码放到 $APP_DIR/Web"
    echo "你可以手动 scp 或 git clone 代码到 $APP_DIR"
    # 不退出，后续创建虚拟环境时会失败，但脚本继续
fi

# ------------------ 5. 创建虚拟环境并安装依赖 ------------------
echo "=== 5. 创建 Python 虚拟环境 ==="
cd $APP_DIR
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== 安装 PyTorch (CPU 版) ==="
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "=== 安装其他 Python 依赖 ==="
pip install transformers scikit-learn pandas numpy tqdm tabulate
pip install flask jieba gunicorn

# ------------------ 6. 创建 systemd 服务 ------------------
echo "=== 6. 配置 systemd 服务 ==="
cat > /etc/systemd/system/trisentiment.service << 'EOF'
[Unit]
Description=TriClassSentiment Web Service
After=network.target

[Service]
Type=simple
User=trisentiment
WorkingDirectory=/opt/TriClassSentiment/Web
Environment="PATH=/opt/TriClassSentiment/.venv/bin"
ExecStart=/opt/TriClassSentiment/.venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# ------------------ 7. 配置 Nginx ------------------
echo "=== 7. 配置 Nginx 反向代理 ==="
cat > /etc/nginx/conf.d/trisentiment.conf << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /static {
        alias /opt/TriClassSentiment/Web/static;
        expires 30d;
    }
}
EOF

# ------------------ 8. 设置文件权限 ------------------
echo "=== 8. 设置目录权限 ==="
chown -R $APP_USER:$APP_USER $APP_DIR

# ------------------ 9. 开放防火墙（如有）------------------
echo "=== 9. 配置防火墙 ==="
if command -v firewall-cmd &> /dev/null; then
    firewall-cmd --permanent --add-service=http 2>/dev/null || true
    firewall-cmd --reload 2>/dev/null || true
    echo "防火墙已放行 HTTP 服务"
else
    echo "未检测到 firewalld，跳过防火墙配置"
fi

# ------------------ 10. 启动服务 ------------------
echo "=== 10. 启动服务 ==="
systemctl daemon-reload
systemctl enable --now trisentiment
systemctl enable --now nginx
systemctl restart nginx

# ------------------ 完成提示 ------------------
echo ""
echo "========================================="
echo "✅ 部署完成！"
echo "========================================="
echo "访问地址: http://$(curl -s ifconfig.me 2>/dev/null || echo '你的服务器IP')"
echo "检查服务状态: systemctl status trisentiment"
echo "查看日志: journalctl -u trisentiment -f"
echo "注意: 如果 Web 代码尚未放入 $APP_DIR/Web，请先上传代码，然后重启服务:"
echo "      systemctl restart trisentiment"
echo "========================================="