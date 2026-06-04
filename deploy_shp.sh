# 1. 修正 systemd 服务文件
cat > /etc/systemd/system/trisentiment.service << 'EOF'
[Unit]
Description=TriClassSentiment Web Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/TriClassSentiment
Environment="PATH=/root/TriClassSentiment/.venv/bin"
ExecStart=/root/TriClassSentiment/.venv/bin/gunicorn -w 4 -b 127.0.0.1:8000 web.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 2. 重新加载并启动服务
systemctl daemon-reload
systemctl restart trisentiment

# 3. 查看状态（应该显示 active (running)）
systemctl status trisentiment