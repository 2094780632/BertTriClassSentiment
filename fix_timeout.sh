#!/bin/bash
# ============================================
# TriClassSentiment 超时修复脚本
# 解决问题：批量预测返回 HTML 而非 JSON（Worker 超时）
# 用法: chmod +x fix_timeout.sh && sudo ./fix_timeout.sh
# ============================================
set -e

echo "=== 修复 Gunicorn 超时配置 ==="

cat > /etc/systemd/system/trisentiment.service << 'EOF'
[Unit]
Description=TriClassSentiment Web Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/TriClassSentiment
Environment="PATH=/root/TriClassSentiment/.venv/bin"
ExecStart=/root/TriClassSentiment/.venv/bin/gunicorn \
    -w 2 \
    -b 127.0.0.1:8000 \
    --timeout 300 \
    --graceful-timeout 30 \
    --worker-class sync \
    --max-requests 100 \
    --max-requests-jitter 20 \
    web.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "=== 修复 Nginx 代理超时 ==="

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
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 60s;
    }

    location /static {
        alias /root/TriClassSentiment/Web/static;
        expires 30d;
    }
}
EOF

echo "=== 重启服务 ==="
systemctl daemon-reload
systemctl restart trisentiment
systemctl restart nginx

echo ""
echo "=== 修复完成！==="
echo "   现在重试批量预测应该正常了"
echo "   查看日志: journalctl -u trisentiment -f"
echo "   检查状态: systemctl status trisentiment"
