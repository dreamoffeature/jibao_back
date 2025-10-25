# build.sh
#!/usr/bin/env bash
# 退出当任何命令失败时
set -o errexit

# 安装依赖
pip install -r requirements.txt

# 收集静态文件
python manage.py collectstatic --noinput