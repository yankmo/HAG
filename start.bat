@echo off
echo 🌟 HAG 项目启动中...
echo ==================================================

echo 🚀 启动后端服务...
start "HAG Backend" cmd /k "python backend_api.py"

echo ⏳ 等待后端服务启动...
timeout /t 3 /nobreak > nul

echo 🎨 启动前端服务...
cd frontend
if not exist node_modules (
    echo 📦 安装前端依赖...
    npm install
)
npm start

pause