#!/usr/bin/env python3
"""
HAG 项目启动脚本
同时启动前端和后端服务
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def run_backend():
    """启动后端服务"""
    print("🚀 启动后端服务...")
    try:
        # 确保在项目根目录
        os.chdir(Path(__file__).parent)
        
        # 启动 FastAPI 后端
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  后端服务已停止")
    except Exception as e:
        print(f"❌ 后端启动失败: {e}")

def run_frontend():
    """启动前端服务"""
    print("🎨 启动前端服务...")
    try:
        # 切换到前端目录
        frontend_dir = Path(__file__).parent / "frontend"
        os.chdir(frontend_dir)
        
        # 检查是否已安装依赖
        if not (frontend_dir / "node_modules").exists():
            print("📦 安装前端依赖...")
            subprocess.run(["npm", "install"], check=True)
        
        # 启动前端开发服务器
        subprocess.run(["npm", "start"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  前端服务已停止")
    except Exception as e:
        print(f"❌ 前端启动失败: {e}")

def main():
    """主函数"""
    print("🌟 HAG 项目启动中...")
    print("=" * 50)
    
    # 检查依赖
    try:
        import uvicorn
        import fastapi
    except ImportError:
        print("❌ 缺少后端依赖，请运行: pip install fastapi uvicorn")
        return
    
    # 检查Node.js
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 请先安装 Node.js 和 npm")
        return
    
    try:
        # 在后台启动后端
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # 等待后端启动
        print("⏳ 等待后端服务启动...")
        time.sleep(3)
        
        # 启动前端（主线程）
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n👋 正在关闭服务...")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()