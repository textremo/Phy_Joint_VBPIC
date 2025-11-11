# import sys
# import platform

# print("=" * 50)
# print("系统诊断信息")
# print("=" * 50)
# print(f"操作系统: {platform.system()} {platform.release()}")
# print(f"系统架构: {platform.architecture()}")
# print(f"处理器: {platform.processor()}")
# print(f"Python版本: {sys.version}")
# print(f"Python路径: {sys.executable}")

# # 检查重要DLL
# import os
# dll_paths = os.environ.get('PATH', '').split(';')
# print("\n重要的DLL路径:")
# for path in dll_paths:
#     if 'windows' in path.lower() or 'system32' in path.lower():
#         print(f"  {path}")

# print("\n尝试导入TensorFlow...")
# try:
#     import tensorflow as tf
#     print("✓ TensorFlow导入成功!")
#     print(f"TensorFlow版本: {tf.__version__}")
# except Exception as e:
#     print(f"✗ TensorFlow导入失败: {e}")
    
    
try:
    import tensorflow as tf
except:
    print("tensorflow fails");