import subprocess
import sys

def check_avx_support():
    """检查CPU是否支持AVX指令集"""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        print("CPU信息:")
        print(f"  型号: {info['brand_raw']}")
        print(f"  AVX支持: {'avx' in info['flags']}")
        print(f"  AVX2支持: {'avx2' in info['flags']}")
        return 'avx' in info['flags']
    except ImportError:
        print("安装py-cpuinfo来检测CPU功能...")
        return None

def check_windows_version():
    """检查Windows版本"""
    import platform
    print(f"Windows版本: {platform.platform()}")

if __name__ == "__main__":
    check_windows_version()
    avx_support = check_avx_support()
    
    if avx_support is False:
        print("\n⚠️ 你的CPU可能不支持AVX指令集！")
        print("这是TensorFlow 2.x的常见问题。")
        print("解决方案：安装支持旧CPU的TensorFlow版本")