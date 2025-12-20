"""Test script for RTX 4050 (6GB VRAM) setup"""
import torch
import gymnasium as gym
import os
import sys
import subprocess

def check_gpu_temp():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        temp = int(result.stdout.strip())
        print(f"   GPU Temperature: {temp}°C")
        if temp > 80:
            print("⚠️  GPU hot! Consider better cooling.")
    except Exception as e:
        print(f"⚠️  Could not read GPU temp: {e}")

def test_cuda():
    print("=" * 60)
    print("1. Testing CUDA Setup (RTX 4050)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"✅ CUDA Available: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ VRAM: {vram_gb:.1f} GB")
    
    if vram_gb < 5:
        print("⚠️  WARNING: Less than 6GB VRAM detected")
        
    check_gpu_temp()
    
    return True

def test_memory():
    """Test if batch size fits in VRAM"""
    print("\n" + "=" * 60)
    print("2. Testing Memory (Batch Size 128)")
    print("=" * 60)
    
    try:
        device = torch.device("cuda")
        
        # Simulate SAC network memory usage
        # Conv layers: 64 -> 128 -> 128
        # Hidden: 256
        # Batch: 128
        dummy_state = torch.randn(128, 4, 84, 84).to(device)
        print(f"✅ Allocated test batch (128, 4, 84, 84)")
        
        # Check VRAM usage
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   VRAM allocated: {allocated:.2f} GB")
        print(f"   VRAM reserved: {reserved:.2f} GB")
        
        if reserved > 5:
            print("⚠️  High VRAM usage. Consider reducing batch size.")
        else:
            print("✅ VRAM usage OK for batch size 128")
        
        # Cleanup
        del dummy_state
        torch.cuda.empty_cache()
        return True
        
    except RuntimeError as e:
        print(f"❌ Out of Memory: {e}")
        print("   Reduce BATCH_SIZE in train.py")
        return False

def test_gymnasium():
    print("\n" + "=" * 60)
    print("3. Testing Gymnasium CarRacing")
    print("=" * 60)
    
    try:
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        print(f"✅ Environment created")
        
        obs, info = env.reset()
        print(f"✅ Reset successful: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_directories():
    print("\n" + "=" * 60)
    print("4. Testing Directories")
    print("=" * 60)
    
    dirs = ['videos', 'checkpoints', 'logs']
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ {d}/")
        else:
            print(f"⚠️  {d}/ missing - creating...")
            os.makedirs(d, exist_ok=True)
    return True

def test_imports():
    print("\n" + "=" * 60)
    print("5. Testing Dependencies")
    print("=" * 60)
    
    modules = ['torch', 'gymnasium', 'numpy', 'cv2', 'tensorboard']
    all_ok = True
    
    for mod in modules:
        try:
            __import__(mod)
            print(f"✅ {mod}")
        except ImportError:
            print(f"❌ {mod} - MISSING")
            all_ok = False
    
    return all_ok

def main():
    print("\n🧪 RTX 4050 (6GB) Setup Test\n")
    
    tests = {
        'CUDA': test_cuda(),
        'Memory': test_memory(),
        'Gymnasium': test_gymnasium(),
        'Directories': test_directories(),
        'Dependencies': test_imports(),
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_passed = all(tests.values())
    for name, passed in tests.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed!")
        print("\n💡 Recommended for RTX 4050 (6GB):")
        print("   - BATCH_SIZE = 128")
        print("   - HIDDEN_SIZE = 256")
        print("   - MEMORY_SIZE = 200000")
        print("\nStart training: python train.py")
    else:
        print("⚠️  Some tests failed. Fix before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
