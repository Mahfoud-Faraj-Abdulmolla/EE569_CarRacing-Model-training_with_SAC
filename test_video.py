import gymnasium as gym
import os

def test_video_recording():
    print("🎥 Testing Video Recording...")
    
    video_dir = "./videos_test"
    os.makedirs(video_dir, exist_ok=True)
    
    try:
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=video_dir, 
            episode_trigger=lambda x: True,
            name_prefix="test_video"
        )
        
        print("✅ Wrapper created successfully")
        
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
        
        env.close()
        print(f"✅ Video saved to {video_dir}")
        
        # Verify file exists
        files = os.listdir(video_dir)
        mp4_files = [f for f in files if f.endswith('.mp4')]
        
        if len(mp4_files) > 0:
            print(f"✅ Found video file: {mp4_files[0]}")
            return True
        else:
            print("❌ No .mp4 file found!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if test_video_recording():
        print("\n🎉 Video recording works! You can now run full training.")
    else:
        print("\n❌ Video recording failed.")
