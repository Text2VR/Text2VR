#!/usr/bin/env python3
"""
Test script for PLY generation API endpoint
"""

import requests
import sys
import json

# API URL
API_URL = "http://localhost:8001"

def test_train_gaussian(panorama_path: str):
    """Test the /train_gaussian endpoint"""

    print("=" * 60)
    print("🧪 Testing /train_gaussian API endpoint")
    print("=" * 60)

    # Convert to container path
    container_path = panorama_path.replace(
        "/home/0in/workspace/Text2VR/", "/workspace/"
    )

    print(f"📁 Host path: {panorama_path}")
    print(f"📦 Container path: {container_path}")
    print()

    # Prepare request
    request_data = {
        "panorama_path": container_path,
        "scene_name": "test_scene",
        "iterations": 100,
        "save_iterations": [50, 100],
        "gen_res": 512,
        "white_background": False,
        "sh_degree": 3
    }

    print("📤 Request data:")
    print(json.dumps(request_data, indent=2))
    print()

    try:
        print("🚀 Sending request...")
        response = requests.post(
            f"{API_URL}/train_gaussian",
            json=request_data,
            timeout=600
        )

        print(f"📊 Response status: {response.status_code}")
        print()

        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print()
            print("📦 Response data:")
            print(json.dumps(result, indent=2))
            print()

            # Convert paths back to host
            if result.get("initial_ply_path"):
                host_path = result["initial_ply_path"].replace(
                    "/workspace/", "/home/0in/workspace/Text2VR/"
                )
                print(f"📍 Initial PLY (host): {host_path}")

            if result.get("trained_ply_paths"):
                print(f"📍 Trained PLY files ({len(result['trained_ply_paths'])}):")
                for ply_path in result["trained_ply_paths"]:
                    host_path = ply_path.replace(
                        "/workspace/", "/home/0in/workspace/Text2VR/"
                    )
                    print(f"   - {host_path}")

            if result.get("model_path"):
                host_path = result["model_path"].replace(
                    "/workspace/", "/home/0in/workspace/Text2VR/"
                )
                print(f"📍 Model path (host): {host_path}")

        else:
            print("❌ Failed!")
            print(f"Error: {response.text}")

    except requests.exceptions.Timeout:
        print("❌ Request timed out (>10 minutes)")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ply_api.py <panorama_path>")
        print()
        print("Example:")
        print("  python test_ply_api.py /home/0in/workspace/Text2VR/data/scene_xxx/panorama.png")
        print()
        print("Or use inpainted panorama:")
        print("  python test_ply_api.py /home/0in/workspace/Text2VR/inpainted_pano/scene_xxx/inpainted_panorama.png")
        sys.exit(1)

    panorama_path = sys.argv[1]
    test_train_gaussian(panorama_path)
