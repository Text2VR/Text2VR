#!/usr/bin/env python3
"""
DreamScene360 간단 파노라마 생성기
텍스트 입력 → 파노라마 생성
"""

import os
import sys
import argparse

# DreamScene360 모듈 임포트
sys.path.append('stitch_diffusion/kohya_trainer')
sys.path.append('stitch_diffusion/kohya_trainer/library')  # library 모듈 경로 추가

try:
    from stitch_diffusion.kohya_trainer.StitchDiffusionPipeline import StitchDiffusion, my_args
    STITCHDIFFUSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ StitchDiffusion import failed: {e}")
    STITCHDIFFUSION_AVAILABLE = False
from Text2PanoRunner import Text2PanoRunner

def generate_panorama(text_prompt, output_dir="panorama_output", api_key=None, use_self_refinement=False):
    """텍스트로부터 파노라마 생성"""
    print(f"🎨 Generating panorama from: '{text_prompt}'")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 텍스트 파일 생성
    text_file = os.path.join(output_dir, "prompt.txt")
    with open(text_file, 'w') as f:
        f.write(text_prompt)
    
    if use_self_refinement and api_key:
        # Self-refinement 사용 (GPT-4o로 품질 개선)
        print("📝 Using self-refinement with GPT-4o...")
        runner = Text2PanoRunner(
            api_key=api_key,
            testfile=text_file,
            num_prompt=3,
            max_rounds=3,
            foldername="panorama_scene"
        )
        runner.run_command()
        
        # 가능한 결과 경로들 확인
        possible_paths = [
            "self_refinement/panorama_scene/iter_best/image.png",
            "candidates/panorama_scene/iter_best/image.png", 
            "panorama_scene/iter_best/image.png",
            "iter_best/image.png"
        ]
        
        pano_path = None
        for path in possible_paths:
            if os.path.exists(path):
                pano_path = path
                print(f"📁 Found result at: {path}")
                break
        
        # 결과를 출력 디렉토리로 복사
        if pano_path and os.path.exists(pano_path):
            final_path = os.path.join(output_dir, "panorama.png")
            os.system(f"cp '{pano_path}' '{final_path}'")
            return final_path
        else:
            print("❌ Self-refinement output not found, trying basic generation...")
    
    # 기본 StitchDiffusion 사용
    if not STITCHDIFFUSION_AVAILABLE:
        print("❌ StitchDiffusion not available due to import errors")
        return None
        
    print("🖼️ Using basic StitchDiffusion...")
    try:
        sd = StitchDiffusion(my_args)
        pano_path = os.path.join(output_dir, "panorama.png")
        sd.inference(text_prompt, savename=pano_path)
        
        if os.path.exists(pano_path):
            return pano_path
        else:
            print("❌ StitchDiffusion output not created")
            return None
    except Exception as e:
        print(f"❌ StitchDiffusion failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='DreamScene360 Simple Panorama Generator')
    parser.add_argument('--text', type=str, required=True, help='Text prompt for panorama generation')
    parser.add_argument('--output_dir', type=str, default='panorama_output', help='Output directory')
    parser.add_argument('--api_key', type=str, help='OpenAI API key (required for self-refinement)')
    parser.add_argument('--self_refinement', action='store_true', help='Use self-refinement for better quality (requires API key)')
    
    args = parser.parse_args()
    
    # Self-refinement을 사용하려면 API 키가 필요
    if args.self_refinement and not args.api_key:
        print("❌ Self-refinement requires --api_key")
        return
    
    print("🚀 DreamScene360 Simple Panorama Generator")
    print("=" * 50)
    print(f"📝 Text prompt: {args.text}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔧 Self-refinement: {'Yes' if args.self_refinement else 'No'}")
    print("=" * 50)
    
    # 파노라마 생성
    pano_path = generate_panorama(
        args.text, 
        args.output_dir,
        args.api_key,
        args.self_refinement
    )
    
    if pano_path and os.path.exists(pano_path):
        print(f"✅ Panorama generation completed!")
        print(f"🖼️ Result saved to: {pano_path}")
        
        # 파일 정보 출력
        file_size = os.path.getsize(pano_path) / (1024 * 1024)  # MB
        print(f"📊 File size: {file_size:.2f} MB")
        
        # 이미지 해상도 정보
        try:
            from PIL import Image
            with Image.open(pano_path) as img:
                print(f"📐 Resolution: {img.width} x {img.height}")
        except:
            pass
            
    else:
        print("❌ Panorama generation failed!")

if __name__ == "__main__":
    main()


    