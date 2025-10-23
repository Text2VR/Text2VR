#!/usr/bin/env python3
"""
Simple test script for LangGraph workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.workflows.workflow import create_workflow

def test_workflow():
    print("🚀 Testing LangGraph workflow...")
    
    try:
        # Create workflow
        workflow = create_workflow()
        print("✅ Workflow created successfully")
        
        # Test input
        test_input = {
            "user_input": "실내 모던 침실",
            "rewritten_query": "",
            "scene_name": "",
            "panorama_path": "",
            "segmentation_data": {},
            "messages": []
        }
        
        print(f"📝 Input: {test_input['user_input']}")
        print("🔄 Running workflow...")
        
        # Run workflow
        result = workflow.invoke(test_input)
        
        print("\n=== RESULTS ===")
        print(f"📝 Original: {result['user_input']}")
        print(f"✨ Rewritten: {result['rewritten_query']}")
        print(f"🎬 Scene: {result['scene_name']}")
        print(f"🖼️  Panorama: {result['panorama_path']}")
        
        if result['panorama_path']:
            print("✅ Workflow completed successfully!")
        else:
            print("❌ Workflow failed - no panorama generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_workflow()