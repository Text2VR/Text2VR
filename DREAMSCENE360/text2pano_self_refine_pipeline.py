#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-refinement loop for text->panorama using GPT-4o + StitchDiffusion.

Main fixes:
- Avoid using extremely long folder names derived from the prompt.
- Use a short, stable folder name (args.foldername) and flat filenames under tmp/.
- Ensure all output directories exist before writing files.
- Keep final best image at: self_refinement/<foldername>/iter_best/image.png
  (compatible with pano_generator.py's search paths).
"""

import os
import time
import argparse
import json
import cv2
import base64
import random
from tqdm import tqdm
import requests
import torch
from PIL import Image

import sys
sys.path.append('stitch_diffusion/kohya_trainer')
from stitch_diffusion.kohya_trainer.StitchDiffusionPipeline import StitchDiffusion, my_args


# -----------------------------
# Utilities
# -----------------------------
def encode_img(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_img(image_path):
    base64_image = encode_img(image_path)
    image_meta = "data:image/png;base64" if 'png' in image_path.lower() else "data:image/jpeg;base64"
    return {
        "type": "image_url",
        "image_url": {
          "url": f"{image_meta},{base64_image}",
          "detail": "low"
        }
    }

def build_content_list(idea_transcript):
    """Build OpenAI content list mixing text and image_url blocks."""
    out = []
    for item in idea_transcript:
        if isinstance(item, str):
            out.append({"type": "text", "text": item})
        elif isinstance(item, dict) and "type" in item:
            out.append(item)
    return out

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def safe_join(*parts):
    return os.path.join(*[str(p) for p in parts])

# -----------------------------
# OpenAI call (GPT-4o)
# -----------------------------
api_key = None

def llm_request(transcript=None, temp=0.0, max_tokens=512):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        'model': 'gpt-4o',
        'max_tokens': max_tokens,
        'temperature': temp,
        'top_p': 0.5,
        'messages': transcript or []
    }
    wait = 10
    for _ in range(8):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions",
                              headers=headers, data=json.dumps(data))
            j = r.json()
            if r.status_code == 200 and 'choices' in j:
                return j["choices"][0]["message"]["content"]
            time.sleep(wait)
            temp = min(temp + 0.2, 1.0)
            data['temperature'] = temp
        except Exception:
            time.sleep(wait)
    return ""

# -----------------------------
# LLM prompt helpers
# -----------------------------
def llm_init_prompt(user_prompt, idea_transcript, args):
    system_content = (
        "You are a helpful assistant.\n\n"
        "Instruction: Convert the IDEA into concise image prompts.\n"
        "- Each prompt <= 50 words.\n"
        "- Describe concrete visual details; avoid abstract feelings.\n"
        "- Use comma-separated modifiers (style, lighting, etc.).\n"
        "- No mention of 'given image'; be self-contained."
    )
    user_content = build_content_list(idea_transcript)
    user_content.append({"type": "text",
                         "text": f"Write exactly {args.num_prompt} prompts. "
                                 "Each wrapped with <START> and <END>."})
    transcript = [{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}]
    resp = llm_request(transcript, temp=0.0)
    if '<START>' not in resp or '<END>' not in resp:
        resp = llm_request(transcript, temp=0.2)
    prompts = resp.split('<START>')[1:]
    prompts = [x.strip().split('<END>')[0] for x in prompts]
    return prompts

def llm_select_best(idea_transcript, listofimages, args):
    n = len(listofimages)
    system_content = (
        f"You are a helpful assistant.\n\n"
        f"You are a judge to rank {n} generated images (indexed 0..{n-1}). "
        "Score how well each matches the IDEA; return the best index wrapped "
        "with <START> and <END>."
    )
    user_content = build_content_list(idea_transcript)
    for i, path in enumerate(listofimages):
        user_content.append({"type": "text", "text": f"{i}. "})
        user_content.append(load_img(path))
    user_content.append({"type": "text",
                         "text": f"Return best index only, inside <START>..</END>."})
    transcript = [{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}]
    resp = llm_request(transcript, temp=0.0)
    if '<START>' not in resp or '<END>' not in resp:
        resp = llm_request(transcript, temp=0.2)
    if '<START>' in resp and '<END>' in resp:
        s = resp.split('<START>')[1].split('<END>')[0].strip()
        return int(s) if s.isdigit() else 0, resp
    return 0, resp

def llm_reflect(user_prompt, idea_transcript, image_history, prompt_history, args):
    round_id = len(image_history)
    system_content = (
        "You refine the next prompt by reflecting on the difference between IDEA "
        "and the current image. Provide one concrete REASON wrapped with "
        "<START>..</END> about what to improve next round."
    )
    user_content = build_content_list(idea_transcript)
    if round_id > 1:
        user_content.append({"type": "text", "text": "Iteration history:"})
        for r in range(0, len(image_history)-1):
            user_content.append({"type": "text",
                                 "text": f"Round {r+1} prompt: {prompt_history[r]}"})
            user_content.append(load_img(image_history[r]))
    user_content.append({"type": "text",
                         "text": f"Current round {round_id} prompt: {prompt_history[-1]}"})
    user_content.append(load_img(image_history[-1]))
    user_content.append({"type": "text",
                         "text": "Write REASON inside <START> and <END>."})
    transcript = [{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}]
    resp = llm_request(transcript, temp=0.0)
    if '<START>' not in resp or '<END>' not in resp:
        resp = llm_request(transcript, temp=0.2)
    if '<START>' in resp and '<END>' in resp:
        return resp.split('<START>')[1].split('<END>')[0].strip()
    return resp.strip()

def llm_revision(user_prompt, idea_transcript, image_history, prompt_history, reflection_history, args):
    round_id = len(image_history)
    system_content = (
        "Convert the IDEA into improved prompts for next round.\n"
        "- Each prompt <= 50 words, add concrete visual modifiers.\n"
        "- Address the REASON from previous round.\n"
        "- Avoid repeating previous prompts.\n"
        "- Return exactly the requested number; wrap each with <START>..</END>."
    )
    user_content = build_content_list(idea_transcript)
    if round_id > 1:
        user_content.append({"type": "text", "text": "Iteration history:"})
        for r in range(0, len(image_history)-1):
            user_content.append({"type": "text", "text": f"Round {r+1} prompt: {prompt_history[r]}"})
            user_content.append(load_img(image_history[r]))
            user_content.append({"type": "text", "text": f"Reason: {reflection_history[r]}"})
    user_content.append(load_img(image_history[-1]))
    user_content.append({"type": "text", "text": f"Reason: {reflection_history[-1]}"})
    user_content.append({"type": "text",
                         "text": f"Write {args.num_prompt} prompts wrapped in <START>..</END>."})
    transcript = [{"role": "system", "content": system_content},
                  {"role": "user", "content": user_content}]
    resp = llm_request(transcript, temp=0.0)
    if '<START>' not in resp or '<END>' not in resp:
        resp = llm_request(transcript, temp=0.2)
    prompts = resp.split('<START>')[1:]
    prompts = [x.strip().split('<END>')[0] for x in prompts]
    while len(prompts) < args.num_prompt:
        prompts.append("blank image")
    return prompts

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--testfile", type=str, default="testsample.txt")
    parser.add_argument("--num_img", type=int, default=1)
    parser.add_argument("--num_prompt", type=int, default=3)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--foldername", type=str, default="panorama_scene")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--final_name", type=str, default="image")
    args = parser.parse_args()
    assert args.num_img == 1

    global api_key
    api_key = args.api_key

    # Stable, short folder layout (NO prompt-derived subfolders).
    root_dir = safe_join("self_refinement", args.foldername)
    tmp_dir = safe_join(root_dir, "tmp")
    round1_dir = safe_join(root_dir, "round1")
    iter_dir = safe_join(root_dir, "iter")
    best_dir = safe_join(root_dir, "iter_best")
    for d in [root_dir, tmp_dir, round1_dir, iter_dir, best_dir]:
        ensure_dir(d)

    # Load prompts file (can contain multiple lines)
    sample_list = [x.strip() for x in open(args.testfile, 'r').read().splitlines()]
    t2i_model = StitchDiffusion(my_args)

    for sample_idx in tqdm(range(len(sample_list))):
        # Build IDEA transcript (supports legacy "<IMG>…<IMG>" blocks)
        user_prompt_full = sample_list[sample_idx]
        parts = user_prompt_full.split('<IMG>')
        idea_transcript = ["IDEA: %s." % parts[0]]
        for i in range(1, len(parts)):
            if i % 2 == 1:
                idea_transcript.append(load_img(parts[i]))
            else:
                idea_transcript.append(parts[i])
        idea_transcript.append("End of IDEA.\n")

        # Logging file (short name, not derived from prompt)
        text_record = safe_join(tmp_dir, "trace_rounds.txt")

        # -------- Iterative rounds --------
        image_history, prompt_history, reflection_history = [], [], []
        for r in range(args.max_rounds):
            if r == 0:
                llm_prompts = llm_init_prompt(parts[0], idea_transcript, args)
            else:
                llm_prompts = llm_revision(parts[0], idea_transcript, image_history, prompt_history, reflection_history, args)

            # Generate candidates this round → write under tmp/
            cand_paths = []
            for p_i in range(args.num_prompt):
                out_path = safe_join(tmp_dir, f"{r}_{p_i}_0.png")
                t2i_model.inference(llm_prompts[p_i], out_path)
                cand_paths.append(out_path)

            # Select best this round
            round_best_idx, select_resp = llm_select_best(idea_transcript, cand_paths, args)
            best_img_path = cand_paths[round_best_idx]
            image_history.append(best_img_path)
            prompt_history.append(llm_prompts[round_best_idx])

            # Copy round1 snapshot
            if r == 0:
                os.system(f'cp "{best_img_path}" "{safe_join(round1_dir, "seed.png")}"')

            # Reflection text (except last round)
            if r != args.max_rounds - 1:
                refl = llm_reflect(parts[0], idea_transcript, image_history, prompt_history, args)
            else:
                refl = ''
            reflection_history.append(refl)

            # Trace
            trace = []
            trace.append(f"=========== End of round {r} ===========")
            trace.append(f"prompt: {prompt_history[-1]}")
            trace.append(f"image:  {image_history[-1]}")
            trace.append(f"select: {select_resp[:300]}...")
            trace.append(f"reason: {refl}")
            with open(text_record, "a", encoding="utf-8") as f:
                f.write("\n".join(trace) + "\n")

        # Save last-round best into iter/
        os.system(f'cp "{image_history[-1]}" "{safe_join(iter_dir, "last_round.png")}"')

        # Global best over all rounds (start at 0)
        global_best_idx, global_resp = llm_select_best(idea_transcript, image_history, args)
        final_out = safe_join(best_dir, f"{args.final_name}.png")
        os.system(f'cp "{image_history[global_best_idx]}" "{final_out}"')

        # Also copy round summaries under tmp/ for convenience
        for tag, src in [("round1.png", safe_join(round1_dir, "seed.png")),
                         ("iter.png", safe_join(iter_dir, "last_round.png")),
                         ("iter_best.png", final_out)]:
            if os.path.exists(src):
                os.system(f'cp "{src}" "{safe_join(tmp_dir, tag)}"')


if __name__ == '__main__':
    main()


'''
import os
import time
import argparse
import csv, json
import cv2, base64
from tqdm import tqdm
import requests
import random
import torch
from datetime import datetime
from PIL import Image

import sys
sys.path.append('stitch_diffusion/kohya_trainer')

from stitch_diffusion.kohya_trainer.StitchDiffusionPipeline import StitchDiffusion, my_args

def encode_img(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_img(image_path):
    base64_image = encode_img(image_path)
    image_meta = "data:image/png;base64" if 'png' in image_path else "data:image/jpeg;base64"
    img_dict = {
        "type": "image_url",
        "image_url": {
          "url": f"{image_meta},{base64_image}",
          "detail": "low"
        }
    }
    return img_dict

def llm_request(transcript=None, temp=0.):
    max_tokens = 512
    wait_time = 10

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        'model': 'gpt-4o',
        'max_tokens': max_tokens, 
        'temperature': temp,
        'top_p': 0.5,
        'messages': transcript if transcript is not None else []
    }

    response_text, retry, response_json = '', 0, None
    while len(response_text) < 2:
        retry += 1
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data)) 
            response_json = response.json()
        except Exception as e:
            if random.random() < 1: print(e)
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            print(response.headers, response.content)
            if random.random() < 0.01: print(f"The response status code for is {response.status_code} (Not OK)")
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]
    return response_json["choices"][0]["message"]["content"]

def build_content_list(idea_transcript):
    """Helper function to build content list properly"""
    content_list = []
    for item in idea_transcript:
        if isinstance(item, str):
            content_list.append({"type": "text", "text": item})
        elif isinstance(item, dict) and "type" in item:
            content_list.append(item)
    return content_list

def llm_init_prompt(user_prompt, img_prompt, idea_transcript, args):
    system_content = ("You are a helpful assistant.\n\n"
                     "Instruction: Given a user imagined IDEA of the scene, converting the IDEA into a self-contained sentence prompt that will be used to generate an image.\n"
                     "Here are some rules to write good prompts:\n"
                     "- Each prompt should consist of a description of the scene followed by modifiers divided by commas.\n"
                     "- The modifiers should alter the mood, style, lighting, and other aspects of the scene.\n"
                     "- Multiple modifiers can be used to provide more specific details.\n"
                     "- When generating prompts, reduce abstract psychological and emotional descriptions.\n"
                     "- When generating prompts, explain images and unusual entities in IDEA with detailed descriptions of the scene.\n"
                     "- Do not mention 'given image' in output, use detailed texts to describe the image in IDEA instead.\n"
                     "- Generate diverse prompts.\n"
                     "- Each prompt should have no more than 50 words.\n")
    
    user_content = build_content_list(idea_transcript)
    user_content.append({
        "type": "text", 
        "text": f"Based on the above information, you will write {args.num_prompt} detailed prompts exactly about the IDEA follow the rules. Each prompt is wrapped with <START> and <END>.\n"
    })
    
    transcript = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    response = llm_request(transcript)
    if '<START>' not in response or '<END>' not in response:  # one format retry
        response = llm_request(transcript, temp=0.1)
    if args.verbose:
        print('llm_init_prompt    IDEA: %s.\n %s\n' % (user_prompt, response))
    
    prompts = response.split('<START>')[1:]
    prompts = [x.strip().split('<END>')[0] for x in prompts]
    return prompts

def llm_reflection_prompt_selectbest(user_prompt, img_prompt, idea_transcript, listofimages, args):
    num_img = len(listofimages)
    
    system_content = (f"You are a helpful assistant.\n\n"
                     f"You are a judge to rank provided images. Below are {num_img} images generated by an AI art generation model, indexed from 0 to {num_img-1}. "
                     f"From scale 1 to 10, decide how similar each image is to the user imagined IDEA of the scene.")
    
    user_content = build_content_list(idea_transcript)
    
    for img_i in range(num_img):
        user_content.append({"type": "text", "text": f"{img_i}. "})
        user_content.append(load_img(listofimages[img_i]))

    user_content.append({
        "type": "text", 
        "text": f"Let's think step by step. Check all aspects to see how well these images strictly follow the content in IDEA, including having correct object counts, attributes, entities, relationships, sizes, appearance, and all other descriptions in the IDEA. Then give a score for each input images. Finally, consider the scores and select the image with the best overall quality with image index 0 to {num_img-1} wrapped with <START> and <END>. Only wrap single image index digits between <START> and <END>."
    })

    transcript = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    response = llm_request(transcript)
    if '<START>' not in response or '<END>' not in response:  # one format retry
        response = llm_request(transcript, temp=0.1)
    if args.verbose:
        print('llm_reflection_prompt_selectbest\n %s\n' % (response))
    
    if '<START>' not in response or '<END>' not in response:
        return random.randint(0, num_img-1), response
    
    prompts = response.split('<START>')[1]
    prompts = prompts.strip().split('<END>')[0]
    return int(prompts) if prompts.isdigit() else random.randint(0, num_img-1), response

def llm_reflection_prompt_textreflection(user_prompt, img_prompt, idea_transcript, round_best, listofimages, image_history, prompt_history, reflection_history, args):
    current_round = len(image_history)
    
    system_content = ("You are a helpful assistant.\n\n"
                     "You are iteratively refining the sentence prompt by analyzing the images produced by an AI art generation model, seeking to find out the differences between the user imagined IDEA of the scene and the actual output.\n"
                     "If the generated image is not perfect, provide key REASON on ways to improve the image and sentence prompt to better follow the user imagined IDEA of the scene. Here are some rules to write good key REASON:\n"
                     "- Carefully compare the current image with the IDEA to strictly follow the details described in the IDEA, including object counts, attributes, entities, relationships, sizes, and appearance. Write down what is different in detail.\n"
                     "- Avoid hallucinating information or asks that is not mentioned in IDEA.\n"
                     "- Explain images and unusual entities in IDEA with detailed text descriptions of the scene.\n"
                     "- Explain how to modify prompts to address the given reflection reason.\n"
                     "- Focus on one thing to improve in each REASON. \n"
                     "- Avoid generating REASON identical with the REASON in previous rounds.\n")
    
    user_content = build_content_list(idea_transcript)
    user_content.append({"type": "text", "text": f"This is the round {current_round} of the iteration.\n"})
    
    if current_round != 1:
        user_content.append({"type": "text", "text": "The iteration history are:\n"})
        for rounds in range(0, len(image_history)-1):
            user_content.append({
                "type": "text", 
                "text": f"Round {rounds+1}:\nGenerated sentence prompt: {prompt_history[rounds]}\nCorresponding image generated by the AI art generation model:"
            })
            user_content.append(load_img(image_history[rounds]))
            user_content.append({"type": "text", "text": f"However, {reflection_history[rounds]}."})
    
    user_content.append({
        "type": "text", 
        "text": f"Generated sentence prompt for current round {current_round} is: {prompt_history[-1]}\nCorresponding image generated by the AI art generation model:"
    })
    user_content.append(load_img(image_history[-1]))
    user_content.append({
        "type": "text", 
        "text": "Based on the above information, you will write REASON that is wrapped with <START> and <END>.\n REASON: "
    })

    transcript = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    response = llm_request(transcript)
    if '<START>' not in response or '<END>' not in response:  # one format retry
        response = llm_request(transcript, temp=0.1)
    if args.verbose:
        print('llm_reflection_prompt_textreflection\n %s\n' % (response))
    
    if '<START>' not in response or '<END>' not in response:
        return response
    
    prompts = response.split('<START>')[1]
    prompts = prompts.strip().split('<END>')[0]
    return prompts

def llm_revision_prompt(user_prompt, img_prompt, idea_transcript, image_history, prompt_history, reflection_history, args):
    current_round = len(image_history)
    
    system_content = ("You are a helpful assistant.\n\n"
                     "Instruction: Given a user imagined IDEA of the scene, converting the IDEA into a sentence prompt that will be used to generate an image.\n"
                     "Here are some rules to write good prompts:\n"
                     "- Each prompt should consist of a description of the scene followed by modifiers divided by commas.\n"
                     "- The modifiers should alter the mood, style, lighting, spatial details, and other aspects of the scene.\n"
                     "- Multiple modifiers can be used to provide more specific details.\n"
                     "- When generating prompts, reduce abstract psychological and emotional descriptions.\n"
                     "- When generating prompts, explain images and unusual entities in IDEA with detailed descriptions of the scene.\n"
                     "- Do not mention 'given image' in output, use detailed texts to describe the image in IDEA.\n"
                     "- Generate diverse prompts.\n"
                     "- Output prompt should have less than 50 words.\n")
    
    user_content = build_content_list(idea_transcript)
    user_content.append({
        "type": "text", 
        "text": f"You are iteratively improving the sentence prompt by looking at the images generated by an AI art generation model and find out what is different from the given IDEA.\nThis is the round {current_round} of the iteration.\n"
    })
    
    if current_round != 1:
        user_content.append({"type": "text", "text": "The iteration history are:\n"})
        for rounds in range(0, len(image_history)-1):
            user_content.append({
                "type": "text", 
                "text": f"Round {rounds+1}:\nGenerated sentence prompt: {prompt_history[rounds]}\nCorresponding image generated by the AI art generation model:"
            })
            user_content.append(load_img(image_history[rounds]))
            user_content.append({"type": "text", "text": f"However, {reflection_history[rounds]}."})
    
    user_content.append({
        "type": "text", 
        "text": f"Generated sentence prompt for current round {current_round} is: {prompt_history[-1]}\nCorresponding image generated by the AI art generation model:"
    })
    user_content.append(load_img(image_history[-1]))
    user_content.append({"type": "text", "text": f"However, {reflection_history[-1]}."})
    
    user_content.append({
        "type": "text", 
        "text": f"Based on the above information, to improve the image, you will write {args.num_prompt} detailed prompts exactly about the IDEA follow the rules. Make description of the scene more detailed and add modifiers to address the given key reasons to improve the image. Avoid generating prompts identical with the ones in previous rounds. Each prompt is wrapped with <START> and <END>.\n"
    })

    transcript = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    response = llm_request(transcript)
    if '<START>' not in response or '<END>' not in response:  # one format retry
        response = llm_request(transcript, temp=0.1)
    if args.verbose:
        print('llm_revision_prompt    IDEA: %s.\n %s\n' % (user_prompt, response))
    
    prompts = response.split('<START>')[1:]
    prompts = [x.strip().split('<END>')[0] for x in prompts]
    while len(prompts) < args.num_prompt:
        prompts = prompts + ['blank image']
    return prompts

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_key", type=str, help="OpenAI GPT-4V API key; https://platform.openai.com/docs/guides/vision")
    parser.add_argument("--testfile", type=str, default="testsample.txt")
    parser.add_argument("--num_img", type=int, default=1, help="number of images to generate per prompt")
    parser.add_argument("--num_prompt", type=int, default=3, help="number of prompts to search each round")
    parser.add_argument("--max_rounds", type=int, default=3, help="max number of iter rounds")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--foldername", type=str, default="candidates")
    parser.add_argument("--strength", type=float, default=1.00, help="strength of img2img pipeline")
    parser.add_argument("--final_name", default='image', type=str, help="name of file with best img")
    args = parser.parse_args()
    assert(args.num_img == 1)

    global api_key
    api_key = args.api_key

    os.system('mkdir -p self_refinement/%s' % args.foldername)
    os.system('mkdir -p self_refinement/%s/iter' % args.foldername)
    os.system('mkdir -p self_refinement/%s/round1' % args.foldername)
    os.system('mkdir -p self_refinement/%s/iter_best' % args.foldername)
    os.system('mkdir -p self_refinement/%s/tmp' % args.foldername)

    sample_list = [x.strip() for x in list(open(args.testfile, 'r'))]
    t2i_model = StitchDiffusion(my_args)
    
    for sample_ii in tqdm(range(len(sample_list))):
        user_prompt, img_prompt = sample_list[sample_ii], None
        prompt_list = user_prompt.split('<IMG>')
        user_prompt = user_prompt.split('<IMG>')[0]  # legacy, for naming use only
        idea_transcript = []
        
        for ii in range(len(prompt_list)):
            if ii == 0:
                idea_transcript.append("IDEA: %s." % prompt_list[0])
            elif ii % 2 == 1:
                idea_transcript.append(load_img(prompt_list[ii]))
            elif ii % 2 == 0:
                idea_transcript.append("%s" % prompt_list[ii])
        idea_transcript.append("End of IDEA.\n")

        text_record = 'self_refinement/%s/tmp/%s.txt' % (args.foldername, user_prompt.replace(' ', '').replace('.', ''))
        os.system('mkdir -p self_refinement/%s/tmp/%s' % (args.foldername, user_prompt.replace(' ', '').replace('.', '')))

        ### LLM prompting iter
        current_prompts, prompt_history, select_history, image_history, reflection_history, bestidx_history = [], [], [], [], [], []
        
        for rounds in range(args.max_rounds):
            if args.verbose: 
                print('ROUND %d:\n' % rounds)
            
            ###### new rounds' prompt (init/revision)
            if rounds == 0:
                llm_prompts = llm_init_prompt(user_prompt, None, idea_transcript, args)
            else:
                llm_prompts = llm_revision_prompt(user_prompt, None, idea_transcript, image_history, prompt_history, reflection_history, args)
            current_prompts = llm_prompts
            
            ###### t2i generation
            for ii in range(args.num_prompt):
                for jj in range(args.num_img):
                    # t2i_model.inference(llm_prompts[ii], 'self_refinement/%s/tmp/%s/%d_%d_%d.png' % (args.foldername, user_prompt.replace(' ', '').replace('.', ''), rounds, ii, jj))
                    t2i_model.inference(llm_prompts[ii], 'self_refinement/%s/tmp/%s/%d_%d_%d.png' % (args.foldername, args.foldername, rounds, ii, jj))
            
            ###### reflection: first select best, then give reason to improve (i.e., reflection)
            round_best, select_response = llm_reflection_prompt_selectbest(user_prompt, img_prompt, idea_transcript, 
                                                                          ['self_refinement/%s/tmp/%s/%d_%d_0.png' % (args.foldername, user_prompt.replace(' ', '').replace('.', ''), rounds, ii) for ii in range(args.num_prompt)], args)
            
            ## select the best, give an index. two separate calls
            prompt_history.append(current_prompts[round_best])
            select_history.append('Round selection: %d. || ' % round_best + select_response)
            image_history.append('self_refinement/%s/tmp/%s/%d_%d_0.png' % (args.foldername, user_prompt.replace(' ', '').replace('.', ''), rounds, round_best))
            bestidx_history.append(round_best)
            
            if rounds != args.max_rounds - 1:
                reflection_text = llm_reflection_prompt_textreflection(user_prompt, img_prompt, idea_transcript, round_best, 
                                                                     ['self_refinement/%s/tmp/%s/%d_%d_0.png' % (args.foldername, user_prompt.replace(' ', '').replace('.', ''), rounds, ii) for ii in range(args.num_prompt)], 
                                                                     image_history, prompt_history, reflection_history, args)
            else:
                reflection_text = ''
            reflection_history.append(reflection_text)
            
            trace_string = ''
            trace_string += '===========\nEnd of round %d:\n' % rounds
            trace_string += 'user_prompt: %s\n' % user_prompt
            trace_string += 'image_history: %s\n' % image_history[-1]
            trace_string += 'select_history: %s\n' % select_history[-1]
            trace_string += 'prompt_history: %s\n' % prompt_history[-1]
            trace_string += 'reflection_history: %s\n==========\n' % reflection_history[-1]
            print(trace_string)
            
            with open(text_record, 'a') as f:
                f.write(trace_string)
            
            if rounds == 0:
                os.system('cp %s self_refinement/%s/round1/%s.png' % (image_history[-1], args.foldername, user_prompt.replace(' ', '').replace('.', '')))
        
        ## save indexed image
        os.system('cp %s self_refinement/%s/iter/%s.png' % (image_history[-1], args.foldername, user_prompt.replace(' ', '').replace('.', '')))

        start_ind = 0  # 1
        global_best, select_response = llm_reflection_prompt_selectbest(user_prompt, img_prompt, idea_transcript, image_history[start_ind:], args)
        global_best += start_ind
        os.system('cp %s self_refinement/%s/iter_best/%s.png' % (image_history[global_best], args.foldername, args.final_name))
        
        with open(text_record, 'a') as f:
            f.write('Final selection: %d. || ' % global_best + select_response)
            f.write('===========\nFinal Selection: Round: %d.\n===========' % global_best)
    
    for key in ['round1', 'iter', 'iter_best']:
        os.system('cp -r self_refinement/%s/%s self_refinement/%s/tmp/%s' % (args.foldername, key, args.foldername, key))

if __name__ == '__main__':
    main()
'''