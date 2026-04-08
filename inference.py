"""
MANDATORY INFERENCE SCRIPT
Participant LLM evaluation hook for ONDCAgentEnv
"""

import os
import re
import math
import textwrap
from openai import OpenAI

# Import the specific environment this repository uses
from ondc_env import ONDCAgentEnv, EnvConfig

# ---------------------------------------------------------
# MANDATORY VARIABLES (As requested by the eval framework)
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ONDC Action mapping for reference in the prompt
VALID_ACTIONS = {
    0: "SEARCH_PRODUCTS",
    1: "SELECT_SELLER_0",
    2: "SELECT_SELLER_1",
    3: "SELECT_SELLER_2",
    4: "SELECT_SELLER_3",
    5: "SELECT_SELLER_4",
    6: "INIT_ORDER",
    7: "CONFIRM_ORDER",
    8: "CANCEL_BEFORE_CONFIRM",
    9: "TRACK_ORDER",
    10: "ACCEPT_DELIVERY",
    11: "CANCEL_ORDER",
    12: "RETURN_ITEM",
    13: "FILE_GRIEVANCE",
    14: "WAIT"
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an intelligent AI buyer navigating the ONDC (Open Network for Digital Commerce) protocol.
    Your goal is to successfully purchase an item while managing your budget and keeping delivery times low.
    
    You must reply with EXACTLY ONE action integer from 0 to 14.
    
    Valid actions are based on the Beckn protocol phase you are currently in.
    0: SEARCH_PRODUCTS (Search phase)
    1-5: SELECT_SELLER_0 through SELECT_SELLER_4 (Select phase)
    6: INIT_ORDER (Init phase)
    7: CONFIRM_ORDER (Confirm phase)
    8: CANCEL_BEFORE_CONFIRM
    9: TRACK_ORDER (Track phase)
    10: ACCEPT_DELIVERY (Post-order phase)
    11: CANCEL_ORDER
    12: RETURN_ITEM
    13: FILE_GRIEVANCE
    14: WAIT (Available anytime to wait for sellers/events)
    
    Do not include explanations, text, or formatting. Just the integer ID of the action.
    """
)

def state_to_prompt(state):
    """ Converts the internal data structures of ONDCAgentEnv into readable text for the LLM """
    prompt = f"Current Environment State:\n"
    prompt += f"Remaining Budget: {state.budget}\n"
    prompt += f"Current Protocol Phase: {state.current_phase.name}\n"
    
    prompt += "Available Sellers:\n"
    for i, s in enumerate(state.sellers):
        if s.is_available:
            prompt += f"  Seller {i} -> Rating: {s.rating:.1f}, Price: {s.price:.2f}, ETA: {s.delivery_eta}, Stock: {s.stock}\n"

    seller_id = state.selected_offer.seller_id if state.selected_offer else "None"
    prompt += f"Currently Selected Seller ID: {seller_id}\n"
    if state.order_status:
        prompt += f"Order Status: {state.order_status.name}\n"
    
    return prompt

def extract_action(text_response):
    """ Gracefully pull action IDs (0-14) out of whatever the LLM spits out """
    match = re.search(r'\b(1[0-4]|[0-9])\b', text_response)
    if match:
        return int(match.group(1))
    return 14 # default to WAIT if parsing completely fails

def play_episode(client, env, model_name, task_name):
    obs, info = env.reset(seed=42)
    done = False
    
    print(f"[START] task={task_name}", flush=True)
    
    step_count = 0
    total_reward = 0.0

    while not done:
        step_count += 1
        state_prompt = state_to_prompt(env.state)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": state_prompt}
        ]

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=10
            )
            raw_output = response.choices[0].message.content.strip()
            action = extract_action(raw_output)
        except Exception as e:
            action = 14

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"[STEP] step={step_count} reward={reward}", flush=True)
        
        done = terminated or truncated

    # Sigmoid scaling to bound strictly between 0 and 1, as required by evaluator
    score = 1.0 / (1.0 + math.exp(-total_reward / 10.0))
    score = max(0.01, min(0.99, score))
    
    print(f"[END] task={task_name} score={score:.4f} steps={step_count}", flush=True)

def main():
    print(f"Connecting to Endpoint -> {API_BASE_URL}")
    print(f"Model ID -> {MODEL_NAME}")
    
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN is not set in environment. Inference might fail.")

    try:
        # Mandatory OpenAI Client Usage
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print("Failed to initialize OpenAI client:", e)
        return

    tasks = [
        {"name": "task_1_easy_budget", "budget": 2000.0, "max_steps": 50},
        {"name": "task_2_strict_urgency", "budget": 1000.0, "max_steps": 30},
        {"name": "task_3_low_budget", "budget": 500.0, "max_steps": 50},
    ]

    for t in tasks:
        config = EnvConfig(max_steps=t["max_steps"], initial_budget=t["budget"])
        env = ONDCAgentEnv(config)
        play_episode(client, env, MODEL_NAME or "test-model", t["name"])

if __name__ == "__main__":
    main()
