import asyncio
import sys
import os
import json
from langchain_openai import ChatOpenAI
from browser_use import Agent
import base64

sys.path.insert(0, '/Users/xueyingjia/Documents/GitHub/browser-use')
print("Python path:", sys.path)

from dotenv import load_dotenv
load_dotenv()


task = """Find cats available for adoption within 10 miles of zip code 94587, Young or adult-age cats, sorted by Oldest Addition. Operated within https://www.petfinder.com/ website."""
task_id = "c94551d2b18f9ad0ab31b0bd98ca42e3"

# create a directory to save screenshots and conversations
task_result_dir = f"task_execution_{task_id}"
os.makedirs(task_result_dir, exist_ok=True)
screenshots_trajectory_dir = os.path.join(task_result_dir, "trajectory")
os.makedirs(screenshots_trajectory_dir, exist_ok=True)
details_dir = os.path.join(task_result_dir, "details")
os.makedirs(details_dir, exist_ok=True)

# create a global variable to store action history
action_history = []
# store conversation history
conversation_history = []

# hold initial actions
initial_actions = [] # {"go_to_url": {"url": "https://www.bestbuy.com/"}}

# add initial actions to action history if any
for i, action in enumerate(initial_actions):
    action_name = next(iter(action.keys()))
    action_params = action.get(action_name, {})
    action_history.append(f"Step {i+1}: Initialize -> {action_name}({action_params})")


# define a callback function to save screenshots and action history
async def save_step_screenshots(browser_state_summary, agent_output, step_number):
    """save screenshots and action history after each step"""
    global action_history
    
    # only save if there is a screenshot
    if browser_state_summary.screenshot:
        # filename for the screenshot
        filename = f"{screenshots_trajectory_dir}/{step_number:03d}_screenshot.png"
        
        # decode the Base64 screenshot data and save it to a file
        try:
            screenshot_data = base64.b64decode(browser_state_summary.screenshot)
            with open(filename, "wb") as f:
                f.write(screenshot_data)
            print(f"📸 save step {step_number} screenshot: {filename}")
            
            # save additional state information about the step
            info_filename = f"{details_dir}/step_{step_number:03d}_info.txt"
            with open(info_filename, "w", encoding="utf-8") as f:
                f.write(f"URL: {browser_state_summary.url}\n")
                f.write(f"Title: {browser_state_summary.title}\n\n")
                f.write(f"Thoughts: {agent_output.current_state.memory}\n")
                f.write(f"Goal: {agent_output.current_state.next_goal}\n")

                # save actions in a readable format
                f.write("\nActions:\n")
                for i, action in enumerate(agent_output.action):
                    action_data = action.model_dump(exclude_unset=True)
                    f.write(f"  {i+1}. {action_data}\n")
            
            # record the action in the history
            action_description = f"Step {step_number}: {agent_output.current_state.next_goal}. Operated within website: {browser_state_summary.url}."
            action_details = []
            for action in agent_output.action:
                action_data = action.model_dump(exclude_unset=True)
                action_name = next(iter(action_data.keys())) if action_data else 'unknown'
                action_params = action_data.get(action_name, {})
                action_details.append(f"{action_name}({action_params})")
            
            # combine action details into a single string
            combined_action = f"{action_description} -> {', '.join(action_details)}"
            action_history.append(combined_action)   

            # update the result.json file with the current step's action history
            result_data = {
                "task_id": task_id,
                "task": task,
                "action_history": action_history,
            }
            
            with open(f"{task_result_dir}/result.json", "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Have already updated result.json: {step_number} steps completed.")
                
        except Exception as e:
            print(f"❌ Failed to save in json: {e}")


# This function will be called when the task is completed
async def on_task_complete(agent_history_list):
    """callback function to save final result after task completion"""
    global action_history, conversation_history
    
    # check if the task execution was successful
    is_successful = agent_history_list.is_successful()
    success_status = "success" if is_successful else "failure"
    
    # create the final result with task_id, task, action history, and success status
    final_result = {
        "task_id": task_id,
        "task": task,
        "action_history": action_history,
        "success": is_successful,
        "status": success_status,
        "total_steps": len(agent_history_list.history),
        "duration_seconds": agent_history_list.total_duration_seconds()
    }
    
    # save the final result to a JSON file
    with open(f"{task_result_dir}/result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    # Save all conversations to a single file
    with open(f"{details_dir}/all_conversations.json", "w", encoding="utf-8") as f:
        json.dump(conversation_history, f, indent=2, ensure_ascii=False)
    
    print("🏁 Task completed! The final result has been saved to result.json")
    print(f"📝 All conversations have been saved to {details_dir}/all_conversations.json")


# Custom message handler to capture messages and responses
def message_handler(messages, response):
    """Capture messages sent to LLM and the response"""
    global conversation_history
    
    # Store timestamp, messages and response
    conversation = {
        "timestamp": str(datetime.now()),
        "messages": [m.model_dump() for m in messages],  # Convert Message objects to dict
        "response": response.model_dump() if hasattr(response, "model_dump") else str(response)
    }
    
    # Add to history
    conversation_history.append(conversation)
    
    # Save conversation for this step
    step_number = len(conversation_history)
    conversation_file = f"{details_dir}/conversation_{step_number:03d}.json"
    with open(conversation_file, "w", encoding="utf-8") as f:
        json.dump(conversation, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Saved conversation for step {step_number}")
    
    return response  # Return the original response

# Init the LLM model with message handler
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler

# Create a custom callback handler
class MessageCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.conversation_history = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_conversation = {
            "timestamp": str(datetime.now()),
            "prompts": prompts,
            "messages": kwargs.get("messages", [])
        }

    def on_llm_end(self, response, **kwargs):
        self.current_conversation["response"] = response
        self.conversation_history.append(self.current_conversation)
        
        # Save conversation for this step
        step_number = len(self.conversation_history)
        conversation_file = f"{details_dir}/conversation_{step_number:03d}.json"
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(self.current_conversation, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Saved conversation for step {step_number}")

# Create the callback handler instance
message_capture_handler = MessageCaptureHandler()

# Init the LLM model
llm = ChatOpenAI(
    model="neulab/gpt-4o-mini-2024-07-18",
    temperature=0.0,
    openai_api_key=os.environ.get("LITELLM_API_KEY"),
    openai_api_base="https://cmu.litellm.ai",
    callbacks=[message_capture_handler]
)

agent = Agent(
    task=task, 
    llm=llm,  
    initial_actions=initial_actions, 
    register_new_step_callback=save_step_screenshots, 
    register_done_callback=on_task_complete,
    save_conversation_path=f"{details_dir}/raw_conversation"  # Enable built-in conversation saving
)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
