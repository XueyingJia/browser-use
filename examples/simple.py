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

# create a directory to save screenshots
screenshots_dir = "agent_screenshots_with_history"
os.makedirs(screenshots_dir, exist_ok=True)

# create a global variable to store action history
action_history = []

# hold initial actions
initial_actions = []

# add initial actions to action history if any
for i, action in enumerate(initial_actions):
    action_name = next(iter(action.keys()))
    action_params = action.get(action_name, {})
    action_history.append(f"Initial Step {i+1}: Initialize -> {action_name}({action_params})")


# define a callback function to save screenshots and action history
async def save_step_screenshots(browser_state_summary, agent_output, step_number):
    """save screenshots and action history after each step"""
    global action_history
    
    # only save if there is a screenshot
    if browser_state_summary.screenshot:
        # filename for the screenshot
        filename = f"{screenshots_dir}/{step_number:03d}_screenshot.png"
        
        # decode the Base64 screenshot data and save it to a file
        try:
            screenshot_data = base64.b64decode(browser_state_summary.screenshot)
            with open(filename, "wb") as f:
                f.write(screenshot_data)
            print(f"📸 save step {step_number} screenshot: {filename}")
            
            # save additional state information about the step
            info_filename = f"{screenshots_dir}/step_{step_number:03d}_info.txt"
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
            action_description = f"Step {step_number}: {agent_output.current_state.next_goal}"
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
            
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Have already updated result.json: {step_number} steps completed.")
                
        except Exception as e:
            print(f"❌ Failed to save in json: {e}")

# This function will be called when the task is completed
async def on_task_complete(agent_history_list):
    """callback function to save final result after task completion"""
    global action_history
    
    # create the final result with task_id, task, and action history
    final_result = {
        "task_id": task_id,
        "task": task,
        "action_history": action_history,
    }
    
    # save the final result to a JSON file
    with open(f"{screenshots_dir}/result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    
    print("🏁 Task completed! The final result has been saved to result.json")

# Init the LLM model
llm = ChatOpenAI(
    model="neulab/gpt-4o-mini-2024-07-18",
    temperature=0.0,
    openai_api_key=os.environ.get("LITELLM_API_KEY"),
    openai_api_base="https://cmu.litellm.ai"
)
task = 'Open the page for the first Best Paper Award video recording of talks from ICLR 2016.'
task_id = "customized_id"
agent = Agent(task=task, llm=llm,  initial_actions=initial_actions, register_new_step_callback=save_step_screenshots, register_done_callback=on_task_complete)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
