import asyncio
import sys
import os
import json
import base64
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from browser_use import Agent
from langchain.callbacks.base import BaseCallbackHandler

# Add project root to Python path
sys.path.insert(0, '/Users/xueyingjia/Documents/GitHub/browser-use')
from dotenv import load_dotenv
load_dotenv()

# Create a custom callback handler for capturing LLM conversations
class MessageCaptureHandler(BaseCallbackHandler):
    def __init__(self, details_dir):
        super().__init__()
        self.conversation_history = []
        self.details_dir = details_dir

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
        conversation_file = f"{self.details_dir}/conversation_{step_number:03d}.json"
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump(self.current_conversation, f, indent=2, ensure_ascii=False)

def load_tasks_from_dataset(split="test", num_tasks=None, website_filter=None):
    """
    Load tasks from the XueyingJia/online-mind2web-sorted dataset.
    
    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        num_tasks: Maximum number of tasks to load (None for all)
        website_filter: Optional list of websites to filter tasks by
        
    Returns:
        List of (task_id, task_text) tuples
    """
    print(f"Loading tasks from XueyingJia/online-mind2web-sorted dataset, split: {split}")
    dataset = load_dataset("XueyingJia/online-mind2web-sorted", split=split)
    
    tasks = []
    for item in dataset:
        task_id = item.get('id', '')
        task_text = item.get('confirmed_task', '')
        website = item.get('website', '')
        
        if not task_id or not task_text:
            continue
            
        # Apply website filter if specified
        if website_filter and website not in website_filter:
            continue
            
        tasks.append((task_id, task_text))
        
        # Stop if we've reached the requested number of tasks
        if num_tasks is not None and len(tasks) >= num_tasks:
            break
    
    print(f"Loaded {len(tasks)} tasks")
    return tasks

async def process_task(task_id: str, task: str, llm) -> Dict[str, Any]:
    """
    Process a single task and return the result.
    
    Args:
        task_id: Unique identifier for the task
        task: Task description text
        llm: Language model to use
        
    Returns:
        Dictionary with task results
    """
    print(f"\n{'='*80}\nProcessing task: {task_id}\n{task}\n{'='*80}")
    
    # Create directories for this task
    task_result_dir = f"task_execution_{task_id}"
    os.makedirs(task_result_dir, exist_ok=True)
    screenshots_trajectory_dir = os.path.join(task_result_dir, "trajectory")
    os.makedirs(screenshots_trajectory_dir, exist_ok=True)
    details_dir = os.path.join(task_result_dir, "details")
    os.makedirs(details_dir, exist_ok=True)
    
    # Create local variables for this task
    action_history = []
    conversation_history = []
    initial_actions = []  # {"go_to_url": {"url": "https://www.bestbuy.com/"}}
    
    # Add initial actions to action history if any
    for i, action in enumerate(initial_actions):
        action_name = next(iter(action.keys()))
        action_params = action.get(action_name, {})
        action_history.append(f"Step {i+1}: Initialize -> {action_name}({action_params})")
    
    # Create a message capture handler for this task
    message_capture_handler = MessageCaptureHandler(details_dir)
    
    # Create task-specific callbacks
    async def save_step_screenshots(browser_state_summary, agent_output, step_number):
        """Save screenshots and action history after each step"""
        nonlocal action_history
        
        # Only save if there is a screenshot
        if browser_state_summary.screenshot:
            # Filename for the screenshot
            filename = f"{screenshots_trajectory_dir}/{step_number:03d}_screenshot.png"
            
            # Decode the Base64 screenshot data and save it to a file
            try:
                screenshot_data = base64.b64decode(browser_state_summary.screenshot)
                with open(filename, "wb") as f:
                    f.write(screenshot_data)
                print(f"📸 Task {task_id}: Save step {step_number} screenshot: {filename}")
                
                # Save additional state information about the step
                info_filename = f"{details_dir}/step_{step_number:03d}_info.txt"
                with open(info_filename, "w", encoding="utf-8") as f:
                    f.write(f"URL: {browser_state_summary.url}\n")
                    f.write(f"Title: {browser_state_summary.title}\n\n")
                    f.write(f"Thoughts: {agent_output.current_state.memory}\n")
                    f.write(f"Goal: {agent_output.current_state.next_goal}\n")

                    # Save actions in a readable format
                    f.write("\nActions:\n")
                    for i, action in enumerate(agent_output.action):
                        action_data = action.model_dump(exclude_unset=True)
                        f.write(f"  {i+1}. {action_data}\n")
                
                # Record the action in the history
                action_description = f"Step {step_number}: {agent_output.current_state.next_goal} -> "
                action_details = []
                for action in agent_output.action:
                    action_data = action.model_dump(exclude_unset=True)
                    action_name = next(iter(action_data.keys())) if action_data else 'unknown'
                    action_params = action_data.get(action_name, {})
                    action_details.append(f"{action_name}({action_params})")
                
                # Combine action details into a single string
                combined_action = f"{action_description}{', '.join(action_details)}"
                action_history.append(combined_action)   

                # Update the result.json file with the current step's action history
                result_data = {
                    "task_id": task_id,
                    "task": task,
                    "action_history": action_history,
                }
                
                with open(f"{task_result_dir}/result.json", "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                    
                print(f"✅ Task {task_id}: Updated result.json: {step_number} steps completed.")
                    
            except Exception as e:
                print(f"❌ Task {task_id}: Failed to save step data: {e}")

    # Task completion callback
    async def on_task_complete(agent_history_list):
        """Callback function to save final result after task completion"""
        nonlocal action_history, conversation_history
        
        # Check if the task execution was successful
        is_successful = agent_history_list.is_successful()
        success_status = "success" if is_successful else "failure"
        
        # Create the final result with task_id, task, action history, and success status
        final_result = {
            "task_id": task_id,
            "task": task,
            "action_history": action_history,
            "success": is_successful,
            "status": success_status,
            "total_steps": len(agent_history_list.history),
            "duration_seconds": agent_history_list.total_duration_seconds()
        }
        
        # Save the final result to a JSON file
        with open(f"{task_result_dir}/result.json", "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        # Save all conversations to a single file
        with open(f"{details_dir}/all_conversations.json", "w", encoding="utf-8") as f:
            json.dump(message_capture_handler.conversation_history, f, indent=2, ensure_ascii=False)
        
        print(f"🏁 Task {task_id} completed! Result: {success_status}")
        
        return final_result

    # Create and run the agent
    agent = Agent(
        task=task, 
        llm=llm,
        initial_actions=initial_actions, 
        register_new_step_callback=save_step_screenshots, 
        register_done_callback=on_task_complete,
        save_conversation_path=f"{details_dir}/raw_conversation"
    )
    
    try:
        # Run the agent and wait for it to complete
        await agent.run()
        
        # Get the result from the result.json file
        with open(f"{task_result_dir}/result.json", "r", encoding="utf-8") as f:
            result = json.load(f)
        
        return result
    except Exception as e:
        print(f"❌ Error processing task {task_id}: {str(e)}")
        # Create an error result
        error_result = {
            "task_id": task_id,
            "task": task,
            "error": str(e),
            "status": "error",
            "success": False
        }
        
        # Save the error result
        with open(f"{task_result_dir}/result.json", "w", encoding="utf-8") as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
            
        return error_result

async def run_tasks_batch(tasks, batch_size=3, max_concurrent=3):
    """
    Run multiple tasks in batches.
    
    Args:
        tasks: List of (task_id, task) tuples to process
        batch_size: Number of tasks to process in each batch
        max_concurrent: Maximum number of tasks to run concurrently
        
    Returns:
        List of task results
    """
    all_results = []
    
    # Init the LLM model
    llm = ChatOpenAI(
        model="neulab/gpt-4o-mini-2024-07-18",
        temperature=0.0,
        openai_api_key=os.environ.get("LITELLM_API_KEY"),
        openai_api_base="https://cmu.litellm.ai"
    )
    
    # Process tasks in batches
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}/{len(tasks) // batch_size + 1} ({len(batch)} tasks)")
        
        # Run tasks concurrently with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(task_id, task):
            async with semaphore:
                return await process_task(task_id, task, llm)
        
        # Create tasks and run them
        batch_tasks = [process_with_semaphore(task_id, task) for task_id, task in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process results
        for result in batch_results:
            if isinstance(result, Exception):
                print(f"❌ Task failed with exception: {result}")
            else:
                all_results.append(result)
        
        print(f"✅ Completed batch {i // batch_size + 1} ({len(batch_results)} tasks)")
    
    return all_results

async def main():
    parser = argparse.ArgumentParser(description="Run multiple Mind2Web tasks")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to use")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks to run (default: 5)")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size (default: 3)")
    parser.add_argument("--max_concurrent", type=int, default=2, help="Maximum concurrent tasks (default: 2)")
    parser.add_argument("--website", nargs="+", help="Filter tasks by website (e.g. 'bestbuy')")
    args = parser.parse_args()
    
    # Load tasks from dataset
    tasks = load_tasks_from_dataset(
        split=args.split,
        num_tasks=args.num_tasks,
        website_filter=args.website
    )
    
    if not tasks:
        print("No tasks to process! Check your filter settings.")
        return
    
    print(f"Starting to process {len(tasks)} tasks in batches of {args.batch_size}, "
          f"max {args.max_concurrent} concurrent tasks")
    
    # Process all tasks
    results = await run_tasks_batch(
        tasks,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    )
    
    # Save summary of all results
    summary_file = "task_execution_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_tasks": len(tasks),
            "completed_tasks": len(results),
            "success_count": sum(1 for r in results if r.get("success", False)),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Completed {len(results)}/{len(tasks)} tasks")
    print(f"Successful tasks: {sum(1 for r in results if r.get('success', False))}/{len(results)}")
    print(f"Summary saved to {summary_file}")
    print(f"{'='*80}")

if __name__ == '__main__':
    asyncio.run(main())