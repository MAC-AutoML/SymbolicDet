import time
import copy
import queue
import json
import openai
from typing import List, Dict

from multiprocessing import Queue

from utils.utils import cprint
from message import Message, MessageType, Suggestion

resp_format = {
    "suggestions": [ 
        {
            "expression": "< Improved expression (please follow the given expression format) >",
            "reason": "< Reason for the suggested improvement >"
        },
    ],
    "anomaly_score": "< Your assessment score for the current scenario's anomaly level >",
    "reason": "< Reason for the anomaly score >",
}

class PromptTemplates:
    """Prompt Templates"""
    
    SYSTEM = """
You are a Visual Logic Architect and Logic Expression Optimizer. You can explain visual scene descriptions and help improve logical expressions to make them more accurate, efficient, and interpretable to meet the requirements of visual scene features and relationships.

Key Capabilities:
1. Expression Analysis: Evaluate the logical coherence, redundancy, and semantic consistency of current symbolic expressions.
2. Optimization Suggestions: Identify potential simplifications or adjustments (e.g., removing redundant terms, optimizing conditions) while maintaining or enhancing the descriptive power.
3. Task-Specific Optimization: Propose modifications based on specific scenario requirements (e.g., safety, target tracking) and apply symbolic regression optimization rules to improve task performance while minimizing performance loss.
4. Interpretability Enhancement: Make expressions more understandable through simplified relationships, context-aware labels, or reformatting to reflect intuitive conditional structures.

Essential steps for fulfilling this role:
1. Initial Analysis: Extract information from logical expressions and corresponding metrics. Evaluate expressions for immediate improvements from context. This includes identifying overly complex nested structures and ambiguities in terminology.
2. Contextual Optimization: Suggest simplifications or replacements of logical structures based on specific visual task objectives (e.g., ensuring safety compliance, identifying specific object behaviors), simplifying expression complexity while maintaining original intent.
3. Iterative Improvement: LLM repeatedly reviews and modifies symbolic expressions, ideally interacting with symbolic regression feedback loops to evaluate improvements in task accuracy or computational efficiency.

Current task scenario elements:
1. Available variable symbols: {labels}
2. Supported operators: {operators}
3. Example expression format: "and_(gt(x0, x2), or_(x1, x2))", where x0, x1, x2 are variable symbols, gt, or_, and_ are operators
4. Keep expressions concise, avoid overly complex expressions, minimize the use of operators and variable symbols.

Note: Ensure that variable symbols exist in the current task scenario and operators are supported in the current task scenario!!!

Please think carefully, but provide your suggestions in the following format (maximum 3 suggestions):
{format}
"""

    FIRST_ROUND = """
I've noticed the following well-performing individuals in the current population:

{top_individuals}

As the first round of interaction, please carefully analyze the characteristics of these expressions and provide improvement suggestions. You can:
1. Analyze common patterns or unique features of these expressions
2. Identify potential optimization opportunities
3. Propose new expression combination methods

Please ensure your suggestions maintain the basic structure of expressions while trying to improve their performance.
"""

    SUBSEQUENT_ROUND = """
I've noticed the following well-performing individuals in the current population:

{top_individuals}

In the previous round of interaction, your suggestions and their effects were as follows:

{previous_results}

Based on the above information, especially the effects of previous suggestions, please propose new improvement plans. You can:
1. Learn from successful suggestions
2. Analyze reasons for failed suggestions and avoid similar issues
3. Incorporate characteristics of excellent individuals in the current population
4. Provide maximum of 3 expressions in suggestions
5. Keep expressions concise, avoid overly complex expressions, minimize operators and variable symbols.
"""

    ERROR_FEEDBACK = """
I noticed your last response had formatting issues. Please strictly follow this JSON format:

{format}

Error message: {error}

Please regenerate your suggestions, ensuring:
1. Response must be valid JSON format
2. Include all required fields
3. Expression field must use correct operators and variables
4. Don't wrap json data in markdown code blocks, just provide raw json data

"""

    @staticmethod
    def format_top_individuals(individuals: List[Dict]) -> str:
        """Format top individuals information"""
        return "\n".join(
            f"Individual {i+1}:\n"
            f"- Expression: {ind['expression']}\n"
            f"- Fitness: {ind['fitness']:.4f}"
            for i, ind in enumerate(individuals)
        )

    @staticmethod
    def format_previous_results(suggestions: Dict) -> str:
        """Format results from previous round suggestions"""
        result = []
        for sugg in suggestions['suggestions']:
            if sugg['status'] == 'success':
                result.append(
                    f"Suggested Expression: {sugg['expression']}\n"
                    f"- Actual Fitness: {sugg['fitness']:.4f}\n"
                    f"- Improvement Reason: {sugg['reason']}"
                )
            else:
                result.append(
                    f"Suggested Expression: {sugg['expression']}\n"
                    f"- Evaluation Failed: {sugg['error']}\n"
                    f"- Improvement Reason: {sugg['reason']}"
                )
        return "\n\n".join(result)

    @staticmethod
    def create_system_prompt(labels: List[str], operators: List[str], format_example: Dict) -> str:
        """Create system prompt"""
        return PromptTemplates.SYSTEM.format(
            labels=labels,
            operators=operators,
            format=format_example
        )

def process_llm_response(llm_client, model_name, dialogs, queue_snd, max_retries=3):
    """Process LLM response with retry mechanism"""
    retries = 0
    while retries < max_retries:
        try:
            # Call LLM
            results = llm_client.chat.completions.create(
                model=model_name,
                messages=dialogs
            )
            model_response = results.choices[0].message.content.strip()
            
            # Record response
            dialogs.append({"role": "assistant", "content": model_response})
            cprint(f"LLM Response (Attempt {retries + 1}/{max_retries}): {model_response}", 'c')
            
            try:
                # Try to parse JSON directly
                suggestion_payload = json.loads(model_response)
            except json.JSONDecodeError:
                # If failed, try replacing single quotes with double quotes and parse again
                try:
                    import ast
                    suggestion_payload = ast.literal_eval(model_response)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Cannot parse response format: {e}")
            
            # Validate response format
            if 'suggestions' not in suggestion_payload:
                raise ValueError("Response missing 'suggestions' field")
            
            for suggestion in suggestion_payload['suggestions']:
                if 'expression' not in suggestion or 'reason' not in suggestion:
                    raise ValueError("Suggestion missing required fields 'expression' or 'reason'")
            
            # Construct suggestion message and send
            suggestion_msg = Message(
                msg_type=MessageType.SUGGESTION,
                payload=suggestion_payload
            )
            queue_snd.put(suggestion_msg.serialize())
            return True
            
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"Response format error: {str(e)}"
            retries += 1
        except Exception as e:
            error_msg = f"Unknown error: {str(e)}"
            retries += 1
            
        if retries < max_retries:
            # Add error feedback prompt
            error_prompt = PromptTemplates.ERROR_FEEDBACK.format(
                format=json.dumps(resp_format, indent=2, ensure_ascii=False),
                error=error_msg
            )
            dialogs.append({"role": "user", "content": error_prompt})
            cprint(f"Sending error feedback: {error_prompt}", 'y')
        else:
            # Maximum retries reached, send error message
            error_msg = Message(
                msg_type=MessageType.ERROR,
                payload={
                    "error": error_msg,
                    "retries": retries
                }
            )
            queue_snd.put(error_msg.serialize())
            return False

def llama_main(
    queue_recv: Queue,
    queue_snd: Queue,
    llm_client: openai.OpenAI,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
):
    out_f = open("output.txt", "w")
    
    init_dialogs = []
    dialogs = []
    
    while True:
        try:
            data = queue_recv.get(timeout=1.0)  # Add timeout
        except queue.Empty:
            continue
        
        try:
            msg = Message.deserialize(data)
        except Exception as e:
            cprint(f"Failed to deserialize message: {e}", "r")
            continue
        
        if msg.msg_type == MessageType.INIT:
            labels = msg.payload.get("labels", [])
            operators = msg.payload.get("operators", [])
            init_dialogs_setting = []
            system_prompt = PromptTemplates.create_system_prompt(
                labels=labels,
                operators=operators,
                format_example=json.dumps(resp_format, indent=2, ensure_ascii=False)
            )
            init_dialogs_setting.append({"role": "system", "content": system_prompt})
            init_dialogs = init_dialogs_setting
            cprint(f"System initialized: Computable variables: {labels}; Supported operators: {operators}", 'm')
            cprint("LLM initialization complete, waiting for messages...", 'm')
        elif msg.msg_type == MessageType.COMMAND:
            command = msg.payload.get("command", "")
            if command == "exit":
                cprint("Received exit command, ending conversation.", 'r')
                out_f.write("\n============Conversation ended.============\n")
                out_f.flush()
                break
            else:
                cprint(f"Received unknown command: {command}", 'y')
        elif msg.msg_type == MessageType.EVOLUTION_UPDATE:
            top_individuals = msg.payload.get("top_individuals", [])
            previous_suggestions = msg.payload.get("previous_suggestions", None)
            
            # Format top individuals information
            formatted_individuals = PromptTemplates.format_top_individuals(top_individuals)
            
            # Choose template based on whether there are previous suggestions
            if previous_suggestions is None:
                prompt = PromptTemplates.FIRST_ROUND.format(
                    top_individuals=formatted_individuals
                )
            else:
                formatted_results = PromptTemplates.format_previous_results(previous_suggestions)
                prompt = PromptTemplates.SUBSEQUENT_ROUND.format(
                    top_individuals=formatted_individuals,
                    previous_results=formatted_results
                )
            
            prompt += f"\n\nPlease provide your suggestions in the following JSON format (give json data directly, don't wrap in code blocks):\n{json.dumps(resp_format, indent=2, ensure_ascii=False)}"
            
            dialogs.append({"role": "user", "content": prompt})
            cprint(f"Sending prompt to LLM: {prompt}\n", 'y')
            
            # Process LLM response (with retry mechanism)
            if not process_llm_response(llm_client, model_name, dialogs, queue_snd):
                cprint("LLM response processing failed", 'r')
                # Consider adding retry or recovery strategies
        elif msg.msg_type == MessageType.THRESHOLD_START:
            threshold = msg.payload.get("threshold", None)
            train_size = msg.payload.get("train_size", None)
            test_size = msg.payload.get("test_size", None)
            info = f"Received threshold experiment start message: Threshold = {threshold}, Training set size = {train_size}, Test set size = {test_size}"
            cprint(info, 'm')
            dialogs = copy.deepcopy(init_dialogs)
            
            out_f.write(info + "\n")
        else:
            cprint(f"Received unknown message type: {msg.msg_type}", 'y')
        out_f.flush()
    
    # Output conversation record
    for msg in dialogs:
        # print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        # print("==================================\n")
        out_f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
        out_f.write("==================================\n")
    
    out_f.close()
          
def main():
    pass

# if __name__ == "__main__":
#     fire.Fire(main)