import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Set, Dict, Any

def calculate_metrics(oracle_file: str, agent_file: str, tool_weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Calculates various metrics for tool selection based on oracle and agent outputs.

    Args:
        oracle_file (str): Path to the JSON file with oracle tool choices.
        agent_file (str): Path to the JSON file with agent tool choices.
        tool_weights (Dict[str, float], optional): A dictionary mapping tool names to weights
                                                   for the Weighted Intersection metric.
                                                   If None, all tools have a weight of 1.

    Returns:
        Dict[str, Any]: A dictionary containing the calculated metrics.
    """
    if tool_weights is None:
        tool_weights = {} # Default to empty dict, meaning default weight of 1.0 for all tools

    try:
        with open(oracle_file, 'r') as f:
            oracle_data = json.load(f)
        with open(agent_file, 'r') as f:
            agent_data = json.load(f)
    except FileNotFoundError as e:
        return {"error": f"File not found: {e.filename}", "status": "error"}
    except json.JSONDecodeError as e:
        return {"error": f"Error decoding JSON in file: {e}", "status": "error"}

    if len(oracle_data) != len(agent_data):
        # Try to match by query_id if lengths are different due to missing items
        print("Warning: Oracle and Agent data files have different numbers of entries. Attempting to match by query_id.")

    # Convert to dictionaries for easier lookup by query_id
    oracle_map = {item['query_id']: {"tools": set(item['selected_tools']), "text": item["query_text"]} for item in oracle_data}
    agent_map = {item['query_id']: {"tools": set(item['selected_tools']), "text": item["query_text"]} for item in agent_data}
    
    # Use query_ids present in both, or all from oracle if lengths differ initially
    common_query_ids = set(oracle_map.keys()) & set(agent_map.keys())
    if not common_query_ids and oracle_map: # If no common IDs but oracle has data, use oracle IDs
        print("Warning: No common query_ids found. Reporting based on oracle_data entries if agent data is missing for them.")
        query_ids_to_process = sorted(list(oracle_map.keys()))
    elif not common_query_ids and not oracle_map:
         return {"error": "No queries found in the data files.", "status": "error"}
    else:
        query_ids_to_process = sorted(list(common_query_ids))


    num_processed_queries = len(query_ids_to_process)
    if num_processed_queries == 0:
        return {"error": "No common queries found to process between oracle and agent data.", "status": "error"}

    simple_intersection_hits = 0
    exact_match_hits = 0
    agent_subset_correct_hits = 0
    
    total_weighted_oracle_score_sum = 0.0
    total_weighted_intersection_score_sum = 0.0

    detailed_results = []

    for query_id in query_ids_to_process:
        oracle_tools_set = oracle_map[query_id]["tools"]
        agent_item = agent_map.get(query_id) # Agent might be missing this query_id
        
        agent_tools_set = set() # Default to empty if agent data is missing for this query
        if agent_item:
            agent_tools_set = agent_item["tools"]
            
        query_text = oracle_map[query_id]["text"]

        # --- 1. Simple Intersection ---
        # A hit if they share at least one tool, OR if both correctly selected NO tools.
        is_simple_hit = False
        if not oracle_tools_set and not agent_tools_set: # Both correct NO TOOL
            is_simple_hit = True
        elif oracle_tools_set and agent_tools_set and bool(oracle_tools_set & agent_tools_set): # Common tool
            is_simple_hit = True
        # All other cases (one empty, other not; or both non-empty but no intersection) are misses.

        if is_simple_hit:
            simple_intersection_hits += 1

        # --- 2. Exact Match ---
        is_exact_hit = (oracle_tools_set == agent_tools_set)
        if is_exact_hit:
            exact_match_hits += 1

        # --- 3. Agent is Subset/Correct ---
        # Agent used only correct tools (subset of oracle's) or correctly used no tools.
        is_subset_hit = False
        if not oracle_tools_set: # Oracle: NO TOOL
            if not agent_tools_set: # Agent: NO TOOL (Correct)
                is_subset_hit = True
        elif agent_tools_set: # Oracle: TOOL(s), Agent: TOOL(s)
            if agent_tools_set.issubset(oracle_tools_set): # Agent tools are a subset of (or equal to) oracle tools
                is_subset_hit = True
        # If oracle expects tools and agent uses none, it's not a subset hit.
        # If agent uses tools not in oracle's list, it's not a subset hit.

        if is_subset_hit:
            agent_subset_correct_hits += 1
            
        # --- 4. Weighted Intersection ---
        # Default weight of 1.0 for any tool not in tool_weights
        current_max_weighted_score_for_query = sum(tool_weights.get(tool, 1.0) for tool in oracle_tools_set)
        current_weighted_intersection_for_query = sum(tool_weights.get(tool, 1.0) for tool in (oracle_tools_set & agent_tools_set))
        
        # Handle "NO TOOL" cases for weighted score:
        # If oracle expects NO tools, and agent also uses NO tools, this is a "perfect score" for this query.
        # We can define perfect score for "NO TOOL" as 1.0 for this query's contribution.
        if not oracle_tools_set:
            if not agent_tools_set: # Agent correctly used no tools
                current_weighted_intersection_for_query = 1.0
                current_max_weighted_score_for_query = 1.0
            else: # Agent used tools when none were expected (penalty)
                current_weighted_intersection_for_query = 0.0
                current_max_weighted_score_for_query = 1.0 # Max was 1 (for doing nothing)
        
        # Ensure max score is not zero if oracle expects tools, to avoid division by zero later
        # This case is mostly for when oracle_tools_set is empty, handled above.
        # If oracle_tools_set is not empty, current_max_weighted_score_for_query will be > 0 (assuming weights >= 0).
        # If all weights are 0 for expected tools, this still needs care.
        # For simplicity, if oracle_tools_set is non-empty and all its tools have 0 weight (unlikely),
        # then max score is 0, and intersection score will also be 0. Ratio would be undefined or 0.
        # The current logic handles the "NO TOOL" case correctly by setting max to 1.

        total_weighted_oracle_score_sum += current_max_weighted_score_for_query
        total_weighted_intersection_score_sum += current_weighted_intersection_for_query
        
        detailed_results.append({
            "query_id": query_id,
            "query_text": query_text,
            "oracle_tools": sorted(list(oracle_tools_set)),
            "agent_tools": sorted(list(agent_tools_set)),
            "simple_hit": is_simple_hit,
            "exact_hit": is_exact_hit,
            "subset_hit": is_subset_hit,
            "weighted_score_achieved": current_weighted_intersection_for_query,
            "max_weighted_score_possible": current_max_weighted_score_for_query
        })


    metrics = {
        "status": "success",
        "total_queries_processed": num_processed_queries,
        "simple_intersection_hit_rate": (simple_intersection_hits / num_processed_queries) * 100 if num_processed_queries > 0 else 0,
        "exact_match_hit_rate": (exact_match_hits / num_processed_queries) * 100 if num_processed_queries > 0 else 0,
        "agent_subset_correct_hit_rate": (agent_subset_correct_hits / num_processed_queries) * 100 if num_processed_queries > 0 else 0,
        "weighted_intersection_score_percentage": (total_weighted_intersection_score_sum / total_weighted_oracle_score_sum) * 100 if total_weighted_oracle_score_sum > 0 else 0,
        "raw_counts": {
            "simple_intersection_hits": simple_intersection_hits,
            "exact_match_hits": exact_match_hits,
            "agent_subset_correct_hits": agent_subset_correct_hits,
            "total_weighted_intersection_score_achieved": total_weighted_intersection_score_sum,
            "total_max_weighted_oracle_score_possible": total_weighted_oracle_score_sum
        },
        "detailed_results": detailed_results
    }

    return metrics

def plot_metrics_comparison(metrics_results: Dict[str, Any], output_filename="tool_selection_metrics_plot.png"):
    """
    Generates and saves a bar chart comparing the hit rates.
    """
    if metrics_results.get("status") != "success":
        print(f"Cannot generate plot due to error in metrics calculation: {metrics_results.get('error')}")
        return

    labels = [
        'Simple Intersection\nHit Rate (%)', 
        'Exact Match\nHit Rate (%)', 
        'Agent Subset Correct\nHit Rate (%)', 
        'Weighted Intersection\nScore (%)'
    ]
    scores = [
        metrics_results["simple_intersection_hit_rate"],
        metrics_results["exact_match_hit_rate"],
        metrics_results["agent_subset_correct_hit_rate"],
        metrics_results["weighted_intersection_score_percentage"]
    ]

    x = np.arange(len(labels))
    width = 0.5  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects = ax.bar(x, scores, width, label='Agent Performance', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])

    ax.set_ylabel('Scores / Hit Rate (%)')
    ax.set_title('Tool Selection Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105) # y-axis from 0 to 105 for percentage

    # Add text labels on top of bars
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(output_filename)
    print(f"Metrics comparison plot saved as {output_filename}")
    # plt.show() # Uncomment to display plot if running in an environment that supports it

if __name__ == "__main__":


    oracle_filename = "/workspaces/Aviation_project/evaluation/oracle_gemini_tools.json"
    agent_filename = "/workspaces/Aviation_project/evaluation/agent_tools.json"


    print("\nCalculating metrics with custom tool weights...")
    results = calculate_metrics(oracle_filename, agent_filename)
    
    if results.get("status") == "success":
        print(json.dumps(results, indent=2, ensure_ascii=False))
        plot_metrics_comparison(results, output_filename="tool_selection_metrics_plot.png")
    else:
        print(f"Error during metric calculation: {results.get('error')}")