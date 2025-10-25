import time
from typing import Optional
import alphabeta
import chess

def calculate_time_allocation(remaining_time, increment, moves_to_go: int = 25) -> float:
   
    # Basic formula: use remaining_time / moves_to_go + part_of_increment
    base_time = remaining_time / moves_to_go
    increment_bonus = increment * 0.7  # Use 70% of increment
    
    # Be more conservative when low on time
    if remaining_time < 30.0:
        base_time = remaining_time / (moves_to_go * 2)  # Use half as much per move
    elif remaining_time < 60.0:
        base_time = remaining_time / (moves_to_go * 1.5)
    
    total_time = base_time + increment_bonus
    
    # Never use more than 1/3 of remaining time
    total_time = min(total_time, remaining_time * 0.33)
    
    # Always leave at least 0.5 seconds for emergency moves
    total_time = min(total_time, remaining_time - 0.5)
    
    return max(0.1, total_time)  





# now for the big boy algo

def iterativeDeepen(board, table, allowed_moves: Optional[list] = None, remaining_time: float = 60.0, increment: float = 0.0, max_depth: Optional[int] = None ):
    start_time = time.time()
    budget = calculate_time_allocation(remaining_time, increment)
 
    best_move = None
    depth = 1
    
    #main loop 
    while True:
        #check conditions
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Time check 
        if elapsed > budget * 0.8:  
            break
            
        if max_depth and depth > max_depth:
            break
        
        # Safety: don't search beyond reasonable depth
        if depth > 50:
            break
        
        
        try:
            current_best = alphabeta.pick_move(board, depth, table, allowed_moves, best_move)
            
            if current_best != chess.Move.null():
                best_move = current_best
                
            
            # Estimate time for next depth (very rough)
            if elapsed > 0:
                time_per_depth = elapsed / depth
                estimated_next_time = elapsed + time_per_depth * 2  # Next depth might take 2x longer
                if estimated_next_time > budget * 0.9:
                    break
            
        except Exception as e:
            print(f"Error at depth {depth}: {e}")
            break
            
        depth += 1
        
    print("depth", depth)
    return best_move if best_move else chess.Move.null()
