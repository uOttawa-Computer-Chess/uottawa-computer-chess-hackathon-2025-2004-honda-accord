import time
from typing import Optional
from alphabeta import pick_move
import chess

# Configuration constants
SAFETY_MARGIN = 0.88  # Use 88% of allocated time budget
MIN_MOVE_TIME = 1.0   # Minimum time per move (seconds)
PANIC_THRESHOLD = 5.0  # Switch to panic mode below this time
PANIC_DEPTH = 4       # Maximum depth in panic mode
INCREMENT_USAGE = 0.65  # Use 65% of increment in time calculation
MAX_TIME_PER_MOVE = 0.20  # Never use more than 20% of remaining time on one move

def calculate_time_allocation(remaining_time: float, increment: float, 
                             board: chess.Board, moves_to_go: Optional[int] = None) -> float:
    """
    Calculate time to spend on this move.
    Returns a conservative time budget that ensures we don't run out of time.
    """
    # Panic mode: very little time left
    if remaining_time < PANIC_THRESHOLD:
        return max(MIN_MOVE_TIME, remaining_time * 0.25)
    
    # Estimate moves remaining in game
    if moves_to_go is None:
        piece_count = len(board.piece_map())
        move_num = board.fullmove_number
        
        if piece_count >= 28 and move_num <= 15:
            moves_to_go = 35  # Opening
        elif piece_count >= 20:
            moves_to_go = 25  # Middlegame
        elif piece_count >= 10:
            moves_to_go = 20  # Early endgame
        else:
            moves_to_go = 15  # Late endgame
    
    # Base allocation: divide remaining time by expected moves
    base_time = remaining_time / moves_to_go
    
    # Add portion of increment (we'll gain it back after the move)
    increment_bonus = increment * INCREMENT_USAGE
    
    # Game phase adjustments
    piece_count = len(board.piece_map())
    move_num = board.fullmove_number
    
    if piece_count >= 28 and move_num <= 15:
        # Opening: use less time, positions are more standard
        phase_multiplier = 0.6
    elif piece_count >= 20:
        # Middlegame: most critical phase, use more time
        phase_multiplier = 1.2
    elif piece_count >= 10:
        # Early endgame: precision matters
        phase_multiplier = 1.0
    else:
        # Late endgame: can calculate more precisely
        phase_multiplier = 0.8
    
    base_time *= phase_multiplier
    total_time = base_time + increment_bonus
    
    # Apply safety constraints
    # Never use more than MAX_TIME_PER_MOVE of remaining time
    total_time = min(total_time, remaining_time * MAX_TIME_PER_MOVE)
    
    # Always leave at least 2 seconds reserve
    total_time = min(total_time, remaining_time - 2.0)
    
    # Ensure minimum thinking time
    total_time = max(MIN_MOVE_TIME, total_time)
    
    return total_time

def iterativeDeepen(board: chess.Board, table, allowed_moves: Optional[list] = None, 
                   remaining_time: float = 60.0, increment: float = 0.0, 
                   max_depth: Optional[int] = None) -> chess.Move:
    """
    Iterative deepening search with reliable time management.
    
    Strategy:
    - Start at depth 1, increase until time budget is reached
    - Only use results from fully completed searches
    - Stop with safety margin to avoid time forfeit
    - Simple, predictable behavior
    """
    start_time = time.time()
    time_budget = calculate_time_allocation(remaining_time, increment, board)
    
    # Panic mode: do minimal search
    if remaining_time < PANIC_THRESHOLD:
        print(f"PANIC MODE: {remaining_time:.2f}s remaining")
        max_depth = PANIC_DEPTH
        time_budget = min(time_budget, remaining_time * 0.3)
    
    print(f"Time budget: {time_budget:.2f}s | Remaining: {remaining_time:.2f}s | Phase: {_get_phase_name(board)}")
    
    # Track completed depths
    completed_moves = {}  # depth -> (move, search_time)
    best_move = None
    depth = 1
    
    # Estimate maximum useful depth based on game phase
    piece_count = len(board.piece_map())
    if max_depth is None:
        if piece_count >= 28:
            max_depth = 8   # Opening
        elif piece_count >= 20:
            max_depth = 12  # Middlegame
        elif piece_count >= 10:
            max_depth = 16  # Early endgame
        else:
            max_depth = 20  # Late endgame
    
    while depth <= max_depth:
        elapsed = time.time() - start_time
        
        # Check if we should stop before starting next depth
        if elapsed >= time_budget * SAFETY_MARGIN:
            print(f"Stopping: reached time budget ({elapsed:.2f}s >= {time_budget * SAFETY_MARGIN:.2f}s)")
            break
        
        # Predict if next depth will exceed budget
        if depth > 1 and len(completed_moves) >= 2:
            last_time = completed_moves[depth - 1][1]
            prev_time = completed_moves[depth - 2][1] if depth > 2 else last_time / 2
            
            # Estimate branching factor from time growth
            if prev_time > 0.01:
                time_ratio = last_time / prev_time
                time_ratio = min(time_ratio, 3.5)  # Cap at reasonable branching factor
            else:
                time_ratio = 2.5
            
            predicted_time = elapsed + (last_time * time_ratio)
            
            # Be conservative: if prediction suggests we might exceed budget, stop
            if predicted_time > time_budget * 0.95:
                print(f"Stopping: prediction suggests next depth would exceed budget")
                print(f"  Predicted: {predicted_time:.2f}s vs Budget: {time_budget:.2f}s")
                break
        
        # Attempt search at this depth
        depth_start = time.time()
        
        try:
            # Use previous best move for move ordering
            move = pick_move(board, depth, table, allowed_moves, best_move)
            depth_time = time.time() - depth_start
            
            # Validate the move
            if move and move != chess.Move.null():
                completed_moves[depth] = (move, depth_time)
                best_move = move
                print(f"Depth {depth}: {move} ({depth_time:.2f}s)")
            else:
                print(f"Depth {depth}: returned null move, stopping")
                break
                
        except KeyboardInterrupt:
            # Allow clean interruption for testing
            print(f"Interrupted at depth {depth}")
            break
        except Exception as e:
            print(f"Error at depth {depth}: {type(e).__name__}: {str(e)[:100]}")
            # If we have at least 2 completed depths, we can stop safely
            if len(completed_moves) >= 2:
                break
            # Otherwise try to continue (might be a transient error)
            if len(completed_moves) >= 1:
                depth += 1
                continue
            else:
                # No completed searches at all - this is bad
                print("ERROR: No completed searches, using fallback")
                break
        
        depth += 1
    
    total_time = time.time() - start_time
    
    # Select best move from completed searches
    if completed_moves:
        max_completed_depth = max(completed_moves.keys())
        final_move = completed_moves[max_completed_depth][0]
        
        print(f"Completed depths: {sorted(completed_moves.keys())}")
        print(f"Using depth {max_completed_depth} move: {final_move}")
        print(f"Total time: {total_time:.2f}s / {time_budget:.2f}s ({total_time/time_budget*100:.0f}%)")
        
        return final_move
    else:
        # Emergency fallback: pick any legal move
        print("WARNING: No completed searches! Picking random legal move")
        legal_moves = list(board.legal_moves) if allowed_moves is None else allowed_moves
        if legal_moves:
            return legal_moves[0]
        else:
            return chess.Move.null()

def _get_phase_name(board: chess.Board) -> str:
    """Helper to identify game phase for logging"""
    piece_count = len(board.piece_map())
    move_num = board.fullmove_number
    
    if piece_count >= 28 and move_num <= 15:
        return "Opening"
    elif piece_count >= 20:
        return "Middlegame"
    elif piece_count >= 10:
        return "Early Endgame"
    else:
        return "Late Endgame"