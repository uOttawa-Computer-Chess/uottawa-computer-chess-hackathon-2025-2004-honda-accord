import time
from typing import Optional
from alphabeta import pick_move
import chess

# Configuration constants
SAFETY_MARGIN = 0.88   # Use 88% of allocated time budget
MIN_MOVE_TIME = 1.0    # Minimum time per move (seconds)
PANIC_THRESHOLD = 5.0  # Switch to panic mode below this time
PANIC_DEPTH = 4        # Maximum depth in panic mode
INCREMENT_USAGE = 0.65 # Use 65% of increment in time calculation
MAX_TIME_PER_MOVE = 0.20  # Never use more than 20% of remaining time on one move

# New tuning for next-depth prediction / refusal
PRED_INFLATION = 1.35       # extra safety on predicted next depth time
HARD_FACTOR = 10.0          # require at least "last_time * HARD_FACTOR" free slack to start next depth
MIN_SLACK_TO_START = 0.75   # seconds of slack we insist on before starting another depth
PV_STABILITY_LENIENT = 0.70 # if PV stable 1 depth, require next depth to fit within 70% budget
PV_STABILITY_STRICT = 0.55  # if PV stable 2+ depths, require within 55% budget
MAX_RATIO_CAP = 24.0        # upper cap on growth factor (very conservative)
BASELINE_RATIO_OPENING = 4.5
BASELINE_RATIO_MIDDLEGAME = 6.5
BASELINE_RATIO_ENDGAME = 5.5

def calculate_time_allocation(remaining_time: float, increment: float,
                             board: chess.Board, moves_to_go: Optional[int] = None) -> float:
    """Conservative time budget per move."""
    if remaining_time < PANIC_THRESHOLD:
        return max(MIN_MOVE_TIME, remaining_time * 0.25)

    # Estimate moves remaining
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

    base_time = remaining_time / moves_to_go
    increment_bonus = increment * INCREMENT_USAGE

    # phase multiplier
    piece_count = len(board.piece_map())
    move_num = board.fullmove_number
    if piece_count >= 28 and move_num <= 15:
        phase_multiplier = 0.6
    elif piece_count >= 20:
        phase_multiplier = 1.2
    elif piece_count >= 10:
        phase_multiplier = 1.0
    else:
        phase_multiplier = 0.8

    total_time = base_time * phase_multiplier + increment_bonus

    # caps & minimums
    total_time = min(total_time, remaining_time * MAX_TIME_PER_MOVE)
    total_time = min(total_time, remaining_time - 2.0)  # leave reserve
    total_time = max(MIN_MOVE_TIME, total_time)
    return total_time

def _phase_baseline_ratio(board: chess.Board) -> float:
    """Return a conservative baseline branching factor based on phase."""
    piece_count = len(board.piece_map())
    move_num = board.fullmove_number
    if piece_count >= 28 and move_num <= 15:
        return BASELINE_RATIO_OPENING
    elif piece_count >= 20:
        return BASELINE_RATIO_MIDDLEGAME
    else:
        return BASELINE_RATIO_ENDGAME

def _predict_next_depth_time(completed_times: dict[int, float], board: chess.Board) -> float:
    """
    Predict time for the next depth using the WORST observed ratio so far,
    a phase-based baseline, and an inflation safety factor.
    """
    if not completed_times:
        return 0.0
    depths = sorted(completed_times.keys())
    last_d = depths[-1]
    last_t = completed_times[last_d]

    # If only one depth, fall back to baseline
    if len(depths) == 1:
        ratio = _phase_baseline_ratio(board)
    else:
        prev_t = completed_times[depths[-2]]
        raw_ratio = last_t / max(prev_t, 1e-3)

        # If we have 3+ depths, consider the previous ratio too and use the worst
        if len(depths) >= 3:
            prev_raw = prev_t / max(completed_times[depths[-3]], 1e-3)
            raw_ratio = max(raw_ratio, prev_raw)

        # Never let prediction be too optimistic: enforce a baseline per phase
        baseline = _phase_baseline_ratio(board)
        ratio = max(baseline, raw_ratio)

    ratio = min(ratio, MAX_RATIO_CAP)
    return last_t * ratio * PRED_INFLATION

def iterativeDeepen(board: chess.Board, table, allowed_moves: Optional[list] = None,
                   remaining_time: float = 60.0, increment: float = 0.0,
                   max_depth: Optional[int] = None) -> chess.Move:
    """
    Iterative deepening with strict pre-start gating for each next depth.
    We never start a depth unless we are confident it will finish within budget.
    """
    start_time = time.time()
    time_budget = calculate_time_allocation(remaining_time, increment, board)

    # Panic mode
    if remaining_time < PANIC_THRESHOLD:
        print(f"PANIC MODE: {remaining_time:.2f}s remaining")
        max_depth = PANIC_DEPTH
        time_budget = min(time_budget, remaining_time * 0.3)

    print(f"Time budget: {time_budget:.2f}s | Remaining: {remaining_time:.2f}s | Phase: {_get_phase_name(board)}")

    completed_moves: dict[int, tuple[chess.Move, float]] = {}  # depth -> (move, time)
    completed_times: dict[int, float] = {}
    best_move = None
    prev_best_move = None
    pv_stability = 0
    depth = 1

    # Phase-based default max depth if not provided
    piece_count = len(board.piece_map())
    if max_depth is None:
        move_num = board.fullmove_number
        if piece_count >= 28 and move_num <= 15:
            max_depth = 8
        elif piece_count >= 20:
            max_depth = 12
        elif piece_count >= 10:
            max_depth = 16
        else:
            max_depth = 20

    while depth <= max_depth:
        elapsed = time.time() - start_time
        slack = time_budget * SAFETY_MARGIN - elapsed  # conservative usable time remaining

        # Stop if we're already near/over budget
        if slack <= 0:
            print(f"Stopping: reached time budget ({elapsed:.2f}s >= {time_budget * SAFETY_MARGIN:.2f}s)")
            break

        # Require some minimum slack to even consider another depth
        if slack < MIN_SLACK_TO_START:
            print(f"Stopping: not enough slack to start depth {depth} (slack {slack:.2f}s < {MIN_SLACK_TO_START:.2f}s)")
            break

        # Predict if the *next* depth is safe to start
        if completed_times:
            predicted_next = _predict_next_depth_time(completed_times, board)
            # Hard refusal based on last depth's time scaled
            last_time = completed_times[max(completed_times.keys())]
            if last_time * HARD_FACTOR > slack:
                print(f"Stopping: hard guard (last {last_time:.2f}s * {HARD_FACTOR:.1f} > slack {slack:.2f}s)")
                break

            # PV-stability heuristics (avoid spending ages to play the same move)
            if best_move is not None and prev_best_move is not None:
                pv_stability = pv_stability + 1 if best_move == prev_best_move else 0
            else:
                pv_stability = 0

            total_if_next = (time.time() - start_time) + predicted_next
            frac_of_budget = total_if_next / max(time_budget, 1e-3)

            # If PV stable 2+, be very strict
            if pv_stability >= 2 and frac_of_budget > PV_STABILITY_STRICT:
                print(f"Stopping: PV stable (≥2) and next depth predicted to exceed {PV_STABILITY_STRICT*100:.0f}% of budget "
                      f"({total_if_next:.2f}s > {time_budget*PV_STABILITY_STRICT:.2f}s)")
                break
            # If PV stable 1, be somewhat strict
            if pv_stability == 1 and frac_of_budget > PV_STABILITY_LENIENT:
                print(f"Stopping: PV stable and next depth predicted to exceed {PV_STABILITY_LENIENT*100:.0f}% of budget "
                      f"({total_if_next:.2f}s > {time_budget*PV_STABILITY_LENIENT:.2f}s)")
                break

            # General conservative check
            if total_if_next > time_budget:
                print(f"Stopping: next depth predicted to exceed budget "
                      f"({total_if_next:.2f}s > {time_budget:.2f}s) [pred {predicted_next:.2f}s]")
                break

        # Attempt search at this depth (blocking)
        depth_start = time.time()
        try:
            # NOTE: alphabeta.pick_move signature is (board, depth, table, allowed_moves)
            move = pick_move(board, depth, table, allowed_moves)
            depth_time = time.time() - depth_start

            if move and move != chess.Move.null():
                completed_moves[depth] = (move, depth_time)
                completed_times[depth] = depth_time
                prev_best_move, best_move = best_move, move
                print(f"Depth {depth}: {move} ({depth_time:.2f}s)")

                # --- NEW: early exit if depth 3 matches depth 1 or depth 2 ---
                if depth == 3:
                    d3 = move
                    same_depth = None
                    if 1 in completed_moves and completed_moves[1][0] == d3:
                        same_depth = 1
                    elif 2 in completed_moves and completed_moves[2][0] == d3:
                        same_depth = 2
                    if same_depth is not None:
                        total_time = time.time() - start_time
                        print(f"Early stop: depth 3 agrees with depth {same_depth}; playing depth 3 move.")
                        print(f"Completed depths: {sorted(completed_moves.keys())}")
                        print(f"Using depth 3 move: {d3}")
                        print(f"Total time: {total_time:.2f}s / {time_budget:.2f}s ({total_time/time_budget*100:.0f}%)")
                        return d3
                # --- END NEW ---

            else:
                print(f"Depth {depth}: returned null move, stopping")
                break

        except KeyboardInterrupt:
            print(f"Interrupted at depth {depth}")
            break
        except Exception as e:
            print(f"Error at depth {depth}: {type(e).__name__}: {str(e)[:100]}")
            if len(completed_moves) >= 1:
                # Keep the last completed result
                break
            else:
                print("ERROR: No completed searches, using fallback")
                break

        depth += 1

    total_time = time.time() - start_time

    # Choose deepest completed move
    if completed_moves:
        max_completed_depth = max(completed_moves.keys())
        final_move = completed_moves[max_completed_depth][0]
        done_depths = sorted(completed_moves.keys())
        print(f"Completed depths: {done_depths}")
        print(f"Using depth {max_completed_depth} move: {final_move}")
        print(f"Total time: {total_time:.2f}s / {time_budget:.2f}s ({total_time/time_budget*100:.0f}%)")
        return final_move
    else:
        print("WARNING: No completed searches! Picking random legal move")
        legal_moves = list(board.legal_moves) if allowed_moves is None else allowed_moves
        return legal_moves[0] if legal_moves else chess.Move.null()

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
