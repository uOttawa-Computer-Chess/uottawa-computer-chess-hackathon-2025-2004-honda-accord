# alphabeta.py
import chess
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from evaluation import evaluate         
from TranspositionTable import TranspositionTable           

# Constants
INF = 10 ** 12
MATE_SCORE = 900_000
MAX_PLY = 100

# Transposition table flags
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# Draw contempt (nudge away from easy repetition/50-move draws)
CONTEMPT = 20  # centipawns

# Move ordering scores
SCORE_TT_MOVE = 20_000_000
SCORE_PROMOTION_BASE = 8_000_000
SCORE_CAPTURE_BASE = 6_000_000
SCORE_CHECK = 300_000

# Piece values for MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
_PV = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

# Quiescence search configuration
QUIESCENCE_MAX_DEPTH = 4  # Prevent infinite quiescence search (optimized for speed)

@dataclass
class _Entry:
    depth: int
    flag: int
    score: int
    move: Optional[chess.Move]

def _tt_key(b: chess.Board) -> Tuple[str, bool, int, Optional[int]]:
    """
    Compact position key for transposition table.
    Note: Excludes halfmove clock, so 50-move rule positions may collide.
    """
    return (b.board_fen(), b.turn, b.castling_rights, b.ep_square)

def _mvv_lva(b: chess.Board, m: chess.Move) -> int:
    """
    MVV-LVA scoring for captures: prioritize capturing valuable pieces
    with less valuable attackers.
    """
    if not b.is_capture(m):
        return 0
    
    # Get victim value
    victim = b.piece_at(m.to_square)
    if victim is None and b.is_en_passant(m):
        v_val = _PV[chess.PAWN]
    else:
        v_val = _PV[victim.piece_type] if victim else 0
    
    # Get attacker value
    attacker = b.piece_at(m.from_square)
    a_val = _PV[attacker.piece_type] if attacker else 0
    
    return 10_000 * v_val - a_val

def _order_moves(b: chess.Board, moves, tt_move: Optional[chess.Move]) -> list:
    """
    Order moves for better alpha-beta pruning:
    1. TT/PV move (from previous search)
    2. Promotions (especially queen)
    3. Captures (MVV-LVA)
    4. Checks
    5. Quiet moves
    """
    def _score(m: chess.Move) -> int:
        s = 0
        
        # TT move gets highest priority
        if tt_move is not None and m == tt_move:
            s += SCORE_TT_MOVE
        
        # Promotions are often strong
        if m.promotion:
            s += SCORE_PROMOTION_BASE + _PV.get(m.promotion, 0)
        
        # Captures (MVV-LVA)
        if b.is_capture(m):
            s += SCORE_CAPTURE_BASE + _mvv_lva(b, m)
        
        # Checks (but don't let this dominate)
        if b.gives_check(m):
            s += SCORE_CHECK
        
        return s
    
    ml = list(moves)
    ml.sort(key=_score, reverse=True)
    return ml

def _is_draw(board: chess.Board) -> bool:
    """
    Fast draw detection.
    Note: is_game_over() is expensive, so check specific conditions.
    """
    return (board.is_stalemate() or 
            board.is_insufficient_material() or
            board.can_claim_fifty_moves() or
            board.can_claim_threefold_repetition())

def quiescence(board: chess.Board,
               alpha: int,
               beta: int,
               table: TranspositionTable,
               q_depth: int = 0) -> int:
    """
    Quiescence search: only search captures and checks to avoid horizon effect.
    OPTIMIZED: Only checks in check, prioritizes good captures.
    """
    # Prevent infinite quiescence
    if q_depth >= QUIESCENCE_MAX_DEPTH:
        color = 1 if board.turn == chess.WHITE else -1
        return color * evaluate(board, table)
    
    # Check for immediate terminal conditions
    in_check = board.is_check()
    
    if not in_check:
        # Check for draws (only if not in check)
        if _is_draw(board):
            return -CONTEMPT  # Slight penalty for draws
    
    # Stand-pat: can we already cause a beta cutoff?
    # Don't stand pat if we're in check (must search all moves)
    if not in_check:
        color = 1 if board.turn == chess.WHITE else -1
        stand_pat = color * evaluate(board, table)
        
        if stand_pat >= beta:
            return beta
        
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Delta pruning: if we're so far behind that even capturing
        # the queen won't help, skip quiescence
        if stand_pat < alpha - 1000:
            return alpha
    
    # Generate tactical moves
    if in_check:
        # When in check, must search all legal moves
        tactical_moves = list(board.legal_moves)
    else:
        # Only search captures and queen promotions (checks are expensive to calculate)
        tactical_moves = []
        for m in board.legal_moves:
            if board.is_capture(m):
                # SEE (Static Exchange Evaluation) pruning:
                # Skip obviously bad captures
                attacker = board.piece_at(m.from_square)
                victim = board.piece_at(m.to_square)
                if attacker and victim:
                    if _PV[attacker.piece_type] > _PV[victim.piece_type] + 200:
                        continue  # Skip bad capture (e.g., rook takes pawn)
                tactical_moves.append(m)
            elif m.promotion == chess.QUEEN:
                tactical_moves.append(m)
    
    # No tactical moves and not in check: return stand-pat
    if not tactical_moves and not in_check:
        return alpha
    
    # Check for checkmate (only if in check and no legal moves)
    if in_check and not tactical_moves:
        return -MATE_SCORE + q_depth
    
    # Order tactical moves (simple ordering, no full sort)
    if not in_check:
        # Order by MVV-LVA for captures
        tactical_moves.sort(key=lambda m: _mvv_lva(board, m), reverse=True)
    
    # Search tactical moves
    for m in tactical_moves:
        board.push(m)
        score = -quiescence(board, -beta, -alpha, table, q_depth + 1)
        board.pop()
        
        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score
    
    return alpha

def negamax(board: chess.Board,
            depth: int,
            alpha: int,
            beta: int,
            table: TranspositionTable,
            search_tt: Dict[Tuple[str, bool, int, Optional[int]], _Entry],
            ply: int = 0) -> int:
    """
    Negamax with alpha-beta pruning, transposition table, and extensions.
    Returns score from current player's perspective.
    """
    # Prevent stack overflow in deep searches
    if ply >= MAX_PLY:
        color = 1 if board.turn == chess.WHITE else -1
        return color * evaluate(board, table)
    
    # Draw contempt for claimable draws (nudge away from easy repetition/50-move)
    if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
        return -CONTEMPT
    
    # Check for immediate terminal conditions
    if board.is_checkmate():
        return -MATE_SCORE + ply  # Prefer shorter mates
    
    if _is_draw(board):
        return -CONTEMPT  # Slight penalty for draws
    
    key = _tt_key(board)
    orig_alpha, orig_beta = alpha, beta
    
    # Probe transposition table
    tte = search_tt.get(key)
    tt_move = None
    
    if tte is not None and tte.depth >= depth:
        tt_move = tte.move
        
        # Use stored bounds if depth is sufficient
        if tte.flag == TT_EXACT:
            return tte.score
        elif tte.flag == TT_LOWER:
            alpha = max(alpha, tte.score)
        elif tte.flag == TT_UPPER:
            beta = min(beta, tte.score)
        
        if alpha >= beta:
            return tte.score
    elif tte is not None:
        tt_move = tte.move  # Use move even if depth insufficient
    
    # Leaf node: enter quiescence search
    if depth <= 0:
        return quiescence(board, alpha, beta, table)
    
    best_score = -INF
    best_move = None
    moves_searched = 0
    
    # Search all legal moves
    for m in _order_moves(board, board.legal_moves, tt_move):
        board.push(m)
        
        # Check extension: if resulting position is check, extend by 1 ply
        next_depth = depth - 1
        if board.is_check():
            next_depth += 1
        
        score = -negamax(board, next_depth, -beta, -alpha, table, search_tt, ply + 1)
        board.pop()
        
        moves_searched += 1
        
        if score > best_score:
            best_score = score
            best_move = m
        
        if best_score > alpha:
            alpha = best_score
        
        # Beta cutoff
        if alpha >= beta:
            break
    
    # No legal moves means checkmate or stalemate (already handled above)
    if moves_searched == 0:
        if board.is_checkmate():
            return -MATE_SCORE + ply
        else:
            return -CONTEMPT  # Stalemate
    
    # Store to transposition table
    if best_score <= orig_alpha:
        flag = TT_UPPER  # All moves failed low
    elif best_score >= orig_beta:
        flag = TT_LOWER  # Beta cutoff occurred
    else:
        flag = TT_EXACT  # Exact score
    
    search_tt[key] = _Entry(depth=depth, flag=flag, score=best_score, move=best_move)
    
    return best_score

def pick_move(board: chess.Board,
              depth: int,
              table: TranspositionTable,
              allowed_moves: Optional[list] = None,
              prev_best: Optional[chess.Move] = None) -> chess.Move:
    """
    Root search driver. Finds the best move at the given depth.
    
    Args:
        board: Current position
        depth: Search depth
        table: Evaluation transposition table
        allowed_moves: Optional list of legal moves to consider (for Lichess bot)
        prev_best: Best move from previous iteration (for move ordering)
    
    Returns:
        Best move found
    """
    # Get legal moves
    legal = list(board.legal_moves)
    if allowed_moves:
        allow_uci = {m.uci() for m in allowed_moves}
        legal = [m for m in legal if m.uci() in allow_uci]
    
    if not legal:
        return chess.Move.null()
    
    if len(legal) == 1:
        return legal[0]  # Only one legal move
    
    # Create search transposition table
    search_tt: Dict[Tuple[str, bool, int, Optional[int]], _Entry] = {}
    
    best_move = legal[0]
    best_score = -INF
    alpha = -INF
    beta = INF
    
    # Aspiration window for deeper searches
    if prev_best and prev_best in legal and depth >= 4:
        # Get score from previous iteration
        board.push(prev_best)
        prev_score = -negamax(board, depth - 1, -INF, INF, table, search_tt, 1)
        board.pop()
        
        # Set aspiration window (±50 centipawns)
        window = 50
        alpha = prev_score - window
        beta = prev_score + window
    
    # Search all root moves
    for m in _order_moves(board, legal, prev_best):
        board.push(m)
        
        # Search with aspiration window
        score = -negamax(board, depth - 1, -beta, -alpha, table, search_tt, 1)
        
        # If we fail outside aspiration window, re-search with full window
        if (score <= alpha or score >= beta) and depth >= 4:
            score = -negamax(board, depth - 1, -INF, INF, table, search_tt, 1)
        
        board.pop()
        
        if score > best_score:
            best_score = score
            best_move = m
        
        # Update alpha for next move
        if score > alpha:
            alpha = score
    
    return best_move