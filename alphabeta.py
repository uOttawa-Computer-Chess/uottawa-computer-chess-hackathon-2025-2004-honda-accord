# alphabeta.py
import chess
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from evaluation import evaluate          # <- your existing evaluator
import TranspositionTable                # <- your simple eval cache (we will reuse the instance you pass)

INF = 10 ** 12
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# Light piece values for capture ordering (MVV-LVA)
_PV = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:     0,
}

@dataclass
class _Entry:
    depth: int
    flag: int
    score: int
    move: Optional[chess.Move]

def _tt_key(b: chess.Board) -> Tuple[str, bool, int, Optional[int]]:
    """
    Version-agnostic, compact position key:
    - board_fen(): piece placement only
    - turn
    - castling_rights bitmask
    - ep_square (or None)
    This avoids dependence on python-chess zobrist availability and
    ignores halfmove/fullmove counters so transpositions collide correctly.
    """
    return (b.board_fen(), b.turn, b.castling_rights, b.ep_square)

def _is_quiet(b: chess.Board, m: chess.Move) -> bool:
    return (not b.is_capture(m)) and (m.promotion is None)

def _mvv_lva(b: chess.Board, m: chess.Move) -> int:
    if not b.is_capture(m):
        return 0
    victim = b.piece_at(m.to_square)
    if victim is None and b.is_en_passant(m):
        v_val = _PV[chess.PAWN]
    else:
        v_val = _PV[victim.piece_type] if victim else 0
    attacker = b.piece_at(m.from_square)
    a_val = _PV[attacker.piece_type] if attacker else 0
    return 10_000 * v_val - a_val  # larger is better

def _order_moves(b: chess.Board, moves, tt_move: Optional[chess.Move]) -> list:
    """Order: TT best → captures (MVV-LVA) → checking moves → others."""
    def _score(m: chess.Move) -> int:
        s = 0
        if tt_move is not None and m == tt_move:
            s += 20_000_000
        if b.is_capture(m):
            s += 6_000_000 + _mvv_lva(b, m)
        try:
            if b.gives_check(m):
                s += 300_000
        except Exception:
            pass
        if m.promotion:
            s += 8_000_000 + _PV.get(m.promotion, 0)
        return s
    ml = list(moves)
    ml.sort(key=_score, reverse=True)
    return ml

# ------------------------ Negamax + Alpha-Beta + TT ------------------------

def negamax(board: chess.Board,
            depth: int,
            alpha: int,
            beta: int,
            table: TranspositionTable.TranspositionTable,
            search_tt: Optional[Dict[Tuple[str, bool, int, Optional[int]], _Entry]] = None) -> int:
    """
    Negamax with alpha-beta pruning.
    Returns a score from the perspective of the side to move (White-positive).
    - `table` is your existing TranspositionTable instance used by evaluate().
    - `search_tt` is an optional dict used only for search bounds/PV (separate from `table`).
    """
    if search_tt is None:
        search_tt = {}

    key = _tt_key(board)
    orig_alpha, orig_beta = alpha, beta

    # Probe search TT
    tte = search_tt.get(key)
    tt_move = None
    if tte is not None and tte.depth >= depth:
        # Bound logic
        if tte.flag == TT_EXACT:
            return tte.score
        if tte.flag == TT_LOWER and tte.score > alpha:
            alpha = tte.score
        elif tte.flag == TT_UPPER and tte.score < beta:
            beta = tte.score
        if alpha >= beta:
            return tte.score
        tt_move = tte.move

    # Terminal / leaf
    if depth == 0 or board.is_game_over():
        color = 1 if board.turn == chess.WHITE else -1
        return color * evaluate(board, table)

    best_score = -INF
    best_move = None

    # Order moves
    for m in _order_moves(board, board.legal_moves, tt_move):
        board.push(m)
        score = -negamax(board, depth - 1, -beta, -alpha, table, search_tt)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = m
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break  # beta cutoff

    # Store to search TT
    flag = TT_EXACT
    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= orig_beta:
        flag = TT_LOWER
    search_tt[key] = _Entry(depth=depth, flag=flag, score=best_score, move=best_move)

    return best_score

# ------------------------ Root helper: pick best move ------------------------

def pick_move(board: chess.Board,
              depth: int,
              table: TranspositionTable.TranspositionTable,
              allowed_moves: Optional[list] = None,
              prev_best: Optional[chess.Board] = None) -> chess.Move:
              
    """
    Root driver for negamax. Respects an optional `allowed_moves` list (e.g., Lichess root_moves).
    Returns the best move at the requested depth.
    """
    legal = list(board.legal_moves)
    if allowed_moves:
        allow_uci = {m.uci() for m in allowed_moves}
        legal = [m for m in legal if m.uci() in allow_uci]

    if not legal:
        return chess.Move.null()

    search_tt: Dict[Tuple[str, bool, int, Optional[int]], _Entry] = {}
    best_move = legal[0]
    best_val = -INF
    
    # use prev_best
    tt_move = prev_best if prev_best in legal else None

    for m in _order_moves(board, legal, tt_move):
        board.push(m)
        val = -negamax(board, depth - 1, -INF, INF, table, search_tt)
        board.pop()

        if val > best_val:
            best_val = val
            best_move = m

    return best_move
