# evaluation.py
import chess
import TranspositionTable
# --- Hook your PSTs here ---
from PieceSquareTable import MG_PST, EG_PST  # dicts: piece type -> np.array[64], white-oriented

MATE = 32000  # normalized mate score (centipawns)

# Base material in centipawns (can be tuned).
_MAT = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # handled by checkmate; PSTs carry king activity
}
MG_VALUES = _MAT
EG_VALUES = _MAT

# Game-phase weights (standard total = 24)
_PHASE_WEIGHT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
_TOTAL_PHASE = 24

# Small heuristics
TEMPO_BONUS = 10
BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 40

def _game_phase(board: chess.Board) -> int:
    """Return phase in [0.._TOTAL_PHASE]; higher => more middlegame."""
    phase = 0
    for pt, w in _PHASE_WEIGHT.items():
        if w == 0:
            continue
        phase += w * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    if phase > _TOTAL_PHASE:
        phase = _TOTAL_PHASE
    if phase < 0:
        phase = 0
    return phase

def _pst_score(board: chess.Board, mg: bool) -> int:
    """Piece-square table sum for both sides (white - black)."""
    PST = MG_PST if mg else EG_PST
    total = 0
    # White pieces: use square as-is
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in board.pieces(pt, chess.WHITE):
            total += int(PST[pt][sq])
    # Black pieces: mirror squares to reuse white-oriented PST and subtract
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for sq in board.pieces(pt, chess.BLACK):
            total -= int(PST[pt][chess.square_mirror(sq)])
    return int(total)

def _material(board: chess.Board, mg: bool) -> int:
    """Material sum (white - black)."""
    VALS = MG_VALUES if mg else EG_VALUES
    score = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        score += VALS[pt] * len(board.pieces(pt, chess.WHITE))
        score -= VALS[pt] * len(board.pieces(pt, chess.BLACK))
    return score

def _bishop_pair(board: chess.Board, mg: bool) -> int:
    """Bonus for having both bishops."""
    w_pair = len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2
    b_pair = len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2
    if mg:
        return (BISHOP_PAIR_MG if w_pair else 0) - (BISHOP_PAIR_MG if b_pair else 0)
    else:
        return (BISHOP_PAIR_EG if w_pair else 0) - (BISHOP_PAIR_EG if b_pair else 0)

def evaluate(board: chess.Board, table) -> int:
    """
    White-centric tapered evaluation (centipawns).
    Positive => good for White; Negative => good for Black.
    Uses material + PST (MG/EG) with game-phase blending, bishop-pair, and a small tempo bonus.
    """
    # Terminal outcomes (white-centric)
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0  # draw
        return MATE if outcome.winner is chess.WHITE else -MATE

    # Transposition-table backed cache (if provided)
    if table.exists(board):
        blended = table.lookup(board)
        return int(blended)

    # Compute phase 0..24
    phase = _game_phase(board)

    # Middlegame & Endgame components
    mg_score = 0
    eg_score = 0

    mg_score += _material(board, mg=True)
    eg_score += _material(board, mg=False)

    mg_score += _pst_score(board, mg=True)
    eg_score += _pst_score(board, mg=False)

    mg_score += _bishop_pair(board, mg=True)
    eg_score += _bishop_pair(board, mg=False)

    # Tiny tempo bonus (white-centric)
    tempo = TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS
    mg_score += tempo
    eg_score += tempo

    # Safe, clamped tapered blend (avoid NumPy overflow if any np.int sneaks in)
    phase = max(0, min(_TOTAL_PHASE, int(phase)))
    mg_score = int(mg_score)
    eg_score = int(eg_score)

    blended = (mg_score * phase + eg_score * (_TOTAL_PHASE - phase)) // _TOTAL_PHASE
    table.storePosition(board, blended)
    return int(blended)
