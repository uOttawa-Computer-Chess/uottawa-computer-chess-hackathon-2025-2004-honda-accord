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

# ---------------- King safety tuning ----------------
# Pawn shield: missing a pawn on rank-2 (or rank-7 for Black) hurts most,
# having one on rank-3 (rank-6) is a partial consolation.
_PAWN_SHIELD_RANK2 = 30
_PAWN_SHIELD_RANK3 = 15
_PAWN_SHIELD_MISSING_BOTH = 35  # extra if a file next to king has no friendly pawn on r2/r3

# (Semi-)open files next to king
_SEMI_OPEN_NEAR_KING = 12    # no friendly pawn on that file
_OPEN_NEAR_KING = 10         # and also no enemy pawn on that file
_CLEAR_FILE_ROOK_QUEEN = 30  # enemy rook/queen on same file with no blockers to king

# Enemy attacker weights into king zone (unique attackers across the zone)
_ATT_W = {
    chess.PAWN:   9,
    chess.KNIGHT: 13,
    chess.BISHOP: 13,
    chess.ROOK:   20,
    chess.QUEEN:  26,
}

# Endgame dampening of king-safety when heavy pieces are off the board
_EG_DAMP_NO_QUEEN = 0.35   # if the opponent has no queen
_EG_DAMP_WITH_QUEEN = 0.55 # if the opponent still has a queen


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


# ---------------- King safety helpers ----------------
def _files_around_king(board: chess.Board, color: chess.Color) -> list[int]:
    ksq = board.king(color)
    if ksq is None:
        return [3, 4, 5]  # fallback
    kf = chess.square_file(ksq)
    return [f for f in (kf - 1, kf, kf + 1) if 0 <= f <= 7]


def _pawn_shield_penalty(board: chess.Board, color: chess.Color) -> int:
    """Penalty (centipawns) for missing friendly pawn shield in files next to king."""
    files = _files_around_king(board, color)
    # White wants pawns on ranks 2/3; Black wants on ranks 7/6.
    if color == chess.WHITE:
        r2, r3 = 1, 2
    else:
        r2, r3 = 6, 5

    pawns = board.pieces(chess.PAWN, color)
    penalty = 0
    for f in files:
        has_r2 = False
        has_r3 = False
        # rank bounds are guaranteed (1/2 for white, 6/5 for black)
        sq_r2 = chess.square(f, r2)
        sq_r3 = chess.square(f, r3)
        if sq_r2 in pawns:
            has_r2 = True
        if sq_r3 in pawns:
            has_r3 = True

        if not has_r2:
            penalty += _PAWN_SHIELD_RANK2
        if not has_r3:
            penalty += _PAWN_SHIELD_RANK3
        if (not has_r2) and (not has_r3):
            penalty += _PAWN_SHIELD_MISSING_BOTH
    return penalty


def _semi_open_files_penalty(board: chess.Board, color: chess.Color) -> int:
    """Penalty for (semi-)open files near king, with extra if enemy R/Q has a clear file to king."""
    enemy = not color
    files = _files_around_king(board, color)

    friendly_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, enemy)
    enemy_heavies = list(board.pieces(chess.ROOK, enemy)) + list(board.pieces(chess.QUEEN, enemy))
    ksq = board.king(color)
    if ksq is None:
        return 0
    kf = chess.square_file(ksq)
    kr = chess.square_rank(ksq)

    def _file_has_pawn(squares, file_idx) -> bool:
        for sq in squares:
            if chess.square_file(sq) == file_idx:
                return True
        return False

    def _clear_line_between_file(piece_sq: int) -> bool:
        """True if no pieces between enemy heavy and king on the same file."""
        if chess.square_file(piece_sq) != kf:
            return False
        r0 = chess.square_rank(piece_sq)
        step = 1 if r0 < kr else -1
        for r in range(r0 + step, kr, step):
            if board.piece_at(chess.square(kf, r)) is not None:
                return False
        return True

    penalty = 0
    for f in files:
        friendly_on_file = _file_has_pawn(friendly_pawns, f)
        enemy_on_file = _file_has_pawn(enemy_pawns, f)
        if not friendly_on_file:
            penalty += _SEMI_OPEN_NEAR_KING
            if not enemy_on_file:
                penalty += _OPEN_NEAR_KING

        # Extra if an enemy rook/queen sits on king file with no blockers
        if f == kf:
            for sq in enemy_heavies:
                if _clear_line_between_file(sq):
                    penalty += _CLEAR_FILE_ROOK_QUEEN
                    break
    return penalty


def _king_zone_attack_penalty(board: chess.Board, color: chess.Color) -> int:
    """
    Penalize number/strength of *unique* enemy attackers on king square and adjacent ring.
    """
    enemy = not color
    ksq = board.king(color)
    if ksq is None:
        return 0

    # King zone: current square + 8 adjacent (king moves). board.attacks(ksq) works (king moves).
    zone = set(board.attacks(ksq))
    zone.add(ksq)

    # Collect unique enemy attackers across the zone
    attackers = set()
    for sq in zone:
        for a in board.attackers(enemy, sq):
            attackers.add(a)

    penalty_units = 0
    for a in attackers:
        p = board.piece_at(a)
        if p is None:
            continue
        w = _ATT_W.get(p.piece_type, 0)
        penalty_units += w

    return penalty_units  # scaled later into MG/EG


def _king_safety(board: chess.Board) -> tuple[int, int]:
    """
    Return (mg_delta, eg_delta) to add to (white - black) score.
    Positive delta favors White; negative favors Black.
    We compute per-color penalties and return (pen_black - pen_white) for each phase.
    """
    # White penalties
    w_shield = _pawn_shield_penalty(board, chess.WHITE)
    w_open = _semi_open_files_penalty(board, chess.WHITE)
    w_att = _king_zone_attack_penalty(board, chess.WHITE)

    # Black penalties
    b_shield = _pawn_shield_penalty(board, chess.BLACK)
    b_open = _semi_open_files_penalty(board, chess.BLACK)
    b_att = _king_zone_attack_penalty(board, chess.BLACK)

    # Weight king zone pressure stronger than pawn/lines
    w_mg_pen = w_shield + w_open + int(1.2 * w_att)
    b_mg_pen = b_shield + b_open + int(1.2 * b_att)

    # Endgame dampening depending on whether opponent still has a queen
    w_enemy_has_queen = len(board.pieces(chess.QUEEN, chess.BLACK)) > 0
    b_enemy_has_queen = len(board.pieces(chess.QUEEN, chess.WHITE)) > 0
    w_eg_factor = _EG_DAMP_WITH_QUEEN if w_enemy_has_queen else _EG_DAMP_NO_QUEEN
    b_eg_factor = _EG_DAMP_WITH_QUEEN if b_enemy_has_queen else _EG_DAMP_NO_QUEEN

    w_eg_pen = int(w_eg_factor * w_mg_pen)
    b_eg_pen = int(b_eg_factor * b_mg_pen)

    # Convert to (white - black): subtract white penalties, add black penalties
    mg_delta = b_mg_pen - w_mg_pen
    eg_delta = b_eg_pen - w_eg_pen
    return mg_delta, eg_delta


def evaluate(board: chess.Board, table) -> int:
    """
    White-centric tapered evaluation (centipawns).
    Positive => good for White; Negative => good for Black.
    Uses material + PST (MG/EG) with game-phase blending, bishop-pair, tempo,
    and a king-safety term to discourage exposed kings.
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

    # --- King safety (white - black delta) ---
    ks_mg, ks_eg = _king_safety(board)
    mg_score += ks_mg
    eg_score += ks_eg

    # Safe, clamped tapered blend
    phase = max(0, min(_TOTAL_PHASE, int(phase)))
    mg_score = int(mg_score)
    eg_score = int(eg_score)

    blended = (mg_score * phase + eg_score * (_TOTAL_PHASE - phase)) // _TOTAL_PHASE
    table.storePosition(board, blended)
    return int(blended)
