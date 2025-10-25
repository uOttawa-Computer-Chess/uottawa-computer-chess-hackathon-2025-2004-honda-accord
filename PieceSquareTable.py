import numpy as np
import chess

# -------------------- White PSTs (middlegame) --------------------

def getWhitePawnPst():
    return np.array([
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0
    ], dtype=np.int16)

def getWhiteKnightPst():
    return np.array([
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50
    ], dtype=np.int16)

def getWhiteBishopPst():
    return np.array([
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10,-10,-10,-10,-10,-20
    ], dtype=np.int16)

def getWhiteRookPst():
    return np.array([
         0,  0,  5, 10, 10,  5,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ], dtype=np.int16)

def getWhiteQueenPst():
    return np.array([
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -10,  5,  5,  5,  5,  5,  0,-10,
         0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
       -10,  0,  5,  5,  5,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20
    ], dtype=np.int16)

# --- King PSTs split into MG and EG ---

def getWhiteKingMgPst():
    """Middlegame: favor castling / edge safety (your original king PST)."""
    return np.array([
         20, 30, 10,  0,  0, 10, 30, 20,
         20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30
    ], dtype=np.int16)

def getWhiteKingEgPst():
    """Endgame: encourage central/active king."""
    return np.array([
        -50,-30,-30,-30,-30,-30,-30,-50,
        -30,-10,  0,  0,  0,  0,-10,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,-10,  0,  0,  0,  0,-10,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ], dtype=np.int16)

# -------------------- Black helpers --------------------

def flipPst(pst: np.ndarray) -> np.ndarray:
    """Return the black piece-square table by flipping the white one (rank+file mirror)."""
    return np.flip(pst)

def getBlackPawnPst():   return flipPst(getWhitePawnPst())
def getBlackKnightPst(): return flipPst(getWhiteKnightPst())
def getBlackBishopPst(): return flipPst(getWhiteBishopPst())
def getBlackRookPst():   return flipPst(getWhiteRookPst())
def getBlackQueenPst():  return flipPst(getWhiteQueenPst())
def getBlackKingMgPst(): return flipPst(getWhiteKingMgPst())
def getBlackKingEgPst(): return flipPst(getWhiteKingEgPst())

# Back-compat old names (map to MG by default, in case other code still imports them)
def getWhiteKingPst(): return getWhiteKingMgPst()
def getBlackKingPst(): return getBlackKingMgPst()

# -------------------- MG / EG dictionaries --------------------

MG_PST = {
    chess.PAWN:   getWhitePawnPst(),
    chess.KNIGHT: getWhiteKnightPst(),
    chess.BISHOP: getWhiteBishopPst(),
    chess.ROOK:   getWhiteRookPst(),
    chess.QUEEN:  getWhiteQueenPst(),
    chess.KING:   getWhiteKingMgPst(),   # MG king
}

# EG reuses MG for all pieces except the king (which has its own EG PST)
EG_PST = {
    chess.PAWN:   getWhitePawnPst().copy(),
    chess.KNIGHT: getWhiteKnightPst().copy(),
    chess.BISHOP: getWhiteBishopPst().copy(),
    chess.ROOK:   getWhiteRookPst().copy(),
    chess.QUEEN:  getWhiteQueenPst().copy(),
    chess.KING:   getWhiteKingEgPst(),   # EG king
}

# (Optional) sanity checks
assert all(len(v) == 64 for v in MG_PST.values()), "MG_PST arrays must be length 64"
assert all(len(v) == 64 for v in EG_PST.values()), "EG_PST arrays must be length 64"
