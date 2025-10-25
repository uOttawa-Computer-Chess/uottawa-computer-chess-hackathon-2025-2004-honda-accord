import chess
class TranspositionTable:
    """Simple transposition table wrapper around a Python dict."""

    def __init__(self):
        self.hashTable = {}

    def _key(self, position: chess.Board):
        """Return a unique and hashable key for the position."""
        return position.fen()

    def storePosition(self, position: chess.Board, eval: int):
        key = self._key(position)
        self.hashTable[key] = eval

    def lookup(self, position: chess.Board):
        key = self._key(position)
        return self.hashTable.get(key)

    def exists(self, position: chess.Board) -> bool:
        key = self._key(position)
        return key in self.hashTable
