import time




def iterativeDeepen(board, max_time):
    start_time = time.time()
    depth = 1
    best_move = None    
    while True:
        if time.time() - start_time > max_time:
            break
        move = print("search would be here when he is done with itnd this takes the best_move")
        if move is not None:
            best_move = move
        depth += 1
        
    return best_move