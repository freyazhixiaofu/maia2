import chess
import chess.engine
from maia2 import model, dataset, inference
import pdb

# stockfish shallow search depth
SF_SHALLOW = 10
# stockfish leaf search
SF_DEEP = 18
# top TOP_N_SF number of stockfish root moves to fetch (the multipv)
TOP_N_SF = 6
# threshold for maia's lowest acceptable prob
P_MIN = 0.8
# depth of maiafish for filtering
MAX_DEPTH = 3
# threshold of branching factor
CHILDREN_CAP = 8

def sf_eval(engine, board, depth):
    """calling stockfish with depth and board given"""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].pov(board.turn).score(mate_score=100000)

def sf_top_lines(engine, board, shallow_depth, multipv):
    """Return dict {first move of the line: centipawn} for top multipv moves"""
    lines = engine.analyse(board, chess.engine.Limit(depth=shallow_depth), multipv=multipv)
    out = {}
    for line in lines:
        pv = line.get("pv", [])
        if not pv:
            continue
        mv = pv[0].uci()
        cp = line["score"].pov(board.turn).score(mate_score=100000)
        out[mv] = cp
    return out

# note: the "line" is the sequence of moves starting from the given board position
def maiafish(board, engine, maia2_model, prepared, depth, elo_self, elo_oppo):
    """ return (best_score, best_lineofmoves) from given position board"""
    if depth >= MAX_DEPTH or board.is_game_over():
        leaf_score = sf_eval(engine, board, SF_DEEP) 
        return leaf_score, []
    
    move_probs, _ = inference.inference_each(
        maia2_model, prepared, board.fen(), elo_self, elo_oppo) # board has to be white move
    
    sf_top = sf_top_lines(engine, board, SF_SHALLOW, TOP_N_SF) # lichess is better?
    # dict {first_move of the line: centipawn}
    
    cand = []
    for mv in board.legal_moves:
        u = mv.uci()
        p = move_probs.get(u, 0)
        cp = sf_top.get(u, None)
        # filtering the left bottom corner
        if ((p > P_MIN) or (u in sf_top)):
            cand.append(u, p, cp if cp is not None else -10**9)
            
    cand = cand[:CHILDREN_CAP]   
    #set a threshold for branching factor, in case too many points in left upper corner     
    
    best_score, best_line = -10**0, []
    for u, p, cp in cand:
        board.push(chess.Move.from_uci(u))
        child_score, child_line = maiafish(board, engine, maia2_model, prepared, depth+1)
        board.pop()
        
        if child_score > best_score:
            best_score = child_score
            best_line = [u] + child_line
        
    return best_score, best_line



engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") #local path
maia2_model = model.from_pretrained(type="rapid", device="gpu")
prepared = inference.prepare()
data = dataset.load_example_test_dataset() # 'board', 'move', 'active_elo', 'opponent_elo'

for fen, move, elo_self, elo_oppo in data.values[:10]:
    board = chess.Board(fen)
    score_cp, pv = maiafish(board, engine, maia2_model, prepared, MAX_DEPTH, 
                            elo_self, elo_oppo)
    pdb.set_trace()
    #print(f"score {score_cp}, pv{pv[0]}")