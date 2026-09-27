#!/usr/bin/env python3
from torch_varlen_candidate import validate_layout

def rejects(fn):
 try: fn()
 except ValueError: return
 raise AssertionError("invalid contract accepted")

assert validate_layout((1,16,16040,128),(1,2,16040,128),(1,2,16040,128),[3478,1191,4972,647,291,4287,1174]) == {"total_tokens":16040,"max_seqlen":4972,"gqa_groups":8}
rejects(lambda: validate_layout((1,16,10,128),(1,2,10,128),(1,2,10,128),[4,5]))
rejects(lambda: validate_layout((1,16,10,128),(1,16,10,128),(1,16,10,128),[10]))
rejects(lambda: validate_layout((1,16,10,128),(1,2,10,64),(1,2,10,128),[10]))
rejects(lambda: validate_layout((1,16,10,128),(1,2,10,128),(1,2,10,128),[0,10]))
print("PASS exact GQA layout, boundary conservation, head and dimension negative controls")
