# preprocessing/segmenter.py

def get_positions(seq_len, start, end):
    """Tính relative position (khoảng cách tương đối từ mỗi token tới entity)."""
    pos = []
    for i in range(seq_len):
        if i < start:
            pos.append(i - start)
        elif i > end:
            pos.append(i - end)
        else:
            pos.append(0)
    return pos

def get_segments(seq_len, h_start, h_end, t_start, t_end):
    """
    Chia câu thành 5 segments: Left, Entity 1, Middle, Entity 2, Right.
    """
    if h_start < t_start:
        e1_s, e1_e = h_start, h_end
        e2_s, e2_e = t_start, t_end
    else:
        e1_s, e1_e = t_start, t_end
        e2_s, e2_e = h_start, h_end
        
    segments = []
    for i in range(seq_len):
        if i < e1_s:
            segments.append(1) # left
        elif e1_s <= i <= e1_e:
            segments.append(2) # e1
        elif e1_e < i < e2_s:
            segments.append(3) # middle
        elif e2_s <= i <= e2_e:
            segments.append(4) # e2
        elif i > e2_e:
            segments.append(5) # right
    return segments