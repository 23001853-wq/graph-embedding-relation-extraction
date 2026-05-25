# preprocessing/parser.py

def align_tokens_to_chars(doc, char_start, char_end, entity_text):
    """
    Robust mapping from character offset to token offset.
    Returns the token range [start_idx, end_idx].
    """
    span = doc.char_span(char_start, char_end)
    if span is not None and span.text.strip() == entity_text.strip():
        return span.start, span.end - 1
        
    span = doc.char_span(char_start, char_end - 1)
    if span is not None and span.text.strip() == entity_text.strip():
        return span.start, span.end - 1
    
    start_idx = -1
    end_idx = -1
    for token in doc:
        if token.idx < char_end and (token.idx + len(token.text)) > char_start:
            if not token.text.strip():
                continue
            if token.text in [',', '.', ';', ':'] and token.text not in entity_text:
                continue
            
            if start_idx == -1:
                start_idx = token.i
            end_idx = token.i
            
    return start_idx, end_idx