def highlight_error(ftext, pos_start,pos_end):
    result=''

    # Calculate Indexes
    idx_start = max(ftext.rfind('\n',0,pos_start.idx),0)
    idx_end   = ftext.find('\n',idx_start+1)
    if idx_end<0: idx_end =len(ftext)

    # Generate Each Line
    line_count =pos_end.line - pos_start.line +1
    for i in range(line_count):
        #Calc line cols
        line = ftext[idx_start:idx_end]
        col_start = pos_start.col if i==0 else 0
        col_end = pos_end.col if i==line_count-1 else len(line)-1
        #Append to results
        result+=line+'\n'
        result+=' '*col_start + '^'*(col_end-col_start)
        #Re-calc idexes
        idx_start = idx_end
        idx_end=ftext.find('\n',idx_start+1)
        if idx_end<0: idx_end=len(ftext)
    return result.replace('\t','')