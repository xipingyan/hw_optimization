import sys

if len(sys.argv) == 3:
    fn1 = sys.argv[1]
    out_fn = sys.argv[2]
    print(f"New fn1 = {fn1}")
    print(f"New fn2 = {out_fn}")
else:
    print("$ app file1.txt output_file.txt")
    exit()

def cvt_log(fn1, out_fn):
    # Open File in Read Mode 
    file_1 = open(fn1, 'r') 

    file_1_line = file_1.readline() 
    
    # Use as a COunter 
    line_no = 1
    new_pair=[]

    # print("Difference Lines in Both Files") 
    while file_1_line != '': 
        # Removing whitespaces 
        file_1_line = file_1_line.rstrip()

        # output[2047]=-1.630859
        key, value = file_1_line.split('=')
        k1, k2 = key.split('[')
        idx, fk = k2.split(']')
    
        # print("idx=", idx, value)
        new_pair.append((int(idx), value))
    
        # Read the next line from the file 
        file_1_line = file_1.readline() 
        line_no += 1

    file_1.close() 

    new_pair = sorted(new_pair, key=lambda x: x[0])

    out_f = open(out_fn, 'w') 
    for k in new_pair:
        # print(k[1])
        out_f.write(k[1] + "\n")
    out_f.close()

    print(f"Total count:{line_no}")

cvt_log(fn1, out_fn)