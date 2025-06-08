import sys

fn1='./output_binary/program1_network1_0_convolution__neck_Bifusion0_cv3_conv_Conv_WithoutBiases_dst0.txt'
fn2='./output_binary_master/program1_network1_0_convolution__neck_Bifusion0_cv3_conv_Conv_WithoutBiases_dst0.txt'

fn1='./output_binary/program1_network1_0_convolution__backbone_ERBlock_2_ERBlock_2.1_block_block.0_block_conv_Conv_WithoutBiases_src2.txt'
fn2='./output_binary_master/program1_network1_0_convolution__backbone_ERBlock_2_ERBlock_2.1_block_block.0_block_conv_Conv_WithoutBiases_src2.txt'

if len(sys.argv) == 3:
    fn1 = sys.argv[1]
    fn2 = sys.argv[2]
    print(f"New fn1={fn1}")
    print(f"New fn2={fn1}")

def compare_files(fn1, fn2):
    # Open File in Read Mode 
    file_1 = open(fn1, 'r') 
    file_2 = open(fn2, 'r') 
  
    # print("Comparing files ", " @ " + 'file1.txt', " # " + 'file2.txt', sep='\n') 
  
    file_1_line = file_1.readline() 
    file_2_line = file_2.readline() 
    
    # Use as a COunter 
    line_no = 1
    diff_num = 0

    # print("Difference Lines in Both Files") 
    while file_1_line != '' or file_2_line != '': 
    
        # Removing whitespaces 
        file_1_line = file_1_line.rstrip() 
        file_2_line = file_2_line.rstrip() 
    
        # Compare the lines from both file 
        if file_1_line != file_2_line: 
            # otherwise output the line on file1 and use @ sign 
            if file_1_line == '': 
                print("@", "Line-%d" % line_no, file_1_line) 
            else: 
                print("@-", "Line-%d" % line_no, file_1_line) 
                
            # otherwise output the line on file2 and use # sign 
            if file_2_line == '': 
                print("#", "Line-%d" % line_no, file_2_line) 
            else: 
                print("#+", "Line-%d" % line_no, file_2_line) 
    
            # Print a empty line 
            print() 
            diff_num = diff_num + 1
    
        # Read the next line from the file 
        file_1_line = file_1.readline() 
        file_2_line = file_2.readline() 
    
        line_no += 1
    
    file_1.close() 
    file_2.close()
    print(f"Total diff: {diff_num} / {line_no}")

compare_files(fn1, fn2)