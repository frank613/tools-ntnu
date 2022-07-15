import sys

def get_dist(ref_sent, input_sent, ref_idx, input_idx, dp_matrix):
    if dp_matrix[ref_idx][input_idx] is not None:
        return dp_matrix[ref_idx][input_idx] # return the stored tuple (distance and sequence)
    #seqLen = max(len(input_sent), len(ref_sent))
    if ref_idx + 1 > len(ref_sent) and input_idx + 1 <= len(input_sent): # ref words used up
        fill_length = len(input_sent) - input_idx
        temp_list = ["I"] * fill_length
        dp_matrix[ref_idx][input_idx] = (fill_length, temp_list) #store the score and the partial sequence
        return (fill_length, temp_list) 
    if input_idx + 1 > len(input_sent) and ref_idx + 1 <= len(ref_sent): # input words used up
        fill_length = len(ref_sent) - ref_idx
        temp_list = ["D"] * fill_length   
        dp_matrix[ref_idx][input_idx] = (fill_length, temp_list) #store the score and the partial sequence
        return (fill_length, temp_list)
    if ref_idx + 1 > len(ref_sent) and input_idx + 1 > len(input_sent): # nothing left to correct
        return (0, [])
    if input_sent[input_idx] == ref_sent[ref_idx]:
        dist, temp_list = get_dist(ref_sent, input_sent, ref_idx+1, input_idx+1, dp_matrix)
        temp_list = ["C"] + temp_list
        dp_matrix[ref_idx][input_idx] = (dist, temp_list)
        return (dist, temp_list)
    else: ## current position is wrong, using the iterative equations
        tuple_a = get_dist(ref_sent, input_sent, ref_idx, input_idx+1, dp_matrix) ## insertion error, need to delete one(input_idx+1) from the input, 
        tuple_b = get_dist(ref_sent, input_sent, ref_idx+1, input_idx, dp_matrix) ## deletion error, need to add one in the input and consume one in the ref (ref_idx+1) 
        tuple_c = get_dist(ref_sent, input_sent, ref_idx+1, input_idx+1, dp_matrix) ## substitution error, need to add one in the input and consume one in the ref (ref_idx+1) 
        winner = min([tuple_a, tuple_b, tuple_c], key = lambda p : p[0])
        dist, temp_list = winner

        if winner == tuple_a: #insertion error
            temp_list = ["I"] + temp_list
        elif winner == tuple_b: #deletion error
            temp_list = ["D"] + temp_list
        else: #substitution error
            temp_list = ["S"] + temp_list 

        dp_matrix[ref_idx][input_idx] = (dist+1, temp_list)
        return (dist+1, temp_list)

        

def edit_dist(ref_sent, input_sent):
    print("comparing {0} to {1}".format(input_sent, ref_sent))
    dp_matrix = [[None] * (len(input_sent) + 1) for i in range(len(ref_sent)+1)]  #+1 for saving the boundary case when ref/input words used up
    #print(dp_matrix)
    dist, sequence = get_dist(ref_sent, input_sent, 0, 0, dp_matrix)
    if dist > max(len(input_sent), len(ref_sent)) or dist < abs(len(input_sent) - len(ref_sent)):
        print("upper or lower bound of the lev dist is not attained")
    print(dist, sequence)
    




if __name__ == "__main__":

    input_sent = "a b c d e f g"
    #input_sent = "b c"
    ref_sent =   "a b b c d e"
    #ref_sent =   "b e"
    edit_dist(ref_sent.split(), input_sent.split())
