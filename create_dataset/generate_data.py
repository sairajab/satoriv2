from random import choice, randint
from readpwms import get_motif_proteins, get_motif_seq, load_data
import random
from mm_seq import GCRichSeq
import numpy as np


def nucleotide_shuffle(sequence):
    # Convert the DNA sequence string to a list for shuffling
    sequence_list = list(sequence)
    
    # Use Fisher-Yates shuffle algorithm to shuffle the list
    for i in range(len(sequence_list) - 1, 0, -1):
        j = random.randint(0, i)
        sequence_list[i], sequence_list[j] = sequence_list[j], sequence_list[i]
    
    # Convert the shuffled list back to a string
    shuffled_sequence = ''.join(sequence_list)
    
    return shuffled_sequence


def dinucleotide_shuffle(sequence):
    # Convert the sequence to a list of dinucleotides
    dinucleotides = [sequence[i:i+2] for i in range(0, len(sequence), 2)]

    # Shuffle the dinucleotides randomly
    random.shuffle(dinucleotides)

    # Recombine the shuffled dinucleotides into a shuffled sequence
    shuffled_sequence = ''.join(dinucleotides)

    return shuffled_sequence


def embed_motifs(seq, motifs, pos = True):
    positions = [0,21,42,65,87,108, 129, 150,171]#np.arange(0,175).tolist()
    #position = choice(positions)
    # for j in range(no_of_motifs):
    #     positions.append()
    #
    # position =  randint(1, len(seq)-20)
    random.shuffle(positions)
    #random.shuffle(dist)
    i = 0
    k = 0
    last_tf = ""
    motifs_count = {}
    for e in motifs:
        if i == 9:
            i = 0
        for cons in motifs[e]:
            position = positions[i]
            bp = len(motifs[e][cons])
            # while not set(np.arange(position, position + bp + 1)).issubset(set(positions)):
            #     print("finding")
            #     position = choice(positions)

            if pos:
                print("pos is" + str(position))
                seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
            else:
                seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
                #print(position)

        i = i + 1
        if len(seq) > 200:
            raise Exception("Sorry, length above 200");
    return seq

def embed_motifs_dist(seq, motifs, pos = True):
    '''
    This function embeds given motifs-pair in the sequence with in <dist> : [10,8,9,11] and returns sequence
    '''
    positions = [[*range(45,55)],[*range(135,145)],[*range(225,235)],
                 [*range(315,320)]]
    #300bp
    positions = [[*range(5,15)],[*range(80,90)],[*range(225,232)],
                 [*range(155, 165)]]
    #print(seq)
    seq = nucleotide_shuffle(seq)
    #print(seq)


    dist = [*range(8,15)]
    # for j in range(no_of_motifs):
    #     positions.append()
    #
    # position =  randint(1, len(seq)-20)
    random.shuffle(positions)
    random.shuffle(dist)
    #random.shuffle(seq)
    i = 0
    k = 0
    last_tf = ""
    motif_to_pos = {} #Which motifs are embedded at which positions
    last_pos = 0
    print(motifs)
    for e in motifs:
        motif_to_pos[e] = {}
        if i == 4:
            i = 0
        position = random.choice(positions[i])
        #print(positions[i], position)
        for cons in motifs[e]:
            bp = len(motifs[e][cons])
            motif_to_pos[e][cons] = position 
            '''If the sequence is positive sequence, embed with distance'''
            if pos:
                seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
                position = position + bp + dist[i]

            else:
                '''If sequence is not positive embed motifs randomly using <positions> variable'''
                # while abs(last_pos - position) < bp + 15:
                #     print(last_pos,position, bp )
                #     position = random.choice(positions[i])
                    
                seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
                print(position)
                last_pos = position
        i = i + 1
        if len(seq) > 300:
           raise Exception("Sorry, length above 200");
    return seq, motif_to_pos

def embed_motifs_updated(seq, motifs, pos = True):
    '''
    This function embeds given motifs-pair in the sequence with in <dist> : [10,8,9,11] and returns sequence
    '''
    
    seq = dinucleotide_shuffle(seq)

    
    sequence_length = len(seq)

    dist = [*range(8,15)]
    # for j in range(no_of_motifs):
    #     positions.append()
    #
    # position =  randint(1, len(seq)-20)
    random.shuffle(dist)
    #random.shuffle(seq)
    i = 0
    k = 0
    last_tf = ""
    motif_to_pos = {} #Which motifs are embedded at which positions
    last_pos = 0
    occupied_positions = set()
    print(motifs)
    for e in motifs:
        motif_to_pos[e] = {}
        keys = list(motifs[e].keys())
        print(keys)
        motif_length = len(motifs[e][keys[0]])
        desired_distance = random.choice(dist)
        # Randomly choose a starting position for motif1 that is not occupied
        if pos:
            max_start_position = sequence_length - motif_length - desired_distance - len(motifs[e][keys[1]])
        else:
            max_start_position = sequence_length - motif_length
            
        if max_start_position < 0:
                raise ValueError("Sequence length is too short for the desired distance.")
        available_positions = [i for i in range(max_start_position + 1) if i not in occupied_positions]
        if not available_positions:
                raise ValueError("No available positions for motif.")
            
        start_position = random.choice(available_positions)

        # Check if the chosen position is within the desired distance of any occupied positions
        while any(abs(start_position - pos) < motif_length for pos in occupied_positions):
            start_position = random.choice(available_positions)
        
        #If sequence is positive
        
        if pos: 

            # Place motif1 in the sequence
            seq = seq[:start_position] + motifs[e][keys[0]] + seq[start_position + motif_length:]

            # Calculate the end position of motif1
            end_position_motif1 = start_position + motif_length
            
            # Calculate the maximum allowed start position for motif2
            #max_start_position_motif2 = min(end_position_motif1 + desired_distance, sequence_length - motif_length)

            #if max_start_position_motif2 <= end_position_motif1:
                # Not enough space to place motif2
            #    raise ValueError("Not enough space in the sequence for motif2.")

            # Randomly choose a starting position for motif2 within the allowed range
            #start_position_motif2 = random.randint(end_position_motif1, max_start_position_motif2)
            
            # Calculate the maximum allowed start position for motif2
            motif_length_2 = len(motifs[e][keys[1]])

 


            # Calculate the start position of motif2 based on the desired distance
            start_position_motif2 = end_position_motif1 + desired_distance
            

            # Check if there is enough space for motif2
            if start_position_motif2 + motif_length_2 > sequence_length:
                 start_position_motif2 = start_position - desired_distance - motif_length_2
                    #raise ValueError("Not enough space in the sequence for motif2.")

            # Place motif2 in the sequence
            seq = seq[:start_position_motif2] + motifs[e][keys[1]] + seq[start_position_motif2 + motif_length_2:]
            
            #Save positions of motifs
            motif_to_pos[e][keys[0]] = start_position
            motif_to_pos[e][keys[1]] = start_position_motif2


            # Update the set of occupied positions
            occupied_positions.update(range(start_position , start_position + motif_length))
            occupied_positions.update(range(start_position_motif2, start_position_motif2 + motif_length))
        else:
            
            # Place motif1 in the sequence
            seq = seq[:start_position] + motifs[e][keys[0]] + seq[start_position + motif_length:]
            
            #Save positions of motifs
            motif_to_pos[e][keys[0]] = start_position
            
            # Update the set of occupied positions
            occupied_positions.update(range(start_position - max(dist)-motif_length, start_position + motif_length + max(dist))) 
            
            


            
            
    #     if i == 4:
    #         i = 0
    #     position = random.choice(positions[i])
    #     #print(positions[i], position)
    #     for cons in motifs[e]:
    #         bp = len(motifs[e][cons])
    #         motif_to_pos[e][cons] = position 
    #         '''If the sequence is positive sequence, embed with distance'''
    #         if pos:
    #             seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
    #             position = position + bp + dist[i]

    #         else:
    #             '''If sequence is not positive embed motifs randomly using <positions> variable'''
    #             while abs(last_pos - position) < bp + 15:
    #                 print(last_pos,position, bp )
    #                 position = random.choice(positions[i])
                    
    #             seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
    #             print(position)
    #             last_pos = position
    #     i = i + 1
    #     # if len(seq) > 200:
    #     #     raise Exception("Sorry, length above 200");
    return seq, motif_to_pos


def motifs2embed(tf_pairs, uniq_tfs, pos = True, last_batch = False):
    '''
    This method returns the ids of motifs to be embedded in positive and negative sequences. 
    Number of pairs : 80
    Number of unique TFs : 93
    '''
    total_pairs = len(tf_pairs)
    total_unique = len(uniq_tfs)
    pairs = []
    pair_choices = [1,2] # add 1 pair or 2 pairs
    pair_prob = [0.4,0.6] # Probability of picking 1 pair or 2
    pair_copy = [1, 0] # Probability of copying pair 
    #pair_copy = [1, 0] # Probability of copying pair which is actually no copy
    if pos: #Positive sequence
        #randomly pick number of pairs
        no_pairs = 3 #pair_choices[np.random.choice(2, p=pair_prob)] + 1
        
        if last_batch:
            no_pairs = 2
        #Randomly pick which pair id from 80 #40 for simplicity                
        values = random.sample(range(total_pairs), no_pairs)
        for i in range(no_pairs):
            
            #randomly pick if you wanna copy a pair
            copy = pair_choices[np.random.choice(2, p=pair_copy)]
            if copy == 2:
                    print("huhuhu I am copying")
                    pairs.append(values[i])
            pairs.append(values[i])
            
        if last_batch:
            
            no_tfs = randint(1,2)
            values = random.sample(range(total_unique), no_tfs)

            for i in range(no_tfs):
                #randomly pick motif to embed from 93 unique motifs
                value = randint(0, total_unique-1) #just from the tf-pairs set #randint(0, 92)
                pairs.append(values[i])
            
    
    else: #Negative sequences
        #Randomly pick number of motifs to embed in negative sequences
        no_tfs = randint(1,3)
        values = random.sample(range(total_unique), no_tfs)

        for i in range(no_tfs):
            #randomly pick motif to embed from 93 unique motifs
            value = randint(0, total_unique-1) #just from the tf-pairs set #randint(0, 92)
            pairs.append(values[i])
    return pairs

def generateData(no_of_positive, no_of_negative, seq_len, pwms, pairs_file,  outfile_name = "seqs93_tf_2_a"):
    '''
    '''
    pairs, tfs = load_data(pairs_file) 
    label_output = open(outfile_name + ".txt", "w")
    fasta_output = open(outfile_name + ".fa", "w")
    annotations_output = open(outfile_name + "_info.txt", "w")
    annotations_output.writelines("This dataset has 2 or 3 interacting pairs from a set of " +str(len(pairs)) + "pairs in it and in negative sequence one by chance (1-3) motifs per sequence from a set of " + str(len(tfs))+ " motifs\n")
    head = 1

    last_batch = no_of_positive*0.0 #keep all batch included (0.1 if want 10% to be different/noisy data)

    for i in range(no_of_positive):
        # Generates GC Rich sequence of a given length 
        seq = GCRichSeq(seq_len)

        if i < no_of_positive - last_batch:
            # Get ids of motif-pairs to embed in positive sequence
            ids = motifs2embed(pairs, tfs)
            #get motif sequence everytime using letter probability matrix
            tf_motif = get_motif_seq(pwms, ids, pairs, tfs, pair=True)
            #Embed motifs in positive sequence 
            seq_embed, motif_to_pos = embed_motifs_dist(seq, tf_motif)

        else:
            
            # Get ids of motif-pairs to embed in positive sequence
            ids= motifs2embed(pairs, tfs, pos=True, last_batch = True)
            #get motif sequence everytime using letter probability matrix
            tf_motif = get_motif_seq(pwms, ids,pairs,tfs,last_batch=True, pair=False)

            #Embed motifs in positive sequence 
            seq_embed, motif_to_pos = embed_motifs_dist(seq, tf_motif, pos = False)
        #Embed motifs new method
        #seq_embed, motif_to_pos = embed_motifs_updated(seq, tf_motif)
        print(len(seq_embed))

        #Get info about which motifs are embedded at which positions and write in file
        info_str = ""
        for num in motif_to_pos:
            for motif in motif_to_pos[num]:
                info_str = motif + "[" + str(motif_to_pos[num][motif]) + "]:" + info_str
        # Write .fa and .txt files
        line = ">Pos:1-" + str(head) + "(+)\n"
        fasta_output.write(line)
        fasta_output.write(seq_embed + "\n")
        label_output.writelines("Pos\t1\t" + str(head) + '\t' + '1' + '\n')
        annotations_output.writelines("Pos\t1\t" + str(head) + '\t' + '1:' + info_str[:-1] + '\n')

        head = head + 1

    for i in range(int(no_of_negative * 1)):
        #Generate GC Rich Sequence
        seq = GCRichSeq(seq_len)
        # Get ids of motifs for negative sequence
        idxs = motifs2embed(pairs, tfs, pos=False)
        #get motif sequence everytime using letter probability matrix
        tf_motif = get_motif_seq(pwms, idxs, pairs, tfs, pair=False)
        #Embed motifs in negative sequence
        seq_embed, motif_to_pos = embed_motifs_dist(seq, tf_motif, pos=False) #embed_motifs
        
        #Embed motifs new method
        #seq_embed, motif_to_pos = embed_motifs_updated(seq, tf_motif, pos=False)
        #Get info about which motifs are embedded at which positions and write in file
        info_str = ""
        for num in motif_to_pos:
            for motif in motif_to_pos[num]:
                info_str = motif + "[" + str(motif_to_pos[num][motif]) + "]:" + info_str
        # Write .fa and .txt files
        line = ">Neg:2-" + str(head) + "(+)\n"
        fasta_output.write(line)
        fasta_output.write(seq_embed + "\n")
        label_output.writelines("Neg\t2\t" + str(head) + '\t' + '0' + '\n')
        annotations_output.writelines("Neg\t2\t" + str(head) + '\t' + '0:' + info_str[:-1] + '\n')

        head = head + 1

    label_output.close()
    annotations_output.close()
    fasta_output.close()
    return True


if __name__ == "__main__":

    path_to_meme = '../../../motif_databases/Jaspar.meme'
    num_of_seq = 60000
    seq_len = 300
    protiens, pwms = get_motif_proteins(path_to_meme)
    #pairs_file = "/s/jawar/i/nobackup/Saira/latest/satori_v2/data/ToyData/Ctfs/tf_pairs_data_1.txt"
    pairs_file = "tf_pairs_80.txt"
    
    
    for i in range(1):
        success = generateData(num_of_seq,num_of_seq, seq_len,pwms,pairs_file, "../data/ToyData/NEWDATA/ctf_80pairs_eq" + str(i+2))



