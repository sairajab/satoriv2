import random
from create_dataset.mm_seq import GCRichSeq
from create_dataset.readpwms import get_motif_proteins, get_motif_seq, load_data

class SimulatedDataGenerator:
    def __init__(self, seq_len, path_to_meme, pairs_file, num_of_seq, output_name="sequences"):
        self.seq_len = seq_len
        self.num_of_seq = num_of_seq
        self.output_name = output_name
        self.proteins, self.pwms = get_motif_proteins(path_to_meme)
        self.pairs, self.tfs = load_data(pairs_file,tf_database=path_to_meme)

    @staticmethod
    def nucleotide_shuffle(sequence):
        sequence_list = list(sequence)
        random.shuffle(sequence_list)
        return ''.join(sequence_list)

    @staticmethod
    def dinucleotide_shuffle(sequence):
        dinucleotides = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
        random.shuffle(dinucleotides)
        return ''.join(dinucleotides)
    
    def generate_positions(self, start, end, gap, range_length):
        """
        Generate positions in ranges with a minimum gap between them.
        
        Parameters:
        - start: Starting position of the first range.
        - end: Maximum end position.
        - gap: Minimum distance required between ranges.
        - range_length: Length of each range.
        
        Returns:
        - List of ranges with positions.
        """
        positions = []
        current_start = start
        while current_start + range_length <= (end-gap):
            current_end = current_start + range_length
            positions.append(list(range(current_start, current_end)))
            current_start = current_end + gap
        return positions

    def embed_motifs_dist_300bp(self, seq, motifs, pos=True):
        # Position ranges are defined to avoid overlap between motifs 
        positions = [[*range(5, 15)], [*range(80, 90)],[*range(155, 165)], [*range(225, 232)]]
        seq = self.nucleotide_shuffle(seq)
        dist = [*range(8, 15)] #distance between two interacting motifs
        random.shuffle(positions)
        random.shuffle(dist)
        motif_to_pos = {}
        for i, e in enumerate(motifs):
            motif_to_pos[e] = {}
            position = random.choice(positions[i % len(positions)])
            for cons in motifs[e]:
                bp = len(motifs[e][cons])
                motif_to_pos[e][cons] = position
                seq = seq[:position] + motifs[e][cons] + seq[position + bp:]

                position = position + bp + dist[i % len(dist)] if pos else position
        return seq, motif_to_pos
    
    def embed_motifs_dist_1500bp(self, seq, motifs, pos=True):
        # Position ranges are defined to avoid overlap between motifs 
        dist = [*range(8, 80)] #distance between two interacting motifs
        #dist = [*range(8, 20)] #copying from 300bp
        gap = 200 # gap between non interatcting motifs
        positions = self.generate_positions(dist[0], self.seq_len, gap, dist[-1] - dist[0])
        seq = self.nucleotide_shuffle(seq)
        random.shuffle(positions)
        random.shuffle(dist)
        motif_to_pos = {}
        for i, e in enumerate(motifs):
            motif_to_pos[e] = {}
            position = random.choice(positions[i % len(positions)])
            for cons in motifs[e]:
                bp = len(motifs[e][cons])
                motif_to_pos[e][cons] = position
                seq = seq[:position] + motifs[e][cons] + seq[position + bp:]
                if pos:
                    position = position + bp + dist[i % len(dist)]
                else:
                    position = position
        return seq, motif_to_pos
    

    def motifs2embed(self, tf_pairs, uniq_tfs, pos=True, last_batch=False):
        pairs = []
        if pos:
            no_pairs = 2 if last_batch else random.choices([2, 3, 4], [0.33, 0.33, 0.34])[0] + 1

            if last_batch:
                no_tfs = random.randint(1, 2)
                values = random.sample(range(len(uniq_tfs)), no_tfs)
                pairs.extend(values)
            else:
                values = random.sample(range(len(tf_pairs)), no_pairs)
                pairs.extend(values)
        else:
            no_tfs = random.randint(2, 5)
            values = random.sample(range(len(uniq_tfs)), no_tfs)
            pairs.extend(values)
        return pairs

    def generate_data(self):
        label_output = open(self.output_name + ".txt", "w")
        fasta_output = open(self.output_name + ".fa", "w")
        annotations_output = open(self.output_name + "_info.txt", "w")
        annotations_output.write(f"This dataset has interacting pairs from a set of {len(self.pairs)} pairs and motifs from a set of {len(self.tfs)} motifs\n")

        for i in range(self.num_of_seq):
            seq = GCRichSeq(self.seq_len)
            if i < self.num_of_seq * 1:
                ids = self.motifs2embed(self.pairs, self.tfs)
                tf_motif = get_motif_seq(self.pwms, ids, self.pairs, self.tfs, pair=True)
                if self.seq_len == 300:
                    seq_embed, motif_to_pos = self.embed_motifs_dist_300bp(seq, tf_motif)
                elif self.seq_len == 1500:
                    seq_embed, motif_to_pos = self.embed_motifs_dist_1500bp(seq, tf_motif)
                else:
                    print("Either pick 300 or 1500..")

            else:
                ids = self.motifs2embed(self.pairs, self.tfs, pos = False)
                tf_motif = get_motif_seq(self.pwms, ids, self.pairs, self.tfs, pair=False)
                if self.seq_len == 300:
                    seq_embed, motif_to_pos = self.embed_motifs_dist_300bp(seq, tf_motif, pos=False)
                elif self.seq_len == 1500:
                    seq_embed, motif_to_pos = self.embed_motifs_dist_1500bp(seq, tf_motif, pos=False)
                else:
                    print("Either pick 300 or 1500..")

            info_str = ":".join([f"{motif}[{pos}]" for e in motif_to_pos for motif, pos in motif_to_pos[e].items()])
            line = f">Pos:1-{i+1}(+)\n"
            fasta_output.write(line)
            fasta_output.write(seq_embed + "\n")
            label_output.write(f"Pos\t1\t{i+1}\t1\n")
            annotations_output.write(f"Pos\t1\t{i+1}\t1:{info_str}\n")
        
        for i in range(self.num_of_seq):
            seq = GCRichSeq(self.seq_len)
            ids = self.motifs2embed(self.pairs, self.tfs, pos = False)
            tf_motif = get_motif_seq(self.pwms, ids, self.pairs, self.tfs, pair=False)
            if self.seq_len == 300:
                seq_embed, motif_to_pos = self.embed_motifs_dist_300bp(seq, tf_motif, pos=False)
            elif self.seq_len == 1500:
                seq_embed, motif_to_pos = self.embed_motifs_dist_1500bp(seq, tf_motif, pos=False)
            else:
                print("Either pick 300 or 1500..")

            
            info_str = ":".join([f"{motif}[{pos}]" for e in motif_to_pos for motif, pos in motif_to_pos[e].items()])
            line = f">Neg:2-{i+1}(+)\n"
            fasta_output.write(line)
            fasta_output.write(seq_embed + "\n")
            label_output.write(f"Neg\t2\t{i+1}\t0\n")
            annotations_output.write(f"Neg\t2\t{i+1}\t0:{info_str}\n")

        label_output.close()
        fasta_output.close()
        annotations_output.close()

if __name__ == "__main__":
    
    path_to_meme = "/s/chromatin/p/nobackup/Saira/motif_databases/Jaspar.meme"
    pairs_file = "tf_pairs_80.txt"
    output_name = "../data/LongSimulated_Data/Data-80/seqs_80_5"
    embedder = SimulatedDataGenerator(seq_len=1500, path_to_meme=path_to_meme, pairs_file=pairs_file, num_of_seq=30000, output_name=output_name)
    embedder.generate_data()
