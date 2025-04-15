import numpy as np
import pandas as pd
import random
import torch
from fastprogress import progress_bar
from random import randint
from torch.utils.data import Dataset
import re

class DatasetLoadAll(Dataset):
    def __init__(self, df_path, num_labels=2, for_embeddings=False):
        self.DNAalphabet = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}
        # just in case the user provide extension
        df_path = df_path.split('.')[0]
        self.df_all = pd.read_csv(df_path+'.txt', delimiter='\t', header=None)
        self.df_seq = pd.read_csv(df_path+'.fa', header=None)
        strand = self.df_seq[0][0][-3:]  # can be (+) or (.)
        self.df_all['header'] = self.df_all.apply(
            lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+strand, axis=1)
        self.chroms = self.df_all[0].unique()
        self.df_seq_all = pd.concat([self.df_seq[::2].reset_index(
            drop=True), self.df_seq[1::2].reset_index(drop=True)], axis=1, sort=False)
        self.df_seq_all.columns = ["header", "sequence"]
        # self.df_seq_all['chrom'] = self.df_seq_all['header'].apply(lambda x: x.strip('>').split(':')[0])
        self.df_seq_all['sequence'].apply(lambda x: x.upper())
        self.num_labels = num_labels
        self.df = self.df_all
        self.df_seq_final = self.df_seq_all
        self.df = self.df.reset_index()
        self.df_seq_final = self.df_seq_final.reset_index()
        # self.df['header'] = self.df.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+'('+x[5]+')', axis=1)
        if for_embeddings == False:
            self.One_hot_Encoded_Tensors = []
            self.Label_Tensors = []
            self.Seqs = []
            self.Header = []
            for i in progress_bar(range(0, self.df.shape[0])):  # tqdm() before
                if self.num_labels == 2:
                    y = self.df[self.df.columns[-2]][i]
                else:
                    y = np.asarray(
                        self.df[self.df.columns[-2]][i].split(',')).astype(int)
                    y = self.one_hot_encode_labels(y)
                header = self.df['header'][i]
                self.Header.append(header)
                X = self.df_seq_final['sequence'][self.df_seq_final['header']
                                                  == header].array[0].upper()
                # X = X.replace('N',list(self.DNAalphabet.keys())[randint(0,3)])
                X = X.replace('N', list(self.DNAalphabet.keys())
                              [random.choice([0, 1, 2, 3])])
                X = X.replace('S', list(self.DNAalphabet.keys())
                              [random.choice([1, 2])])
                X = X.replace('W', list(self.DNAalphabet.keys())
                              [random.choice([0, 3])])
                X = X.replace('K', list(self.DNAalphabet.keys())
                              [random.choice([2, 3])])
                X = X.replace('Y', list(self.DNAalphabet.keys())
                              [random.choice([1, 3])])
                X = X.replace('R', list(self.DNAalphabet.keys())
                              [random.choice([0, 2])])
                X = X.replace('M', list(self.DNAalphabet.keys())
                              [random.choice([0, 1])])
                self.Seqs.append(X)
                X = self.one_hot_encode(X)
                self.One_hot_Encoded_Tensors.append(torch.tensor(X))
                self.Label_Tensors.append(torch.tensor(y))

    def __len__(self):
        return self.df.shape[0]
    
    def get_seq_len(self):
        return len(self.df_seq_all["sequence"][0])
    
    def get_all_data(self):
        return self.df, self.df_seq_final

    def get_all_chroms(self):
        return self.chroms

    def one_hot_encode(self, seq):
        mapping = dict(zip("ACGT", range(4)))
        seq2 = [mapping[i] for i in seq]
        return np.eye(4)[seq2].T.astype(np.int_)

    def one_hot_encode_labels(self, y):
        lbArr = np.zeros(self.num_labels)
        lbArr[y] = 1
        return lbArr.astype(np.int_)

    def __getitem__(self, idx):
        return self.Header[idx], self.Seqs[idx], self.One_hot_Encoded_Tensors[idx], self.Label_Tensors[idx]





class DatasetLazyLoad(Dataset):
    def __init__(self, df_path, num_labels=2):
        self.DNAalphabet = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}
        # just in case the user provide extension
        df_path = df_path.split('.')[0]
        self.df_all = pd.read_csv(df_path+'.txt', delimiter='\t', header=None)
        self.df_seq = pd.read_csv(df_path+'.fa', header=None)
        strand = self.df_seq[0][0][-3:]  # can be (+) or (.)
        self.df_all['header'] = self.df_all.apply(
            lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+strand, axis=1)
        self.chroms = self.df_all[0].unique()
        self.df_seq_all = pd.concat([self.df_seq[::2].reset_index(
            drop=True), self.df_seq[1::2].reset_index(drop=True)], axis=1, sort=False)
        self.df_seq_all.columns = ["header", "sequence"]
        self.df_seq_all['sequence'].apply(lambda x: x.upper())
        self.num_labels = num_labels
        self.df = self.df_all
        self.df_seq_final = self.df_seq_all
        self.df = self.df.reset_index()
        self.df_seq_final = self.df_seq_final.reset_index()
        self.df.columns.values[-2] = "label"  # Rename column at index -2

        print("Columns in df ", self.df.columns)
        print("Columns in df_seq_all ", self.df_seq_final.columns)
        # self.df['header'] = self.df.apply(lambda x: '>'+x[0]+':'+str(x[1])+'-'+str(x[2])+'('+x[5]+')', axis=1)

    def __len__(self):
        return self.df.shape[0]

    def get_seq_len(self):
        return len(self.df_seq_all["sequence"][0])

    def get_all_data(self):
        return self.df, self.df_seq_final

    def get_all_chroms(self):
        return self.chroms

    def one_hot_encode(self, seq):
        # Precompute the mapping of bases to indices
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # Create the mapping matrix
        mapping = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
            [0.25, 0.25, 0.25, 0.25]  # N
        ], dtype=np.float_)
        
        # Convert the sequence into a NumPy array of characters
        seq_array = np.array(list(seq))
    
        # Create an index map using vectorized operations
        index_map = np.vectorize(base_to_index.get)(seq_array, 4)        
        # Use advanced indexing to get the one-hot encoded matrix
        return mapping[index_map].T

    def one_hot_encode_labels(self, y):
        lbArr = np.zeros(self.num_labels)
        lbArr[y] = 1
        return lbArr.astype(np.int_)

    def __getitem__(self, idx):
        if self.num_labels == 2:
            y = self.df[self.df.columns[-2]][idx]
        else:
            y = np.asarray(self.df[self.df.columns[-2]]
                           [idx].split(',')).astype(int)
            y = self.one_hot_encode_labels(y)
        header = self.df['header'][idx]
        #print(header, idx)
        #print((self.df_seq_final['header'] == header).any()
        X = self.df_seq_final['sequence'][self.df_seq_final['header']
                                          == header].array[0].upper()
        seq = X
        
            
        X = self.one_hot_encode(X)
        return header, seq, torch.tensor(X), torch.tensor(y)


class DatasetLazyLoadRC(Dataset):
    def __init__(self, df_path, num_labels=2, rev_complement=False):
        self.DNAalphabet = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}
        
        # Load data
        #df_path = df_path.split('.')[0]
        print(df_path)
        self.df_all = pd.read_csv(df_path + '.txt', delimiter='\t', header=None)
        self.df_seq = pd.read_csv(df_path + '.fa', header=None)
        
        # Extract strand information (+) or (.)
        strand = self.df_seq[0][0][-3:]
        
        # Create headers
        self.df_all['header'] = self.df_all.apply(
            lambda x: f'>{x[0]}:{x[1]}-{x[2]}{strand}', axis=1
        )
        
        # Store unique chromosomes
        self.chroms = self.df_all[0].unique()

        # Combine headers and sequences
        self.df_seq_all = pd.concat(
            [self.df_seq[::2].reset_index(drop=True), self.df_seq[1::2].reset_index(drop=True)], 
            axis=1, sort=False
        )
        self.df_seq_all.columns = ["header", "sequence"]
        self.df_seq_all["sequence"] = self.df_seq_all["sequence"].str.upper()

        # Store number of labels
        self.num_labels = num_labels

        # Assign label column name
        self.df_all = self.df_all.rename(columns={self.df_all.columns[-2]: "label"})
        if rev_complement:
            # Add reverse complements
            self.add_reverse_complement()

        # Reset indices
        self.df_all = self.df_all.reset_index(drop=True)
        self.df_seq_final = self.df_seq_all.reset_index(drop=True)

        print("Columns in df:", self.df_all.columns)
        print("Columns in df_seq_all:", self.df_seq_final.columns)

    def __len__(self):
        return len(self.df_all)

    def get_seq_len(self):
        return len(self.df_seq_final["sequence"].iloc[0])

    def get_all_data(self):
        return self.df_all, self.df_seq_final

    def get_all_chroms(self):
        return self.chroms

    def reverse_complement(self, seq):
        complement = str.maketrans("ACGT", "TGCA")
        return seq.translate(complement)[::-1]

    def add_reverse_complement(self):
        """ Adds reverse complement sequences to the dataset with updated headers and same labels. """
        df_reverse = self.df_seq_all.copy()
        df_reverse["sequence"] = df_reverse["sequence"].apply(self.reverse_complement)
        df_reverse["header"] = df_reverse["header"].str.replace(r"\+", "-", regex=True) + "_rev"

        # Append reverse complement sequences
        self.df_seq_all = pd.concat([self.df_seq_all, df_reverse], ignore_index=True)

        # Duplicate labels for reverse complements
        df_labels_reverse = self.df_all.copy()
        df_labels_reverse["header"] = df_labels_reverse["header"].str.replace(r"\+", "-", regex=True) + "_rev"
        self.df_all = pd.concat([self.df_all, df_labels_reverse], ignore_index=True)

    def one_hot_encode(self, seq):
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        mapping = np.array([
            [1, 0, 0, 0],   # A
            [0, 1, 0, 0],   # C
            [0, 0, 1, 0],   # G
            [0, 0, 0, 1],   # T
            [0.25, 0.25, 0.25, 0.25]  # N
        ], dtype=np.float_)

        seq_array = np.array(list(seq))
        index_map = np.vectorize(base_to_index.get)(seq_array, 4)  
        return mapping[index_map].T

    def one_hot_encode_labels(self, y):
        lbArr = np.zeros(self.num_labels, dtype=np.int_)
        lbArr[y] = 1
        return lbArr

    def __getitem__(self, idx):
        column_name = "label"
        y = (
            self.df_all[column_name].iloc[idx] if self.num_labels == 2
            else self.one_hot_encode_labels(np.asarray(self.df_all[column_name].iloc[idx].split(',')).astype(int))
        )

        header = self.df_all["header"].iloc[idx]
        seq = self.df_seq_final.loc[self.df_seq_final["header"] == header, "sequence"].values[0].upper()

        X = self.one_hot_encode(seq)

        return header, seq, torch.tensor(X), torch.tensor(y)


if __name__ == "__main__":
    
    data_path = "../data/Human_Promoters/encode_roadmap_inPromoter"
    dataFinal = DatasetLazyLoadRC(data_path,num_labels=164, rev_complement=True)
    print(dataFinal.df_all["header"])
    print(dataFinal.df_seq_all["header"])
    # Function to extract chromosome number
    def extract_chromosome(region):
        match = re.search(r'chr(\d+|X|Y)', region)
        return match.group(1) if match else None

    df = dataFinal.df_seq_all.drop_duplicates()
    print(df.shape , df['sequence'].unique().shape, df['header'].unique().shape)
    # Apply function to extract chromosome number
    df['chromosome'] = df['header'].apply(extract_chromosome)
    print(df['chromosome'].unique())
    df['chromosome'] = pd.to_numeric(df['chromosome'], errors='coerce')

    # Define chromosome splits

    # Define chromosome sets
    train_chr = np.concatenate([np.arange(1, 15), np.array([18, 19])])  # Chromosomes 1 to 14, and 18 to 19
    val_chr = np.concatenate([np.array([17]), np.arange(20, 22)])  # Chromosomes 17, 20, and 21
    # The test set will contain all chromosomes except for those in train_chr and val_chr
    all_chr = np.arange(1, 23)  # Chromosomes 1 to 22
    test_chr = np.setdiff1d(all_chr, np.concatenate([train_chr, val_chr]))  # Remaining chromosomes
    unique_chromosomes = np.sort(df['chromosome'].unique())
    # Print the sets
    print("Training Set Chromosomes:", train_chr)
    print("Validation Set Chromosomes:", val_chr)
    print("Test Set Chromosomes:", test_chr)

    # Check the chromosomes that are missing from the splits
    all_chromosomes = np.concatenate([train_chr, val_chr, test_chr])
    missing_chromosomes = np.setdiff1d(unique_chromosomes, all_chromosomes)
    print("Missing schromosome" ,  missing_chromosomes)
    # Get indices for each split
    train_indices = df[df['chromosome'].isin(train_chr)].index.tolist()
    valid_indices = df[df['chromosome'].isin(val_chr)].index.tolist()
    test_indices = df[df['chromosome'].isin(test_chr)].index.tolist()
    
    print("Total indices ", len(test_indices) + len(train_indices) + len(valid_indices))
    np.savetxt('../data/Human_Promoters/valid_indices_rev.txt', valid_indices, fmt='%s')
    np.savetxt('../data/Human_Promoters/test_indices_rev.txt', test_indices, fmt='%s')
    np.savetxt('../data/Human_Promoters/train_indices_rev.txt', train_indices, fmt='%s')

    
    