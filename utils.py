import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import gzip


def parse_fasta_file_silva(file: str, domain: str, out_file: str = None, max_seq_length: int = None, num_records: int = None) -> pd.DataFrame :
    """
        tbd
    """

    rows_list = []
    columns = ['ID','Taxa','Kingdom','Phylum','Class','Order','Family','Genus','Species','SeqLen','Seq']

    fasta_file = SeqIO.parse(file, 'fasta')

    while len(rows_list) < num_records:
        rec = next(fasta_file)
        #if str(rec.description).split()[1].split(';')[1] in ['Firmicutes', 'Proteobacteria', 'Actinobacteriota']:
        if str(rec.description).split()[1].startswith(domain):
            description = str(rec.description)
            id = description.split()[0]
            taxa = description.rsplit(id)[1].strip().replace(';', '~')
            taxa_split = taxa.split('~')

            if len(taxa_split) == 7:
                kingdom, phylum, _class, order, family, genus, species = taxa_split
                seq = str(rec.seq.upper().back_transcribe())
                seq_len = len(seq)

                if max_seq_length:
                    seq = seq[:max_seq_length]
                
                dict1 = dict( (col, val) for (col, val) in 
                              zip(columns, [id, taxa, kingdom, phylum, _class, order, family, genus, species, seq_len, seq]))
                rows_list.append(dict1)

    
    df = pd.DataFrame(rows_list, columns=columns)

    if out_file:
        df.to_csv(path_or_buf=out_file, sep=',', index=False)

    return df


def parse_fasta_file_gtdb(file: str, domain: str, out_file: str = None, min_seq_length: int = None, max_seq_length: int = None) -> pd.DataFrame :
    """
        tbd
    """

    rows_list = []
    columns = ['ID','Taxa','Kingdom','Phylum','Class','Order','Family','Genus','Species','SeqLen','Seq']

    fasta_file = SeqIO.parse(file, 'fasta')

    for rec in fasta_file:
        if str(rec.description).split()[1].startswith(domain):
            description = str(rec.description)
            id = description.split()[0]
            taxa = description.rsplit(id)[1].split(' [')[0].replace(';', '~').strip()
            taxa_split = taxa.split('~')

            if len(taxa_split) == 7:
                kingdom, phylum, _class, order, family, genus, species = taxa_split
                seq = str(rec.seq)
                seq_len = len(seq)

                if max_seq_length:
                    seq = seq[:max_seq_length]
                
                dict1 = dict( (col, val) for (col, val) in 
                              zip(columns, [id, taxa, kingdom, phylum, _class, order, family, genus, species, seq_len, seq]))
                rows_list.append(dict1)

    
    df = pd.DataFrame(rows_list, columns=columns)

    if min_seq_length:
        df = df[df['SeqLen'] >= min_seq_length].reset_index(drop=True)

    if out_file:
        df.to_csv(path_or_buf=out_file, sep=',', index=False)

    return df


def parse_fasta_file_gtdb_gzip(file: str, domain: str, out_file: str = None, min_seq_length: int = None, max_seq_length: int = None) -> pd.DataFrame :
    """
        tbd
    """

    rows_list = []
    columns = ['ID','Taxa','Kingdom','Phylum','Class','Order','Family','Genus','Species','SeqLen','Seq']

    #fasta_file = SeqIO.parse(file, 'fasta')

    with gzip.open(file, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            if str(rec.description).split()[1].startswith(domain):
                description = str(rec.description)
                id = description.split()[0]
                taxa = description.rsplit(id)[1].split(' [')[0].replace(';', '~').strip()
                taxa_split = taxa.split('~')
    
                if len(taxa_split) == 7:
                    kingdom, phylum, _class, order, family, genus, species = taxa_split
                    seq = str(rec.seq)
                    seq_len = len(seq)
    
                    if max_seq_length:
                        seq = seq[:max_seq_length]
                    
                    dict1 = dict( (col, val) for (col, val) in 
                                  zip(columns, [id, taxa, kingdom, phylum, _class, order, family, genus, species, seq_len, seq]))
                    rows_list.append(dict1)

    
    df = pd.DataFrame(rows_list, columns=columns)

    if min_seq_length:
        df = df[df['SeqLen'] >= min_seq_length].reset_index(drop=True)

    if out_file:
        df.to_csv(path_or_buf=out_file, sep=',', index=False)

    return df


def parse_fasta_file_mimt_gzip(file: str, domain: str, out_file: str = None, min_seq_length: int = None, max_seq_length: int = None) -> pd.DataFrame :
    """
        tbd
    """

    rows_list = []
    columns = ['ID','Taxa','Kingdom','Phylum','Class','Order','Family','Genus','Species','SeqLen','Seq']

    with gzip.open(file, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            if str(rec.description).split()[1].startswith(domain):

                description = str(rec.description)
                id, taxa = description.split('\t')

                taxa = taxa.replace('K__', '').replace(' P__', '').replace(' C__', '').replace(' O__', '').replace(' F__', '').replace(' G__', '').replace(' S__', '')
                kingdom, phylum, _class, order, family, genus, species, _ = [level for level in taxa.split(';')]

                seq, seq_len = str(rec.seq), len(str(rec.seq))
    
                if max_seq_length:
                    seq = seq[:max_seq_length]
                    
                dict1 = dict( (col, val) for (col, val) in 
                               zip(columns, [id, taxa, kingdom, phylum, _class, order, family, genus, species, seq_len, seq]))
                rows_list.append(dict1)

    
    df = pd.DataFrame(rows_list, columns=columns)

    if min_seq_length:
        df = df[df['SeqLen'] >= min_seq_length].reset_index(drop=True)

    if out_file:
        df.to_csv(path_or_buf=out_file, sep=',', index=False)

    return df


def get_primers(start:str = None, end:str = None, seq = None):
    f_primer = None
    r_primer = None
    rev_seq = Seq(seq).reverse_complement()
    
    if start == '341F':
        primer = 'CCTACGGGNGGCWGCAG'

        for W in ['A', 'T']:
            for N in ['A', 'C', 'G', 'T']:
                iter_primer = primer.replace('W',W).replace('N',N)
                if iter_primer in seq:
                    f_primer = iter_primer
                    break
    
    if end == '785R':
        primer = 'GACTACHVGGGTATCTAATCC'

        for H in ['A', 'C', 'T']:
            for V in ['G', 'C', 'A']:
                iter_primer = primer.replace('H',H).replace('V',V)
                if iter_primer in rev_seq:
                    r_primer = iter_primer
                    break

    return (f_primer, r_primer)


def get_region(region:str = None, seq:str = None):
    if region == 'V3V4':
        f_primer, r_primer = get_primers(start='341F', end='785R', seq=seq)
        try:
            return str(Seq(str(Seq(seq.split(f_primer)[1]).reverse_complement()).split(r_primer)[1]).reverse_complement())
        except:
            return str('ACGT')
        
def get_clean_df(df: pd.DataFrame = None, level: str = None, min_seq_len: int = None):
    __list = list(df[level].unique())
    __list_clean = [ level for level in __list if not(level[4].isupper() or '-' in level)]
    df = df[df[level].isin(__list_clean)]
    if min_seq_len:
        df = df[df['SeqLen'] >= min_seq_len]
    return df