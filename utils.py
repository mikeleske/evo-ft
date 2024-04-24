import pandas as pd
from Bio import SeqIO



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


def parse_fasta_file_gtdb(file: str, domain: str, out_file: str = None, max_seq_length: int = None, V3V4_start: int = None, V3V4_end: int = None, num_records: int = None) -> pd.DataFrame :
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
                
                if V3V4_start:
                    seq = seq[V3V4_start:V3V4_end]
                
                dict1 = dict( (col, val) for (col, val) in 
                              zip(columns, [id, taxa, kingdom, phylum, _class, order, family, genus, species, seq_len, seq]))
                rows_list.append(dict1)

    
    df = pd.DataFrame(rows_list, columns=columns)

    if out_file:
        df.to_csv(path_or_buf=out_file, sep=',', index=False)

    return df