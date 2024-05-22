from Bio.Seq import Seq
import re

primers = {
    '27F': 'AGAGTTTGATYMTGGCTCAG',
    '341F': 'CCTACGGGNGGCWGCAG',
    '515F': 'GTGCCAGCMGCCGCGGTAA',
    '939F': 'GAATTGACGGGGGCCCGCACAAG',
    '1115F': 'CAACGAGCGCAACCCT',
    '338R': 'GCTGCCTCCCGTAGGAGT',
    '534R': 'ATTACCGCGGCTGCTGG',
    '785R': 'GACTACHVGGGTATCTAATCC',
    '806R': 'GGACTACHVGGGTWTCTAAT',
    '944R': 'GAATTAAACCACATGCTC',
    '1378R': 'CGGTGTGTACAAGGCCCGGGAACG',
    '1492R': 'TACGGYTACCTTGTTACGACTT'
}

primers_regex = {
    '27F': 'AGAGTTTGAT[CT][AC]TGGCTCAG',
    '341F': 'CCTACGGG[ACGT]GGC[AT]GCAG',
    '515F': 'GTGCCAGC[AC]GCCGCGGTAA',
    '939F': 'GAATTGACGGGGGCCCGCACAAG',
    '1115F': 'CAACGAGCGCAACCCT',
    '338R': 'GCTGCCTCCCGTAGGAGT',
    '534R': 'ATTACCGCGGCTGCTGG',
    '785R': 'GACTAC[ACT][GCA]GGGTATCTAATCC',
    '806R': 'GGACTAC[ACT][GCA]GGGT[AT]TCTAAT',
    '944R': 'GAATTAAACCACATGCTC',
    '1378R': 'CGGTGTGTACAAGGCCCGGGAACG',
    '1492R': 'TACGG[CT]TACCTTGTTACGACTT'
}

def get_primers(start:str = None, end:str = None, seq = None):
    rev_seq = str(Seq(seq).reverse_complement())
    f_primer = re.findall(primers_regex[start], seq)[0]
    r_primer = re.findall(primers_regex[end], rev_seq)[0]
    return (f_primer, r_primer)

def get_primers_bak(start:str = None, end:str = None, seq = None):
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
    if region == 'V1V2':
        f_primer, r_primer = get_primers(start='27F', end='338R', seq=seq)
    if region == 'V1V3':
        f_primer, r_primer = get_primers(start='27F', end='534R', seq=seq)
    elif region == 'V3V4':
        f_primer, r_primer = get_primers(start='341F', end='785R', seq=seq)
    elif region == 'V4':
        f_primer, r_primer = get_primers(start='515F', end='806R', seq=seq)
    elif region == 'V4V5':
        f_primer, r_primer = get_primers(start='515F', end='944R', seq=seq)
    elif region == 'V6V8':
        f_primer, r_primer = get_primers(start='939F', end='1378R', seq=seq)
    elif region == 'V7V9':
        f_primer, r_primer = get_primers(start='1115F', end='1492R', seq=seq)

    try:
        return str(Seq(str(Seq(seq.split(f_primer)[1]).reverse_complement()).split(r_primer)[1]).reverse_complement())
    except:
        return str('ACGT')