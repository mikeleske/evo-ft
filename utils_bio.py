from Bio.Seq import Seq

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