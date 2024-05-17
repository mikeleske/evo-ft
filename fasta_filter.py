from utils import parse_fasta_file_gtdb_gzip, parse_fasta_file_mimt_gzip, get_region, get_clean_df

data_file = '.\\MIMT\\MIMt-16S_M2c_24_4.fna.gz'
domain = 'K__Bacteria'
out_file = 'MIMt-16S_M2c_24_4_Genus_top25_sft.csv'
min_seq_length = 1400
max_seq_length = 2000
region = 'V3V4'
level = 'Genus'

top_n_level = 25
top_n_sample = None

##############################################################################

df = parse_fasta_file_mimt_gzip(
    file=data_file,
    domain=domain,
    out_file=out_file,
    #min_seq_length=min_seq_length,
    #max_seq_length=max_seq_length,
)

print('DataFrame after loading:', df.shape)

df = get_clean_df(df=df, level=level, min_seq_len=min_seq_length)
print('DataFrame after cleaning:', df.shape)

if region:
    df['SeqV3V4'] = df['Seq'].apply(lambda x: get_region(region=region, seq=x))

print('DataFrame after loading:', df.shape)

df = df[df['SeqV3V4'] != 'ACGT'].reset_index(drop=True)
print('DataFrame after cleaning V3V4 misalignment:', df.shape)


#df['Len'] = df['SeqV3V4'].apply(lambda x: len(x))
#df = df[df['Len'] <= 470]
#print('DataFrame after cleaning <= 470:', df.shape)

top_n = df[level].value_counts()[:top_n_level].index
if top_n_sample:
    df = df[df[level].isin(top_n)].groupby(level).apply(lambda x: x.sample(n=top_n_sample))
else:
    df = df[df[level].isin(top_n)]
print('DataFrame after top_n filter:', df.shape)


df.to_csv(path_or_buf=out_file, sep=',', index=False)