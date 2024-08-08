import pandas as pd
import requests
import re
import os
import json

# get the data from excel
def read_excel_table(file_name, sheet_index):
    df = pd.read_excel(file_name, sheet_name=sheet_index)
    return df

def get_column_data(file_name, sheet_index, column_index):
    df = read_excel_table(file_name, sheet_index)
    column_data = df.iloc[:, column_index].tolist()
    return column_data

# get sequence from specific gene
def get_gene_info(gene_id):
    url = f"https://rest.ensembl.org/lookup/id/{gene_id}?content-type=application/json"
    response = requests.get(url)
    if response.ok:
        return response.json()
    else:
        raise Exception("Error fetching gene information from Ensembl API")

def get_sequence(species, chromosome, start, end):
    region = f"{chromosome}:{start}-{end}"
    url = f"https://rest.ensembl.org/sequence/region/{species}/{region}?content-type=text/plain"
    response = requests.get(url)
    if response.ok:
        return response.text
    else:
        raise Exception("Error fetching sequence from Ensembl API")

# get promoter region
def fetch_gene_info(gene_id, species='human'):
    """
    Fetch gene information from Ensembl for a given gene ID.
    
    Args:
    gene_id (str): Ensembl gene ID.
    species (str): Species name. Default is 'human'.
    
    Returns:
    dict: Gene information including chromosome, start, end, and strand.
    """
    server = "https://rest.ensembl.org"
    ext = f"/lookup/id/{gene_id}?expand=1"

    headers = { "Content-Type" : "application/json"}

    r = requests.get(server+ext, headers=headers)

    if not r.ok:
        r.raise_for_status()
        return None

    return r.json()

def get_promoter_region(gene_info, upstream=2000, downstream=200):
    """
    Determine the promoter region based on gene information.
    
    Args:
    gene_info (dict): Gene information including chromosome, start, end, and strand.
    upstream (int): Number of bases upstream of the TSS to include in the promoter region.
    downstream (int): Number of bases downstream of the TSS to include in the promoter region.
    
    Returns:
    str: Genomic region representing the promoter.
    """
    chrom = gene_info['seq_region_name']
    strand = gene_info['strand']
    
    if strand == 1:  # Positive strand
        tss = gene_info['start']
        promoter_start = tss - upstream
        promoter_end = tss + downstream
    else:  # Negative strand
        tss = gene_info['end']
        promoter_start = tss - downstream
        promoter_end = tss + upstream

    return f"{chrom}:{promoter_start}-{promoter_end}"

def find_max_min(promoter_start, promoter_end, enhancer_start, enhancer_end):
    # Convert all inputs to integers
    numbers = list(map(int, [promoter_start, promoter_end, enhancer_start, enhancer_end]))
    
    maximum = max(numbers)
    minimum = min(numbers)
    
    return maximum, minimum

if __name__ == "__main__":
    file_name = "c:/Users/13772/OneDrive/Desktop/BU/bio/Gasperini/mmc2.xlsx"
    sheet_index = 2

    ENSG_index = 1
    enhancer_start_index = 9
    enhancer_end_index = 10

    bool_index = 7

    # # enhancer start/end form excel
    bool_data = get_column_data(file_name, sheet_index, bool_index)
    
    ENSG_data_raw = get_column_data(file_name, sheet_index, ENSG_index)
    enhancer_start_data_raw = get_column_data(file_name, sheet_index, enhancer_start_index)
    enhancer_end_data_raw = get_column_data(file_name, sheet_index, enhancer_end_index)

    ENSG_data = []
    enhancer_start_data = []
    enhancer_end_data = []

    for i in range(len(bool_data)):
        if bool_data[i]: 
            ENSG_data.append(ENSG_data_raw[i])
            enhancer_start_data.append(enhancer_start_data_raw[i])
            enhancer_end_data.append(enhancer_end_data_raw[i])

    # # print(ENSG_data)
    # print(enhancer_start_data)
    # print(enhancer_end_data)


    # # promoter start/emd from ensembl
    promoter_start_data = []
    promoter_end_data = []
    chromosome_data = []
    for gene_id in ENSG_data:
        gene_info = fetch_gene_info(gene_id)
        if gene_info:
            promoter_region = get_promoter_region(gene_info)
            # print(f"Promoter region for gene {gene_id}: {promoter_region}")
            pattern_promoter = r":(\d+)-(\d+)"
            pattern_chr = r"([^:]+):"
            match_promoter = re.search(pattern_promoter, promoter_region)
            match_chr = re.search(pattern_chr, promoter_region)
            if match_promoter:
                promoter_start_index = match_promoter.group(1)
                promoter_end_index = match_promoter.group(2)
                chr_index = match_chr.group(1)
                promoter_start_data.append(promoter_start_index)
                promoter_end_data.append(promoter_end_index)
                chromosome_data.append(chr_index)
                # print(f"promoter_start: {promoter_start_index}")
                # print(f"promoter_end: {promoter_end_index}")
            else:
                print("match nothing")
        else:
            print(f"Failed to fetch information for gene {gene_id}")
    
    # print(promoter_start_data)
    # print(promoter_end_data)
    # print(chromosome_data)


    # get sequence between promoter/enhancer
    output_folder = 'output_files_mmc2'
    os.makedirs(output_folder, exist_ok=True)

    for i in range(len(ENSG_data)):
        gene_id = ENSG_data[i]
        chromosome = chromosome_data[i]
        region_end, region_start = find_max_min(promoter_start_data[i], promoter_end_data[i], enhancer_start_data[i], enhancer_end_data[i])
        species = "homo sapiens"
        
        try:
            gene_info = get_gene_info(gene_id)
            sequence = get_sequence(species, chromosome, region_start, region_end)
            
            gene_info_str = json.dumps(gene_info, indent=4) if isinstance(gene_info, dict) else str(gene_info)
            sequence_str = json.dumps(sequence, indent=4) if isinstance(sequence, dict) else str(sequence)
            
            filename = f"{chromosome}_{region_start}_{region_end}.txt"
            file_path = os.path.join(output_folder, filename)
            
            with open(file_path, 'w') as file:
                file.write("Gene Information:\n")
                file.write(gene_info_str + "\n\n")
                file.write(f"Sequence from chromosome {chromosome} between {region_start} and {region_end}:\n")
                file.write(sequence_str + "\n")
            
            print(f"信息已保存到 {file_path}")

        except Exception as e:
            print(f"错误: {e}")


    



