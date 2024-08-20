import os
import re
from collections import defaultdict

def extract_dna_sequence(file_content):
    sequence_match = re.search(r'Sequence from chromosome [\dA-Z]+ between \d+ and \d+:\s*([\s\w]+)', file_content)

    if sequence_match:
        return sequence_match.group(1).replace('\n', '')
    return None

def process_files(input_folder, output_folder):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_to_process = defaultdict(lambda: defaultdict(str))

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            match = re.match(r'([\dA-Z]+)_(\d+)_(\d+)\.txt', filename)
            if match:
                chromosome, promoter, enhancer = match.groups()
                file_path = os.path.join(input_folder, filename)
                
                with open(file_path, 'r') as file:
                    content = file.read()
                
                dna_sequence = extract_dna_sequence(content)
                if dna_sequence:
                    current_max_enhancer = files_to_process[chromosome].get(promoter, (0, ''))[0]
                    if int(enhancer) > current_max_enhancer:
                        files_to_process[chromosome][promoter] = (int(enhancer), dna_sequence)

    for chromosome, promoters in files_to_process.items():
        for promoter, (enhancer, sequence) in promoters.items():
            output_filename = f'{chromosome}_{promoter}_{enhancer}.txt'
            output_path = os.path.join(output_folder, output_filename)
            with open(output_path, 'w') as output_file:
                output_file.write(sequence)
                print(f'文件 {output_filename} 已保存。')

input_folder = 'data/dataset/output_files_Fulco_3a'
output_folder = 'data/DNA_sequence/output_files_Fulco_3a'
process_files(input_folder, output_folder)

