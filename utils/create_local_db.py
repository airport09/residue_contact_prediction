import argparse
import subprocess
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from data.protein_dataset import ProteinDataset


def create_fasta_from_pdbs(pdb_dir, output_fasta):
    dataset = ProteinDataset(dir=pdb_dir)

    all_records = []

    for datapoint in dataset:
        record = SeqRecord(datapoint['residue_codes'], 
                           id=f"{datapoint['id']}_{datapoint['chain_id']}", 
                           description="")
        
        all_records.append(record)

    if all_records:
        SeqIO.write(all_records, output_fasta, "fasta")
        print(f"\nFASTA file created: {output_fasta} (contains {len(all_records)} sequences)")
    else:
        print("No sequences were extracted from the provided PDB files.")

def create_local_blast_db(fasta_file, db_name):
    cmd = ["makeblastdb", "-in", fasta_file, "-dbtype", "prot", "-out", db_name]
    try:
        subprocess.run(cmd, check=True)
        print(f"Local BLAST database '{db_name}' created from {fasta_file}.")
    except subprocess.CalledProcessError as e:
        print("Error creating BLAST database:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_dir",
        type=str,
        help="Dir where PDB files are stored."
    )
    parser.add_argument(
        "--output_fasta",
        type=str,
        help="Output FASTA file"
    )    
    parser.add_argument(
        "--db_name",
        type=str,
        help="Local DB Name"
    )    

    args = parser.parse_args()

    create_fasta_from_pdbs(args.pdb_dir, args.output_fasta)
    create_local_blast_db(args.output_fasta, args.db_name)


