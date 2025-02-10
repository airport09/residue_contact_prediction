import os
import argparse
import tempfile
import subprocess
from Bio.Blast import NCBIXML

def run_local_blast_query(seq, db_name, top_n=10):

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True) as output_xml, \
        tempfile.NamedTemporaryFile(mode='w+', suffix='.fasta', delete=True) as query_file:
        query_file.write(">Query\n")
        query_file.write(seq + "\n")
        query_file.flush()
        query_file.seek(0)

        cmd = [
            "blastp",
            "-query", query_file.name,
            "-db", db_name,
            "-out", output_xml.name,
            "-outfmt", "5",
            "-max_target_seqs", str(top_n)
        ]
        
        # print("Running local BLAST search...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running local BLAST:", e)
            return None

        with open(output_xml.name) as result_handle:
            blast_record = NCBIXML.read(result_handle)
        
        return blast_record

def get_top_seqs(seq, id_, db_name, allow_same_ids=True, top_n=3, max_exp=0.1):
    blast_record = run_local_blast_query(seq, db_name, top_n * 5)

    if not blast_record.alignments:
        return [None] * top_n
    
    protein_id = id_.split('_')[0]
    
    top_seqs = []
    top_seqs_proteins = []

    for alignment in blast_record.alignments:
        hsp = alignment.hsps[0]
        rec_id_ = alignment.hit_def
        rec_protein_id = rec_id_.split('_')[0]
        exp = hsp.expect

        if protein_id == rec_protein_id:
            continue

        if not allow_same_ids and rec_protein_id in top_seqs_proteins:
            continue

        if exp > max_exp:
            continue

        top_seqs.append(rec_id_)
        top_seqs_proteins.append(rec_protein_id)

        if len(top_seqs) == top_n:
            break

    top_seqs.extend([None] * (top_n - len(top_seqs)))

    return top_seqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=str,
        help="Protein sequence to search for."
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Id of a protein."
    )
    parser.add_argument(
        "--local_db_name",
        type=str,
        help="Local DB Name"
    )      

    args = parser.parse_args()

    top_seqs = get_top_seqs(seq=args.seq,
                            id_=args.id,
                            allow_same_ids=False,
                            db_name=args.local_db_name)
    
    print('TOP SEQS:\n', top_seqs)

if __name__ == "__main__":
    main()
