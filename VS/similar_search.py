import openbabel
from openbabel import pybel, openbabel
import csv


def calc_similarity(mol1, mol2):
    fp1 = mol1.calcfp(fptype='FP4')
    fp2 = mol2.calcfp(fptype='FP4')
    return fp1 | fp2


def com_search(z3_path, library_path, score_threshold):
    mol1 = sdf1_mols = next(pybel.readfile('sdf', z3_path))
    sdf2_mols = pybel.readfile('sdf', library_path)

    output_csv = open(output_csv_path, "a", newline="")
    csv_writer = csv.writer(output_csv)
    csv_writer.writerow(["SMILES", "Similarity"])

    for mol2 in sdf2_mols:    
        similarity = calc_similarity(mol1, mol2)
        if similarity > score_threshold:
            with open(output_sdf_path, "a") as sdf_file:
                sdf_file.write(mol2.write("sdf"))
            smiles = mol2.write("smiles").strip()
            csv_writer.writerow([smiles, similarity])

    output_csv.close()


if __name__ =='__main__':
    com_path = "./AG-205.mol"  # 先导化合物路径
    library_path ="/home/zhukai/data/compound-library/chemdiv_2021_all_ligprep_1-out.sdf"  # 虚筛文库路径
    output_sdf_path = "output.sdf"  # 分子指纹相似性>score_threshold 的分子储存路径
    output_csv_path = "output.csv"  # 分子指纹相似性>score_threshold 的分子smiles式+相似性得分 >>>> csv文件
    score_threshold = 0.75
    com_search(com_path, library_path,score_threshold)
    print(f"结果已保存到 {output_sdf_path} 和 {output_csv_path}")
