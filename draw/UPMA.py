import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
from rdkit import Chem
import pickle
from tqdm import tqdm
import argparse
from collections import defaultdict

class ConformationAnalyzer:
    def __init__(self):
        self.results = {}
        
    def load_matched_files(self, xyz_folder, sdf_folder=None):
        """加载匹配的XYZ和SDF文件，支持不同原子数的分子"""
        xyz_files = defaultdict(list)
        sdf_files = defaultdict(list) if sdf_folder is not None else None

        # 收集XYZ文件（不变）
        for f in os.listdir(xyz_folder):
            if f.endswith('.xyz'):
                base = os.path.splitext(f)[0]
                xyz_files[base].append(os.path.join(xyz_folder, f))

        # 收集SDF文件（可选，不变）
        if sdf_folder is not None:
            for f in os.listdir(sdf_folder):
                if f.endswith('.sdf'):
                    base = os.path.splitext(f)[0]
                    sdf_files[base].append(os.path.join(sdf_folder, f))

        # 查找匹配文件（不变）
        if sdf_folder is not None:
            common_bases = set(xyz_files.keys()) & set(sdf_files.keys())
            if not common_bases:
                raise ValueError("No matching XYZ and SDF files found!")
        else:
            common_bases = set(xyz_files.keys())
        print(f"Found {len(common_bases)} molecules to analyze")

        xyz_confs = []
        sdf_confs = [] if sdf_folder is not None else None
        labels = []
        mol_indices = []
        atom_counts = []  # 记录每个构象的原子数

        for mol_idx, base in enumerate(tqdm(common_bases)):
            # 加载XYZ构象
            for xyz_path in xyz_files[base]:
                try:
                    xyz_mol_confs = read(xyz_path, index=':')
                except Exception as e:
                    print(f"[ERROR] reading {xyz_path}: {e}")
                    continue

                for conf in xyz_mol_confs:
                    pos = conf.get_positions()
                    atom_counts.append(len(pos))  # 记录原子数
                    # 不再检查长度一致性
                    xyz_confs.append(pos.flatten())
                    labels.append(base)
                    mol_indices.append(mol_idx)

            # 加载SDF构象（可选）
            if sdf_folder is not None:
                for sdf_path in sdf_files[base]:
                    suppl = Chem.SDMolSupplier(sdf_path)
                    for mol in suppl:
                        if mol is None:
                            continue
                        for conf in mol.GetConformers():
                            pos = conf.GetPositions()
                            # 确保SDF和XYZ的原子数匹配（同分子内）
                            if len(pos) != atom_counts[-1]:  # 与最后一个XYZ构象比较
                                print(f"[WARNING] {base}: SDF atom count {len(pos)} != XYZ {atom_counts[-1]}")
                                continue
                            sdf_confs.append(pos.flatten())

        # 打印统计信息
        print("\nMolecule statistics:")
        df_stats = pd.DataFrame({
            'molecule': labels,
            'atom_count': atom_counts
        })
        print(df_stats.groupby('molecule').agg({'atom_count': ['count', 'min', 'max']}))

        return (
            xyz_confs,
            sdf_confs if sdf_folder is not None else None,
            np.array(labels),
            np.array(mol_indices),
            np.array(atom_counts)
        )

    def analyze_all_conformations_together(self, X_list, labels, atom_counts, prefix="", output_dir="output"):
        """处理不规则形状的构象数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 确定最大原子数
        max_atoms = max(atom_counts)
        max_dims = max_atoms * 3  # 3D坐标
        
        # 2. 补零对齐所有构象
        padded_confs = []
        for coords, n_atoms in zip(X_list, atom_counts):
            padded = np.zeros(max_dims)
            actual_length = n_atoms * 3
            padded[:actual_length] = coords[:actual_length]  # 截断保护
            padded_confs.append(padded)
        
        X_padded = np.vstack(padded_confs)
        
        # 3. 全局标准化
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_padded)
        
        # 4. 降维分析
        print(f"\nAnalyzing {len(X_padded)} conformations from {len(np.unique(labels))} molecules...")
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_standardized)
        
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_padded)//3))
        X_tsne = tsne.fit_transform(X_standardized)
        
        # UMAP
        reducer = umap.UMAP(n_components=2, n_neighbors=min(15, len(X_padded)//3))
        X_umap = reducer.fit_transform(X_standardized)
        
        # 5. 可视化
        self._plot_separate_results(
            pca=X_pca,
            tsne=X_tsne,
            umap=X_umap,
            labels=labels,
            prefix=prefix,
            output_dir=output_dir
        )
        
        # 6. 保存结果
        results = {
            'pca': X_pca,
            'tsne': X_tsne,
            'umap': X_umap,
            'labels': labels,
            'atom_counts': atom_counts
        }
        
        df_results = pd.DataFrame({
            'molecule': labels,
            'atom_count': atom_counts,
            'pca_1': X_pca[:, 0],
            'pca_2': X_pca[:, 1],
            'tsne_1': X_tsne[:, 0],
            'tsne_2': X_tsne[:, 1],
            'umap_1': X_umap[:, 0],
            'umap_2': X_umap[:, 1]
        })
        
        return results, df_results

    from matplotlib.colors import ListedColormap
    import seaborn as sns


    def _plot_separate_results(self, pca, tsne, umap, labels, prefix, output_dir): 
        def get_distinct_colors(n):
            """ 获取n个区别明显的颜色 """
            if n <= 20:
                base = plt.get_cmap('tab20').colors[:n]
            elif n <= 40:
                base = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)
                base = base[:n]
            elif n <= 60:
                base = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + list(plt.get_cmap('tab20c').colors)
                base = base[:n]
            else:
                # 超过60，可以考虑重复颜色或其它方法
                base = sns.color_palette("hls", n_colors=n)
            return base
        unique_labels = np.unique(labels)
        n_colors = len(unique_labels)
        
        colors = get_distinct_colors(n_colors)
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]

        def plot_and_save(data, title, filename):
            fig, ax = plt.subplots(figsize=(9, 7))
            scatter = ax.scatter(data[:, 0], data[:, 1], 
                                c=colors, s=250, alpha=0.85,
                                edgecolors='w', linewidths=0.5)
            ax.set_title(f"{prefix}{title}", fontsize=14, pad=20)
            ax.set_xlabel(f"{title}1", fontsize=11)
            ax.set_ylabel(f"{title}2", fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.2)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{prefix}{filename}.png", dpi=350, bbox_inches='tight')
            plt.close()

        plot_and_save(pca, "PCA", "pca_plot")
        plot_and_save(tsne, "t-SNE", "tsne_plot")
        plot_and_save(umap, "UMAP", "umap_plot")
    
    def compare_generated_confs(self, xyz_confs, sdf_confs, labels, mol_indices, atom_counts, output_dir):
        """更新后的比较方法，处理atom_counts"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析XYZ构象（传入atom_counts）
        print("\nAnalyzing XYZ conformations...")

        xyz_results, xyz_df = self.analyze_all_conformations_together(
            X_list=xyz_confs,  # 原始坐标列表（不同长度）
            labels=labels,
            atom_counts=atom_counts,
            prefix="XYZ_",
            output_dir=os.path.join(output_dir, "global_analysis")
        )
                    
        # 如果有SDF构象，进行分析和比较
        if sdf_confs is not None:
            # 确保构象数量匹配
            if len(xyz_confs) != len(sdf_confs):
                print("Warning: XYZ and SDF conformations count mismatch! Only analyzing XYZ files.")
                sdf_confs = None
            else:
                # 分析SDF构象
                print("\nAnalyzing SDF conformations...")
                sdf_results, sdf_df = self.analyze_conformations(
                    sdf_confs, labels, "SDF ", os.path.join(output_dir, "sdf_analysis"))
                
                # 比较分析
                print("\nComparing XYZ and SDF conformations...")
                self._plot_comparison(xyz_results, sdf_results, output_dir)
                
                # 保存比较数据
                comparison_data = {
                    'xyz': xyz_results,
                    'sdf': sdf_results,
                    'labels': labels
                }
                
                with open(os.path.join(output_dir, 'comparison_results.pkl'), 'wb') as f:
                    pickle.dump(comparison_data, f)
        
        # 如果没有SDF构象，只保存XYZ结果
        else:
            # 保存XYZ结果
            xyz_data = {
                'xyz': xyz_results,
                'labels': labels
            }
            
            with open(os.path.join(output_dir, 'xyz_results.pkl'), 'wb') as f:
                pickle.dump(xyz_data, f)
        
        print(f"\nAll results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze molecular conformations from XYZ files with optional SDF comparison.')
    parser.add_argument('--xyz', type=str, required=True, help='Path to XYZ files folder')
    parser.add_argument('--sdf', type=str, required=False, default=None, help='Path to SDF files folder (optional)')
    parser.add_argument('--output', type=str, default='conformation_results', help='Output directory')
    args = parser.parse_args()
    
    analyzer = ConformationAnalyzer()
    
    # 加载文件（现在接收5个返回值）
    xyz_confs, sdf_confs, labels, mol_indices, atom_counts = analyzer.load_matched_files(args.xyz, args.sdf)
    
    # 分析（传入atom_counts）
    analyzer.compare_generated_confs(
        xyz_confs=xyz_confs,
        sdf_confs=sdf_confs,
        labels=labels,
        mol_indices=mol_indices,
        atom_counts=atom_counts,
        output_dir=args.output
    )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
