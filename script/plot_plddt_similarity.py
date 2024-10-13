import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_plddt_and_similarity(similarity_data, alphafold_scores, esmfold_scores, omega_scores):
    categories = ['Train', 'Validation', 'Test']

    x = np.arange(len(categories))
    bar_width = 0.1
    spacing = 0.06

    train_positions = x - bar_width - spacing
    valid_positions = x
    test_positions = x + bar_width + spacing

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set(style="whitegrid")

    alphafold_color = '#9966CC'
    esmfold_color = '#FEAF78'
    omega_scores_color = '#84CEB9'

    bars_train = ax.bar(train_positions, alphafold_scores, bar_width, 
                        label='Alphafold', color=alphafold_color)
    bars_valid = ax.bar(valid_positions, esmfold_scores, bar_width, 
                        label='ESMfold', color=esmfold_color)
    bars_test = ax.bar(test_positions, omega_scores, bar_width, 
                       label='OMEGAFold', color=omega_scores_color)

    for bars in [bars_train, bars_valid, bars_test]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)

    ax.axhline(y=similarity_data['train'], color='#FF69B4', linestyle='-', 
               linewidth=2.0, label='Train set Sim.')
    ax.axhline(y=similarity_data['validation'], color='#2F4F4F', linestyle='--', 
               linewidth=2.0, label='Validation set Sim.')
    ax.axhline(y=similarity_data['test'], color='#CD5C5C', linestyle='-.', 
               linewidth=2.0, label='Test set Sim.')

    ax.set_ylabel('plDDT Score / Sequence Similarity (%)', fontsize=12)
    ax.set_ylim(0, 95)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis='y', labelsize=12)
    
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('plddt_and_sim.png', dpi=500)
    plt.show()

if __name__ == '__main__':
    similarity_data = {
        'train': 73.28,
        'validation': 71.58,
        'test': 69.60
    }
    alphafold_scores = [92.4, 92.7, 91.8]
    esmfold_scores = [76.2, 76.0, 76.0]
    omega_scores = [82.7, 84.8, 81.2]
    plot_plddt_and_similarity(
        similarity_data,
        alphafold_scores,
        esmfold_scores,
        omega_scores
    )