"""Exemplo de Active Learning com Label Spreading (scikit-learn).

Ideia: treinamos um modelo semi-supervisionado com poucos rotulos e, a cada
rodada, escolhemos os exemplos em que o modelo esta MAIS INCERTO (maior
entropia da distribuicao de rótulos) para "pedir" o rotulo verdadeiro a um
oraculo (aqui simulado pelo proprio y). Esses exemplos entram no conjunto
rotulado e o modelo e treinado de novo.

Baseado no exemplo da documentacao do scikit-learn:
https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html

cd olist > .venv/scripts/activate > uv pip install -r pyproject.toml > python -m module_olist.modeling.active_learning_example
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.semi_supervised import LabelSpreading

# ----------------------------- Configuracao ---------------------------------
SEED = 0
N_AMOSTRAS = 330        # subconjunto do dataset digits
N_ROTULADOS_INICIAL = 40
N_ITERACOES = 5
N_CONSULTAS = 5         # quantos exemplos "rotulamos" por rodada
GAMMA = 0.25
MAX_ITER = 20
ARQUIVO_FIGURA = "active_learning_resultado.png"


def carregar_dados():
    """Embaralha o dataset digits e devolve X, y e as imagens 8x8."""
    digits = load_digits()
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(digits.data))[:N_AMOSTRAS]
    return digits.data[idx], digits.target[idx], digits.images[idx]


def treinar(X, y, indices_rotulados):
    """Treina o LabelSpreading usando apenas os rótulos de `indices_rotulados`."""
    y_parcial = np.full_like(y, -1)  # -1 = sem rotulo
    y_parcial[indices_rotulados] = y[indices_rotulados]
    modelo = LabelSpreading(gamma=GAMMA, max_iter=MAX_ITER)
    modelo.fit(X, y_parcial)
    return modelo


def escolher_mais_incertos(modelo, indices_nao_rotulados, n):
    """Retorna os `n` indices nao rotulados com maior entropia."""
    entropias = entropy(modelo.label_distributions_.T)
    ordem = np.argsort(entropias)[::-1]
    candidatos = ordem[np.isin(ordem, indices_nao_rotulados)]
    return candidatos[:n]


def main():
    X, y, imagens = carregar_dados()

    rotulados = np.arange(N_ROTULADOS_INICIAL)
    nao_rotulados = np.arange(N_ROTULADOS_INICIAL, len(y))

    historico = []  # (n_rotulados, acuracia nos nao rotulados)
    fig, eixos = plt.subplots(
        N_ITERACOES, N_CONSULTAS, figsize=(9, 2 * N_ITERACOES), squeeze=False
    )

    for it in range(N_ITERACOES):
        if len(nao_rotulados) == 0:
            print("Nao restam exemplos sem rotulo.")
            break

        modelo = treinar(X, y, rotulados)
        pred = modelo.transduction_[nao_rotulados]
        verdade = y[nao_rotulados]
        acc = accuracy_score(verdade, pred)
        historico.append((len(rotulados), acc))

        print(f"\n=== Iteração {it} ===")
        print(f"Rotulados: {len(rotulados)} | Nao rotulados: {len(nao_rotulados)}")
        print(f"Acurácia nos nao rotulados: {acc:.3f}")
        if it == N_ITERACOES - 1:
            print(classification_report(verdade, pred))

        # Exemplos mais incertos -> serao rotulados pelo "oraculo"
        escolhidos = escolher_mais_incertos(modelo, nao_rotulados, N_CONSULTAS)
        for col, i in enumerate(escolhidos):
            ax = eixos[it][col]
            ax.imshow(imagens[i], cmap=plt.cm.gray_r, interpolation="none")
            ax.set_title(f"pred: {modelo.transduction_[i]}  real: {y[i]}", fontsize=8)
            ax.axis("off")

        # O oraculo revela os rotulos: movemos os exemplos para o conjunto rotulado
        rotulados = np.concatenate([rotulados, escolhidos])
        nao_rotulados = np.setdiff1d(nao_rotulados, escolhidos)

    fig.suptitle(
        "Active Learning (Label Spreading)\n"
        "Cada linha: os 5 exemplos mais incertos da rodada"
    )
    fig.tight_layout()
    fig.savefig(ARQUIVO_FIGURA, dpi=120)
    print(f"\nFigura salva em {ARQUIVO_FIGURA}")

    print("\nEvolução da acurácia:")
    for n, acc in historico:
        print(f"  {n:3d} rotulos -> {acc:.3f}")

    plt.show()


if __name__ == "__main__":
    main()