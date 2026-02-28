"""
Evolving Blackjack Strategy with Genetic Algorithm
===================================================
Encodes hit/stand strategies as 260-bit chromosomes and evolves them
over generations using selection, crossover, and mutation.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Hard hands: player totals 4–20 (17 values) x dealer upcards Ace–10 (10 values)
HARD_TOTALS   = list(range(4, 21))        # 17 values
SOFT_TOTALS   = list(range(12, 21))       # soft 12–20 (9 values: A+A through A+9)
DEALER_CARDS  = list(range(1, 11))        # Ace=1, 2–9, 10

HARD_BITS = len(HARD_TOTALS) * len(DEALER_CARDS)   # 170
SOFT_BITS = len(SOFT_TOTALS) * len(DEALER_CARDS)   # 90
CHROM_LEN = HARD_BITS + SOFT_BITS                  # 260

# GA parameters
POP_SIZE      = 100
N_GENERATIONS = 100
N_HANDS       = 2000
MUTATION_RATE = 0.01
ELITES        = 2

# ─────────────────────────────────────────────
# CHROMOSOME HELPERS
# ─────────────────────────────────────────────

def random_chromosome():
    return np.random.randint(0, 2, CHROM_LEN, dtype=np.int8)

def get_decision(chrom, player_total, is_soft, dealer_upcard):
    """
    Look up hit(1) or stand(0) from chromosome.
    dealer_upcard: 1=Ace, 2-10.
    """
    d_idx = DEALER_CARDS.index(dealer_upcard)
    if is_soft and player_total in SOFT_TOTALS:
        p_idx = SOFT_TOTALS.index(player_total)
        bit   = HARD_BITS + p_idx * len(DEALER_CARDS) + d_idx
    else:
        # Hard hand; clamp to table range
        clamped = max(4, min(20, player_total))
        p_idx   = HARD_TOTALS.index(clamped)
        bit     = p_idx * len(DEALER_CARDS) + d_idx
    return int(chrom[bit])

# ─────────────────────────────────────────────
# BLACKJACK SIMULATION
# ─────────────────────────────────────────────

def card_value(card):
    """Face value for scoring; Ace handled separately."""
    return min(card, 10)

def build_deck():
    """Standard 52-card deck. Cards stored as 1–13 (1=Ace, 11=J, 12=Q, 13=K)."""
    deck = list(range(1, 14)) * 4
    random.shuffle(deck)
    return deck

def hand_value(cards):
    """
    Returns (total, is_soft).
    is_soft = True if an Ace is being counted as 11.
    """
    total = sum(card_value(c) for c in cards)
    n_aces = sum(1 for c in cards if c == 1)
    is_soft = False
    if n_aces > 0 and total + 10 <= 21:
        total  += 10
        is_soft = True
    return total, is_soft

def dealer_upcard_index(card):
    """Convert raw card to 1-10 index used in strategy."""
    return min(card_value(card), 10)

def play_hand(chrom, deck, pos):
    """
    Play one hand starting at deck position pos.
    Returns (result, new_pos) where result is 1=win, 0=loss, 0.5=tie.
    """
    # Deal: player gets cards[0], cards[2]; dealer gets cards[1], cards[3]
    if pos + 4 > len(deck):
        return None, pos  # not enough cards

    p_cards = [deck[pos], deck[pos+2]]
    d_cards = [deck[pos+1], deck[pos+3]]
    pos += 4

    # Check for natural blackjack
    p_val, p_soft = hand_value(p_cards)
    d_val, d_soft = hand_value(d_cards)

    p_bj = (p_val == 21 and len(p_cards) == 2)
    d_bj = (d_val == 21 and len(d_cards) == 2)

    if p_bj and d_bj:
        return 0.5, pos
    if p_bj:
        return 1, pos
    if d_bj:
        return 0, pos

    # Player's turn
    upcard = dealer_upcard_index(d_cards[0])
    while True:
        p_val, p_soft = hand_value(p_cards)
        if p_val >= 21:
            break
        decision = get_decision(chrom, p_val, p_soft, upcard)
        if decision == 0:   # stand
            break
        # hit
        if pos >= len(deck):
            break
        p_cards.append(deck[pos]); pos += 1

    p_val, _ = hand_value(p_cards)
    if p_val > 21:
        return 0, pos   # player busts

    # Dealer's turn: stand on soft 17
    while True:
        d_val, d_soft = hand_value(d_cards)
        if d_val > 17:
            break
        if d_val == 17 and not d_soft:
            break
        if pos >= len(deck):
            break
        d_cards.append(deck[pos]); pos += 1

    d_val, _ = hand_value(d_cards)
    if d_val > 21:
        return 1, pos   # dealer busts

    if p_val > d_val:
        return 1, pos
    elif p_val < d_val:
        return 0, pos
    else:
        return 0.5, pos

def evaluate_fitness(chrom):
    """Simulate N_HANDS and return win fraction."""
    wins = 0.0
    total = 0
    deck = build_deck()
    pos  = 0

    while total < N_HANDS:
        if pos + 20 > len(deck):   # reshuffle when low
            deck = build_deck()
            pos  = 0
        result, pos = play_hand(chrom, deck, pos)
        if result is None:
            deck = build_deck(); pos = 0
            continue
        wins  += result
        total += 1

    return wins / total

# ─────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────

def roulette_select(population, fitnesses):
    """Roulette wheel selection — returns one individual."""
    total = sum(fitnesses)
    pick  = random.uniform(0, total)
    cumul = 0.0
    for chrom, fit in zip(population, fitnesses):
        cumul += fit
        if cumul >= pick:
            return chrom.copy()
    return population[-1].copy()

def single_point_crossover(p1, p2):
    point = random.randint(1, CHROM_LEN - 1)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(chrom):
    mask = np.random.random(CHROM_LEN) < MUTATION_RATE
    chrom[mask] ^= 1
    return chrom

def evolve(population, fitnesses):
    # Sort by fitness descending
    order = np.argsort(fitnesses)[::-1]
    fitnesses_sorted = [fitnesses[i] for i in order]
    pop_sorted       = [population[i].copy() for i in order]

    next_gen = []
    # Elitism
    for i in range(ELITES):
        next_gen.append(pop_sorted[i].copy())

    # Fill rest
    while len(next_gen) < POP_SIZE:
        p1 = roulette_select(pop_sorted, fitnesses_sorted)
        p2 = roulette_select(pop_sorted, fitnesses_sorted)
        c1, c2 = single_point_crossover(p1, p2)
        next_gen.append(mutate(c1))
        if len(next_gen) < POP_SIZE:
            next_gen.append(mutate(c2))

    return next_gen

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def run():
    print("Initializing population...")
    population = [random_chromosome() for _ in range(POP_SIZE)]

    history = {"min": [], "max": [], "mean": [], "median": []}

    for gen in range(N_GENERATIONS):
        fitnesses = [evaluate_fitness(c) for c in population]

        history["min"].append(np.min(fitnesses))
        history["max"].append(np.max(fitnesses))
        history["mean"].append(np.mean(fitnesses))
        history["median"].append(np.median(fitnesses))

        best = max(fitnesses)
        print(f"Gen {gen+1:3d}/{N_GENERATIONS}  "
              f"max={best:.4f}  mean={np.mean(fitnesses):.4f}  "
              f"min={np.min(fitnesses):.4f}")

        population = evolve(population, fitnesses)

    # Final fitness eval
    final_fitnesses = [evaluate_fitness(c) for c in population]
    print(f"\nFinal best fitness: {max(final_fitnesses):.4f}")

    return population, final_fitnesses, history

# ─────────────────────────────────────────────
# FIGURE 1: Fitness over generations
# ─────────────────────────────────────────────

def plot_fitness(history, save_path="fitness_over_generations.png"):
    gens = range(1, len(history["max"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0f0f14")
    fig.patch.set_facecolor("#0f0f14")

    colors = {"max": "#f7a26a", "mean": "#7c6af7", "median": "#4ade80", "min": "#f87171"}
    labels = {"max": "Max", "mean": "Mean", "median": "Median", "min": "Min"}

    for key in ["max", "mean", "median", "min"]:
        ax.plot(gens, history[key], color=colors[key], linewidth=2,
                label=labels[key], alpha=0.9)

    ax.axhline(0.495, color="white", linewidth=1, linestyle="--", alpha=0.4,
               label="Theoretical optimum (~49.5%)")

    ax.set_xlabel("Generation", color="white", fontsize=12)
    ax.set_ylabel("Fitness (win fraction)", color="white", fontsize=12)
    ax.set_title("Blackjack GA — Fitness Over Generations", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(facecolor="#1a1a24", edgecolor="#333344", labelcolor="white", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(color="#222230", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# FIGURE 2: Strategy heatmap
# ─────────────────────────────────────────────

def plot_strategy_heatmap(population, save_path="strategy_heatmap.png"):
    """
    For each (player total, dealer upcard) cell, show % of population recommending HIT.
    100% hit = red, 100% stand = blue.
    """
    n = len(population)

    # Hard hand matrix: rows=player totals (4-20), cols=dealer (A,2–10)
    hard_matrix = np.zeros((len(HARD_TOTALS), len(DEALER_CARDS)))
    soft_matrix = np.zeros((len(SOFT_TOTALS), len(DEALER_CARDS)))

    for chrom in population:
        for pi, pt in enumerate(HARD_TOTALS):
            for di, dc in enumerate(DEALER_CARDS):
                hard_matrix[pi, di] += get_decision(chrom, pt, False, dc)
        for pi, pt in enumerate(SOFT_TOTALS):
            for di, dc in enumerate(DEALER_CARDS):
                soft_matrix[pi, di] += get_decision(chrom, pt, True, dc)

    hard_matrix /= n
    soft_matrix /= n

    # Color: 0=stand=blue, 1=hit=red
    cmap = mcolors.LinearSegmentedColormap.from_list("bwr", ["#3b82f6", "#e5e7eb", "#ef4444"])

    dealer_labels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    hard_labels   = [str(t) for t in HARD_TOTALS]
    soft_labels   = [f"A+{t-11}" for t in SOFT_TOTALS]  # A+1 through A+9

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    fig.patch.set_facecolor("#0f0f14")

    def draw_heatmap(ax, matrix, row_labels, title):
        ax.set_facecolor("#0f0f14")
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(DEALER_CARDS)))
        ax.set_xticklabels(dealer_labels, color="white", fontsize=10)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, color="white", fontsize=10)
        ax.set_xlabel("Dealer Upcard", color="white", fontsize=11)
        ax.set_ylabel("Player Total", color="white", fontsize=11)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors="white")

        # Annotate cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val  = matrix[i, j]
                text = "H" if val >= 0.5 else "S"
                tc   = "white" if abs(val - 0.5) > 0.2 else "black"
                ax.text(j, i, f"{text}\n{val:.0%}", ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")

        return im

    im1 = draw_heatmap(axes[0], hard_matrix, hard_labels, "Hard Hands")
    im2 = draw_heatmap(axes[1], soft_matrix, soft_labels, "Soft Hands")

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation="horizontal", pad=0.05,
                        fraction=0.03, aspect=40)
    cbar.set_label("% of population recommending HIT", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(tick_vals)
    cbar.ax.set_xticklabels([f"{int(v*100)}%" for v in tick_vals], color="white")
    cbar.outline.set_edgecolor("#333344")

    fig.suptitle("Evolved Blackjack Strategy — Final Population Consensus\n"
                 "Red = Hit  |  Blue = Stand",
                 color="white", fontsize=13, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    population, final_fitnesses, history = run()

    plot_fitness(history, "fitness_over_generations.png")
    plot_strategy_heatmap(population, "strategy_heatmap.png")

    print("\nDone! Output files:")
    print("  fitness_over_generations.png")
    print("  strategy_heatmap.png")
