"""
Evolving Blackjack Card Counting Strategy with Genetic Algorithm
================================================================
Extends the basic strategy GA to evolve:
  - Hit/stand play strategy (260 bits)
  - Card count values per rank (22 bits)
  - Bet multipliers per true count range (12 bits)
Total chromosome: 294 bits
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

HARD_TOTALS  = list(range(4, 21))    # 17 values
SOFT_TOTALS  = list(range(12, 21))   # 9 values (soft 12–20)
DEALER_CARDS = list(range(1, 11))    # Ace=1, 2–10

HARD_BITS = len(HARD_TOTALS) * len(DEALER_CARDS)   # 170
SOFT_BITS = len(SOFT_TOTALS) * len(DEALER_CARDS)   # 90
PLAY_BITS = HARD_BITS + SOFT_BITS                  # 260

# Component 2: count values — 11 card ranks × 2 bits = 22 bits
# Ranks: Ace(1), 2, 3, 4, 5, 6, 7, 8, 9, 10 (covers J/Q/K too)
COUNT_RANKS = list(range(1, 11))   # 10 ranks (1=Ace, 10=ten-valued)
COUNT_BITS  = len(COUNT_RANKS) * 2  # 20 bits
# Note: spec says 11 card values but lists 10 (Ace + 2-9 + 10); using 10.

# Component 3: bet multipliers — 4 ranges × 3 bits = 12 bits
BET_RANGES = ["<= -2", "-1 to +1", "+2 to +4", ">= +5"]
BET_BITS   = 4 * 3   # 12 bits

CHROM_LEN = PLAY_BITS + COUNT_BITS + BET_BITS   # 292 bits

# Offsets into chromosome
COUNT_OFFSET = PLAY_BITS
BET_OFFSET   = PLAY_BITS + COUNT_BITS

# GA parameters
POP_SIZE      = 100
N_GENERATIONS = 100
N_HANDS       = 1000
MUTATION_RATE = 0.01
ELITES        = 2

# Game parameters
N_DECKS      = 6
SHOE_SIZE    = N_DECKS * 52          # 312
PENETRATION  = int(SHOE_SIZE * 0.75) # 234 cards before reshuffle
START_BANK   = 1000
MIN_BET      = 1
MAX_BET      = 8

# Hi-Lo reference for comparison
HILO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}

# ─────────────────────────────────────────────
# CHROMOSOME ENCODING / DECODING
# ─────────────────────────────────────────────

def random_chromosome():
    return np.random.randint(0, 2, CHROM_LEN, dtype=np.int8)

# --- Play strategy ---
def get_decision(chrom, player_total, is_soft, dealer_upcard):
    d_idx = DEALER_CARDS.index(dealer_upcard)
    if is_soft and player_total in SOFT_TOTALS:
        p_idx = SOFT_TOTALS.index(player_total)
        bit   = HARD_BITS + p_idx * len(DEALER_CARDS) + d_idx
    else:
        clamped = max(4, min(20, player_total))
        p_idx   = HARD_TOTALS.index(clamped)
        bit     = p_idx * len(DEALER_CARDS) + d_idx
    return int(chrom[bit])

# --- Count values ---
# 2-bit encoding: 00=-1, 01=0, 10=+1, 11=0 (unused → treat as 0)
_COUNT_MAP = {(0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0}

def get_count_values(chrom):
    """Returns dict: card_rank -> count_value (-1/0/+1)"""
    values = {}
    for i, rank in enumerate(COUNT_RANKS):
        b0 = int(chrom[COUNT_OFFSET + i*2])
        b1 = int(chrom[COUNT_OFFSET + i*2 + 1])
        values[rank] = _COUNT_MAP[(b0, b1)]
    return values

# --- Bet multipliers ---
def get_bet_multipliers(chrom):
    """Returns list of 4 multipliers (1–8), one per true count range."""
    mults = []
    for i in range(4):
        bits = chrom[BET_OFFSET + i*3 : BET_OFFSET + i*3 + 3]
        val  = int(bits[0])*4 + int(bits[1])*2 + int(bits[2])  # 0–7
        mults.append(val + 1)  # map to 1–8
    return mults

def true_count_range_idx(tc):
    """Map true count to index 0–3."""
    if tc <= -2:   return 0
    if tc <= 1:    return 1
    if tc <= 4:    return 2
    return 3

# ─────────────────────────────────────────────
# BLACKJACK SIMULATION
# ─────────────────────────────────────────────

def card_value(card):
    return min(card, 10)

def build_shoe():
    """6-deck shoe. Cards 1–13 (1=Ace, 11=J, 12=Q, 13=K)."""
    shoe = list(range(1, 14)) * 4 * N_DECKS
    random.shuffle(shoe)
    return shoe

def hand_value(cards):
    total  = sum(card_value(c) for c in cards)
    n_aces = sum(1 for c in cards if c == 1)
    is_soft = False
    if n_aces > 0 and total + 10 <= 21:
        total  += 10
        is_soft = True
    return total, is_soft

def count_card(card, count_values):
    """Add card to running count using evolved count values."""
    rank = min(card_value(card), 10)   # 1=Ace, 10=all ten-valued
    return count_values.get(rank, 0)

def evaluate_fitness(chrom, track_bankroll=False):
    """
    Simulate N_HANDS from a 6-deck shoe with card counting and bet sizing.
    Returns final bankroll (fitness). If track_bankroll=True, also returns history.
    """
    count_values = get_count_values(chrom)
    bet_mults    = get_bet_multipliers(chrom)

    shoe         = build_shoe()
    pos          = 0
    running_count = 0
    bankroll     = START_BANK
    hands_played = 0
    bankroll_history = [bankroll] if track_bankroll else None

    while hands_played < N_HANDS and bankroll > 0:
        # Reshuffle at penetration
        if pos >= PENETRATION or pos + 10 > len(shoe):
            shoe = build_shoe()
            pos  = 0
            running_count = 0

        # Calculate true count and bet
        remaining_cards = len(shoe) - pos
        decks_remaining = max(remaining_cards / 52, 0.5)
        tc = round(running_count / decks_remaining)
        mult = bet_mults[true_count_range_idx(tc)]
        bet  = min(mult * MIN_BET, bankroll, MAX_BET)
        bet  = max(bet, min(MIN_BET, bankroll))

        # Deal initial cards
        if pos + 4 > len(shoe):
            break

        p_cards = [shoe[pos], shoe[pos+2]]
        d_cards = [shoe[pos+1], shoe[pos+3]]
        pos += 4

        # Update count for all visible cards (player cards + dealer upcard)
        running_count += count_card(p_cards[0], count_values)
        running_count += count_card(p_cards[1], count_values)
        running_count += count_card(d_cards[0], count_values)  # upcard only (hole card hidden)

        # Check naturals
        p_val, _ = hand_value(p_cards)
        d_val, _ = hand_value(d_cards)
        p_bj = (p_val == 21)
        d_bj = (d_val == 21)

        if p_bj or d_bj:
            # Reveal dealer hole card
            running_count += count_card(d_cards[1], count_values)
            if p_bj and d_bj:
                pass  # push
            elif p_bj:
                bankroll += int(bet * 1.5)   # 3:2 payout
            else:
                bankroll -= bet
            hands_played += 1
            if track_bankroll:
                bankroll_history.append(bankroll)
            continue

        # Player's turn
        upcard = min(card_value(d_cards[0]), 10)
        while True:
            p_val, p_soft = hand_value(p_cards)
            if p_val >= 21:
                break
            decision = get_decision(chrom, p_val, p_soft, upcard)
            if decision == 0:
                break
            if pos >= len(shoe):
                break
            new_card = shoe[pos]; pos += 1
            p_cards.append(new_card)
            running_count += count_card(new_card, count_values)

        p_val, _ = hand_value(p_cards)

        # Dealer's turn — reveal hole card first
        running_count += count_card(d_cards[1], count_values)

        if p_val <= 21:
            while True:
                d_val, d_soft = hand_value(d_cards)
                if d_val > 17:
                    break
                if d_val == 17 and not d_soft:
                    break
                if pos >= len(shoe):
                    break
                new_card = shoe[pos]; pos += 1
                d_cards.append(new_card)
                running_count += count_card(new_card, count_values)

        d_val, _ = hand_value(d_cards)

        # Settle bet
        if p_val > 21:
            bankroll -= bet
        elif d_val > 21 or p_val > d_val:
            bankroll += bet
        elif p_val < d_val:
            bankroll -= bet
        # else push

        hands_played += 1
        if track_bankroll:
            bankroll_history.append(bankroll)

    if track_bankroll:
        return max(bankroll, 0), bankroll_history
    return max(bankroll, 0)

# ─────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────

def roulette_select(population, fitnesses):
    """Fitness-proportional selection. Shift so min=0 to handle low fitnesses."""
    f = np.array(fitnesses, dtype=float)
    f = f - f.min() + 1e-6   # shift so all positive
    total = f.sum()
    pick  = random.uniform(0, total)
    cumul = 0.0
    for chrom, fit in zip(population, f):
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
    order = np.argsort(fitnesses)[::-1]
    fitnesses_sorted = [fitnesses[i] for i in order]
    pop_sorted       = [population[i].copy() for i in order]

    next_gen = []
    for i in range(ELITES):
        next_gen.append(pop_sorted[i].copy())

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
    history    = {"min": [], "max": [], "mean": [], "median": []}

    for gen in range(N_GENERATIONS):
        fitnesses = [evaluate_fitness(c) for c in population]

        history["min"].append(np.min(fitnesses))
        history["max"].append(np.max(fitnesses))
        history["mean"].append(np.mean(fitnesses))
        history["median"].append(np.median(fitnesses))

        print(f"Gen {gen+1:3d}/{N_GENERATIONS}  "
              f"max={max(fitnesses):.1f}  mean={np.mean(fitnesses):.1f}  "
              f"min={min(fitnesses):.1f}  (bankroll $)")

        population = evolve(population, fitnesses)

    final_fitnesses = [evaluate_fitness(c) for c in population]
    best_idx = int(np.argmax(final_fitnesses))
    print(f"\nFinal best bankroll: ${max(final_fitnesses):.2f}")
    return population, final_fitnesses, history, best_idx

# ─────────────────────────────────────────────
# FIGURE 1: Fitness (bankroll) over generations
# ─────────────────────────────────────────────

def plot_fitness(history, save_path="cc_fitness_over_generations.png"):
    gens = range(1, len(history["max"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0f0f14")
    fig.patch.set_facecolor("#0f0f14")

    colors = {"max": "#f7a26a", "mean": "#7c6af7", "median": "#4ade80", "min": "#f87171"}
    labels = {"max": "Max", "mean": "Mean", "median": "Median", "min": "Min"}

    for key in ["max", "mean", "median", "min"]:
        ax.plot(gens, history[key], color=colors[key], linewidth=2,
                label=labels[key], alpha=0.9)

    ax.axhline(START_BANK, color="white", linewidth=1, linestyle="--", alpha=0.4,
               label=f"Starting bankroll (${START_BANK})")

    ax.set_xlabel("Generation", color="white", fontsize=12)
    ax.set_ylabel("Final Bankroll ($)", color="white", fontsize=12)
    ax.set_title("Card Counting GA — Fitness (Final Bankroll) Over Generations",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(facecolor="#1a1a24", edgecolor="#333344", labelcolor="white", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:.0f}"))
    ax.grid(color="#222230", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# FIGURE 2: Strategy heatmap
# ─────────────────────────────────────────────

def plot_strategy_heatmap(population, save_path="cc_strategy_heatmap.png"):
    n = len(population)
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

    cmap = mcolors.LinearSegmentedColormap.from_list("bwr", ["#3b82f6", "#e5e7eb", "#ef4444"])
    dealer_labels = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    hard_labels   = [str(t) for t in HARD_TOTALS]
    soft_labels   = [f"A+{t-11}" for t in SOFT_TOTALS]

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
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val  = matrix[i, j]
                text = "H" if val >= 0.5 else "S"
                tc   = "white" if abs(val - 0.5) > 0.2 else "black"
                ax.text(j, i, f"{text}\n{val:.0%}", ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")
        return im

    im1 = draw_heatmap(axes[0], hard_matrix, hard_labels, "Hard Hands")
    draw_heatmap(axes[1], soft_matrix, soft_labels, "Soft Hands")

    cbar = fig.colorbar(im1, ax=axes, orientation="horizontal", pad=0.05,
                        fraction=0.03, aspect=40)
    cbar.set_label("% of population recommending HIT", color="white", fontsize=10)
    cbar.ax.xaxis.set_tick_params(color="white")
    tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(tick_vals)
    cbar.ax.set_xticklabels([f"{int(v*100)}%" for v in tick_vals], color="white")
    cbar.outline.set_edgecolor("#333344")

    fig.suptitle("Evolved Card Counting Strategy — Final Population Consensus\n"
                 "Red = Hit  |  Blue = Stand",
                 color="white", fontsize=13, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# FIGURE 3: Count values vs Hi-Lo
# ─────────────────────────────────────────────

def plot_count_values(population, save_path="cc_count_values.png"):
    """Bar chart comparing evolved count values to Hi-Lo for each rank."""
    n = len(population)
    rank_names  = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    # Average evolved count value per rank across population
    avg_counts = np.zeros(len(COUNT_RANKS))
    for chrom in population:
        cv = get_count_values(chrom)
        for i, rank in enumerate(COUNT_RANKS):
            avg_counts[i] += cv[rank]
    avg_counts /= n

    hilo_vals = [HILO[r] for r in COUNT_RANKS]

    x = np.arange(len(rank_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#0f0f14")
    fig.patch.set_facecolor("#0f0f14")

    bars1 = ax.bar(x - width/2, avg_counts, width, label="Evolved (population avg)",
                   color="#7c6af7", alpha=0.9, zorder=3)
    bars2 = ax.bar(x + width/2, hilo_vals, width, label="Hi-Lo reference",
                   color="#f7a26a", alpha=0.9, zorder=3)

    ax.axhline(0, color="#555566", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(rank_names, color="white", fontsize=11)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1 (unfavorable)", "0 (neutral)", "+1 (favorable)"],
                       color="white", fontsize=10)
    ax.set_xlabel("Card Rank", color="white", fontsize=12)
    ax.set_ylabel("Count Value", color="white", fontsize=12)
    ax.set_title("Evolved Count Values vs. Hi-Lo System",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a24", edgecolor="#333344", labelcolor="white", fontsize=10)
    ax.grid(axis="y", color="#222230", linewidth=0.5, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02 if h >= 0 else h - 0.08,
                f"{h:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# FIGURE 4: Bet multipliers table
# ─────────────────────────────────────────────

def plot_bet_multipliers(population, save_path="cc_bet_multipliers.png"):
    """Table showing evolved bet multipliers per true count range."""
    n = len(population)
    avg_mults = np.zeros(4)
    for chrom in population:
        mults = get_bet_multipliers(chrom)
        for i, m in enumerate(mults):
            avg_mults[i] += m
    avg_mults /= n

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_facecolor("#0f0f14")
    fig.patch.set_facecolor("#0f0f14")
    ax.axis("off")

    col_labels = ["True Count Range", "Avg Multiplier", "Avg Bet ($)", "Interpretation"]
    interp = ["Deck unfavorable — bet minimum",
              "Neutral — base bet",
              "Deck favorable — increase bet",
              "Strongly favorable — maximum bet"]

    table_data = []
    for i, (rng, interp_str) in enumerate(zip(BET_RANGES, interp)):
        mult = avg_mults[i]
        table_data.append([rng, f"{mult:.2f}x", f"${mult:.2f}", interp_str])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2a2a40")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style rows with gradient based on multiplier
    row_colors = ["#1a1a30", "#1a2a1a", "#2a2a1a", "#2a1a1a"]
    for i in range(1, 5):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(row_colors[i-1])
            table[i, j].set_text_props(color="white")

    ax.set_title("Evolved Bet Multipliers by True Count Range",
                 color="white", fontsize=13, fontweight="bold", pad=20)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# FIGURE 5: Best individual bankroll over 1000 hands
# ─────────────────────────────────────────────

def plot_bankroll(best_chrom, save_path="cc_bankroll_trace.png"):
    """Run a single 1000-hand session for the best individual and plot bankroll."""
    _, bankroll_history = evaluate_fitness(best_chrom, track_bankroll=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#0f0f14")
    fig.patch.set_facecolor("#0f0f14")

    hands = range(len(bankroll_history))
    ax.plot(hands, bankroll_history, color="#7c6af7", linewidth=1.2, alpha=0.9)
    ax.fill_between(hands, START_BANK, bankroll_history,
                    where=[b >= START_BANK for b in bankroll_history],
                    alpha=0.15, color="#4ade80", label="Above start")
    ax.fill_between(hands, START_BANK, bankroll_history,
                    where=[b < START_BANK for b in bankroll_history],
                    alpha=0.15, color="#f87171", label="Below start")

    ax.axhline(START_BANK, color="white", linewidth=1, linestyle="--",
               alpha=0.5, label=f"Starting bankroll (${START_BANK})")

    final = bankroll_history[-1]
    ax.annotate(f"Final: ${final}",
                xy=(len(bankroll_history)-1, final),
                xytext=(-80, 20), textcoords="offset points",
                color="white", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#f7a26a"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a24", edgecolor="#7c6af7"))

    ax.set_xlabel("Hand Number", color="white", fontsize=12)
    ax.set_ylabel("Bankroll ($)", color="white", fontsize=12)
    ax.set_title("Best Individual — Bankroll Over 1,000 Hands",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:.0f}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.legend(facecolor="#1a1a24", edgecolor="#333344", labelcolor="white", fontsize=10)
    ax.grid(color="#222230", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────

def print_summary(population, final_fitnesses, best_idx):
    best = population[best_idx]
    print("\n" + "="*55)
    print("  BEST INDIVIDUAL SUMMARY")
    print("="*55)

    cv = get_count_values(best)
    print("\n  Evolved Count Values vs Hi-Lo:")
    print(f"  {'Rank':<6} {'Evolved':>8} {'Hi-Lo':>8}")
    print(f"  {'-'*24}")
    rank_names = {1:"Ace",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"10+"}
    for rank in COUNT_RANKS:
        match = "✓" if cv[rank] == HILO[rank] else "✗"
        print(f"  {rank_names[rank]:<6} {cv[rank]:>+8}  {HILO[rank]:>+7}  {match}")

    mults = get_bet_multipliers(best)
    print("\n  Evolved Bet Multipliers:")
    for rng, m in zip(BET_RANGES, mults):
        print(f"  {rng:<12}  → ${m} bet")

    print(f"\n  Final bankroll: ${max(final_fitnesses):.2f}")
    print("="*55)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    population, final_fitnesses, history, best_idx = run()

    print_summary(population, final_fitnesses, best_idx)

    plot_fitness(history,                       "cc_fitness_over_generations.png")
    plot_strategy_heatmap(population,           "cc_strategy_heatmap.png")
    plot_count_values(population,               "cc_count_values.png")
    plot_bet_multipliers(population,            "cc_bet_multipliers.png")
    plot_bankroll(population[best_idx],         "cc_bankroll_trace.png")

    print("\nDone! Output files:")
    for f in ["cc_fitness_over_generations.png", "cc_strategy_heatmap.png",
              "cc_count_values.png", "cc_bet_multipliers.png", "cc_bankroll_trace.png"]:
        print(f"  {f}")
