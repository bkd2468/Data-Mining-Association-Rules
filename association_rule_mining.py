"""Association rule mining on a small text dataset.

This script demonstrates:
- creating transactions from text sentences
- calculating support for itemsets
- generating association rules
- calculating confidence and lift for each rule
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def preprocess_texts(texts: Sequence[str]) -> List[Set[str]]:
    """Convert text strings into transactions (sets of lowercase words)."""
    transactions: List[Set[str]] = []
    for text in texts:
        tokens = [token.strip(".,!?;:").lower() for token in text.split() if token.strip(".,!?;:")]
        transactions.append(set(tokens))
    return transactions


def calculate_support(transactions: Sequence[Set[str]], itemset: Set[str]) -> float:
    """Support = count(transactions containing itemset) / total transactions."""
    count = sum(1 for transaction in transactions if itemset.issubset(transaction))
    return count / len(transactions)


def all_itemsets(transactions: Sequence[Set[str]], min_size: int = 1, max_size: int = 3) -> Iterable[Set[str]]:
    """Yield unique candidate itemsets of sizes in [min_size, max_size]."""
    items = sorted(set().union(*transactions))
    for size in range(min_size, max_size + 1):
        for combo in combinations(items, size):
            yield set(combo)


def generate_rules(
    transactions: Sequence[Set[str]],
    min_support: float = 0.2,
    min_confidence: float = 0.5,
) -> List[Tuple[Set[str], Set[str], float, float, float]]:
    """Generate association rules with support, confidence, and lift.

    Returns tuples:
      (antecedent, consequent, support, confidence, lift)
    """
    support_cache: Dict[frozenset[str], float] = {}

    for itemset in all_itemsets(transactions, min_size=1, max_size=3):
        support_cache[frozenset(itemset)] = calculate_support(transactions, itemset)

    rules: List[Tuple[Set[str], Set[str], float, float, float]] = []
    for itemset_key, itemset_support in support_cache.items():
        itemset = set(itemset_key)
        if len(itemset) < 2 or itemset_support < min_support:
            continue

        for antecedent_size in range(1, len(itemset)):
            for antecedent_tuple in combinations(sorted(itemset), antecedent_size):
                antecedent = set(antecedent_tuple)
                consequent = itemset - antecedent

                antecedent_support = support_cache[frozenset(antecedent)]
                consequent_support = support_cache[frozenset(consequent)]

                confidence = itemset_support / antecedent_support
                lift = confidence / consequent_support

                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, itemset_support, confidence, lift))

    rules.sort(key=lambda x: (x[4], x[3], x[2]), reverse=True)
    return rules


def main() -> None:
    # Sample text dataset (each sentence is one transaction)
    texts = [
        "Python data mining finds useful patterns",
        "Python machine learning and data mining",
        "Data mining and association rules in python",
        "Machine learning with python and pandas",
        "Association rules help data mining tasks",
        "Python pandas and data analysis for mining",
        "Learning association rules with data mining",
        "Python data science and machine learning",
    ]

    transactions = preprocess_texts(texts)

    print("Transactions:")
    for i, transaction in enumerate(transactions, start=1):
        print(f"T{i}: {sorted(transaction)}")

    rules = generate_rules(transactions, min_support=0.25, min_confidence=0.6)

    print("\nTop association rules (support >= 0.25, confidence >= 0.60):")
    print("antecedent -> consequent | support | confidence | lift")
    for antecedent, consequent, support, confidence, lift in rules[:20]:
        lhs = ", ".join(sorted(antecedent))
        rhs = ", ".join(sorted(consequent))
        print(f"{{{lhs}}} -> {{{rhs}}} | {support:.2f} | {confidence:.2f} | {lift:.2f}")


if __name__ == "__main__":
    main()
