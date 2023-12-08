import itertools

class Apriori:
    def __init__(self, min_support=0.2, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = None
        self.itemsets = []

    def fit(self, transactions):
        self.transactions = transactions
        self.itemsets = self.generate_frequent_itemsets()

    def generate_frequent_itemsets(self):
        unique_items = set(item for transaction in self.transactions for item in transaction)
        frequent_itemsets = [{frozenset([item]): 0} for item in unique_items]

        k = 1
        while frequent_itemsets[-1]:
            candidates = self.generate_candidates(frequent_itemsets[-1])
            frequent_itemsets.append(self.filter_candidates(candidates))
            k += 1

        return frequent_itemsets[:-1]

    def filter_candidates(self, candidates):
        frequent_itemsets = {}
        for transaction in self.transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    frequent_itemsets[candidate] = frequent_itemsets.get(candidate, 0) + 1

        total_transactions = len(self.transactions)
        return {itemset: support / total_transactions for itemset, support in frequent_itemsets.items()
                if support / total_transactions >= self.min_support}
    
    def generate_candidates(self, itemsets):
        candidates = set()
        k = len(next(iter(itemsets)))
        for i, itemset1 in enumerate(itemsets):
            for j, itemset2 in enumerate(itemsets):
                if i < j:
                    union = itemset1.union(itemset2)
                    if len(union) == k + 1:
                        candidates.add(union)
        return candidates


    def generate_association_rules(self):
        association_rules = []
        for i in range(1, len(self.itemsets)):
            for itemset in self.itemsets[i]:
                self.generate_rules_for_itemset(itemset, association_rules)
        return association_rules

    def generate_rules_for_itemset(self, itemset, association_rules):
        for i in range(1, len(itemset)):
            antecedent_combinations = itertools.combinations(itemset, i)
            for antecedent in antecedent_combinations:
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                support_itemset = self.itemsets[len(itemset) - 1][itemset]
                support_antecedent = self.itemsets[len(antecedent) - 1][antecedent]
                confidence = support_itemset / support_antecedent

                if confidence >= self.min_confidence:
                    association_rules.append((antecedent, consequent, confidence))


# Example usage with a different sample dataset
transactions = [
    {'bread', 'milk', 'eggs'},
    {'bread', 'butter', 'jam'},
    {'milk', 'butter'},
    {'bread', 'milk', 'butter', 'jam'},
    {'bread', 'eggs','eggs'}
]

# Instantiate and fit the Apriori model with lower min_support
apriori = Apriori(min_support=0.2, min_confidence=0.5)
apriori.fit(transactions)

# Print frequent itemsets
print("Frequent Itemsets:")
for i, itemsets in enumerate(apriori.itemsets):
    print(f"Itemsets of length {i + 1}:")
    for itemset, support in itemsets.items():
        print(f"{set(itemset)}: Support = {support:.2%}")

# Generate and print association rules
association_rules = apriori.generate_association_rules()
print("\nAssociation Rules:")
for antecedent, consequent, confidence in association_rules:
    print(f"{set(antecedent)} => {set(consequent)}: Confidence = {confidence:.2%}")

