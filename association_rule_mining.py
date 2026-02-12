import pandas as pd
from itertools import combinations

# Function to calculate support

def support(data, itemset):
    return len(data[data.isin(itemset).all(axis=1)]) / len(data)

# Function to calculate confidence

def confidence(data, itemset, antecedent):
    sup_antecedent = support(data, antecedent)
    sup_both = support(data, itemset)
    return sup_both / sup_antecedent if sup_antecedent > 0 else 0

# Function to calculate lift

def lift(data, itemset, antecedent):
    conf = confidence(data, itemset, antecedent)
    sup_consequent = support(data, itemset) / len(data)
    return conf / sup_consequent if sup_consequent > 0 else 0


if __name__ == '__main__':
    # Sample dataset
    sample_data = pd.DataFrame({
        'Transaction': [1, 2, 3, 4, 5],
        'Items': [
            ['Milk', 'Bread'],
            ['Bread', 'Diaper', 'Beer'],
            ['Milk', 'Diaper', 'Bread'],
            ['Bread', 'Beer'],
            ['Milk', 'Diaper']
        ]
    })
    
    # Creating basket
    basket = (sample_data
              .groupby(['Transaction'])['Items']
              .apply(set)
              .reset_index(name='Items'))
    
    # Defining itemset and antecedent
    itemset = ['Milk', 'Bread']
    antecedent = ['Milk']

    # Computing support, confidence and lift
    print('Support:', support(basket['Items'], itemset))
    print('Confidence:', confidence(basket['Items'], itemset, antecedent))
    print('Lift:', lift(basket['Items'], itemset, antecedent))
