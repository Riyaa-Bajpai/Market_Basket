# %%
import numpy as np
import pandas as pd

# %% [markdown]
# DataSet Loading And Inspection

# %%
df=pd.read_csv('Groceries_dataset.csv')

# %%
print(df.shape)
print(df.info())

# %% [markdown]
# Normalise item names

# %%
import re

# %%
def clean_item(item):
    if pd.isna(item):
        return ""
    item = str(item).lower()                 # lowercase
    item = re.sub(r'[^a-z\s]', '', item)     # keep only letters & spaces
    item = re.sub(r'\s+', ' ', item).strip() # collapse multiple spaces
    return item


# %%
df['cleaned'] = df['itemDescription'].apply(clean_item)
print(df[['itemDescription', 'cleaned']].head(2))

# %% [markdown]
# Manual mapping

# %%
mapping = {
    # Dairy
    "milk": "dairy",
    "whole milk": "dairy",
    "butter milk": "dairy",
    "cream": "dairy",
    "cream cheese": "dairy",
    "curd": "dairy",
    "curd cheese": "dairy",
    "processed cheese": "dairy",
    "soft cheese": "dairy",
    "spread cheese": "dairy",
    "whipped sour cream": "dairy",
    "yogurt": "dairy",
    "butter": "dairy",
    "hard cheese": "dairy",
    "domestic eggs": "dairy",
    "frozen dessert": "dairy",
    
    # Bakery
    "brown bread": "bakery",
    "white bread": "bakery",
    "pastry": "bakery",
    "cake bar": "bakery",
    "rolls/buns": "bakery",
    "zwieback": "bakery",
    "baking powder": "bakery",
    "flour": "bakery",
    
    # Beverages
    "bottled water": "beverages",
    "soda": "beverages",
    "soft drinks": "beverages",
    "fruit/vegetable juice": "beverages",
    "coffee": "beverages",
    "tea": "beverages",
    "instant coffee": "beverages",
    "liqueur": "alcohol",
    "rum": "alcohol",
    "brandy": "alcohol",
    "red/blush wine": "alcohol",
    "white wine": "alcohol",
    "sparkling wine": "alcohol",
    "beer": "alcohol",
    
    # Meat & Seafood
    "beef": "meat",
    "chicken": "meat",
    "pork": "meat",
    "fish": "seafood",
    "canned fish": "seafood",
    "ham": "meat",
    "sausage": "meat",
    "meat": "meat",
    "frozen fish": "seafood",
    "frozen chicken": "meat",
    
    # Produce
    "berries": "produce",
    "citrus fruit": "produce",
    "other vegetables": "produce",
    "onions": "produce",
    "root vegetables": "produce",
    "tropical fruit": "produce",
    "pip fruit": "produce",
    "herbs": "produce",
    "salad": "produce",
    "grapes": "produce",
    "cabbage": "produce",
    "mushrooms": "produce",
    "tomatoes": "produce",
    "pot plants": "produce",
    "flower (seeds/plants)": "produce",
    
    # Household / Cleaning
    "abrasive cleaner": "household",
    "bathroom cleaner": "household",
    "detergent": "household",
    "cleaner": "household",
    "dish cleaner": "household",
    "candles": "household",
    "kitchen towels": "household",
    "napkins": "household",
    "toilet cleaner": "household",
    "sponges": "household",
    "bags": "household",
    "cling film/bags": "household",
    "aluminum foil": "household",
    "light bulbs": "household",
    "matches": "household",
    "cat food": "household",
    "dog food": "household",
    "decalcifier": "household",
    
    # Personal Care
    "baby cosmetics": "personal care",
    "soap": "personal care",
    "cosmetics": "personal care",
    "shampoo": "personal care",
    "hygiene articles": "personal care",
    "oral hygiene": "personal care",
    "dental care": "personal care",
    "razor blades": "personal care",
    "skin care": "personal care",
    "hair spray": "personal care",
    "male cosmetics": "personal care",
    "perfume": "personal care",
    
    # Snacks & Confectionery
    "candy": "snacks",
    "chocolate": "snacks",
    "popcorn": "snacks",
    "waffles": "snacks",
    "salty snack": "snacks",
    "chips": "snacks",
    "biscuits": "snacks",
    "ice cream": "snacks",
    "chewing gum": "snacks",
    "nuts/prunes": "snacks",
    "dessert": "snacks",
    "cake": "snacks",
    
    # Canned & Packaged Foods
    "canned fruit": "canned/packaged",
    "canned vegetables": "canned/packaged",
    "canned beer": "alcohol",
    "soups": "canned/packaged",
    "cereals": "canned/packaged",
    "sugar": "canned/packaged",
    "salt": "canned/packaged",
    "jam": "canned/packaged",
    "honey": "canned/packaged",
    "cooking chocolate": "canned/packaged",
    "pudding powder": "canned/packaged",
    "spices": "canned/packaged",
    "mustard": "canned/packaged",
    "ketchup": "canned/packaged",
    "mayonnaise": "canned/packaged",
    "oil": "canned/packaged",
    "vinegar": "canned/packaged",
    "pickles": "canned/packaged",
    "sauces": "canned/packaged",
    "frozen vegetables": "frozen",
    "frozen potato products": "frozen",
    "frozen meals": "frozen",
    "condensed milk": "canned/packaged",
    
    # Frozen & Ready Meals
    "frozen dessert": "frozen",
    "frozen fish": "frozen",
    "frozen chicken": "frozen",
    "frozen meals": "frozen",
    "frozen vegetables": "frozen",
    "frozen potato products": "frozen",
    
}


# %%
def map_item(item):
    if item in mapping:
        return mapping[item]
    else:
        return "other"

df['mapped'] = df['cleaned'].apply(map_item)

# %% [markdown]
# Group items by Member

# %%
baskets=df.groupby('Member_number')['mapped'].apply(lambda x:list(set(x)))
baskets.head(2)

# %%
all_items = [item for sublist in baskets for item in sublist]
unique_items = sorted(set(all_items))
print(len(unique_items))

# %%
value_counts = df['mapped'].value_counts()
common_items = value_counts[value_counts >= 10].index 
filtered_baskets = baskets.apply(lambda items: [x for x in items if x in common_items])

# %%
def get_all_items():
    return sorted(df['itemDescription'].str.strip().str.lower().unique())

# %% [markdown]
# Transforming

# %%
from mlxtend.preprocessing import TransactionEncoder

# %%
transaction = filtered_baskets.tolist()

# %%
t = TransactionEncoder()

# %%
to_array = t.fit(transaction).transform(transaction)
encoded = pd.DataFrame(to_array, columns = t.columns_)

# %%
print(encoded.head(2))

# %%
df1 = pd.DataFrame(to_array.astype(int),columns = t.columns_)

# %%
print(df1.head(2))

# %% [markdown]
# Fpgrowth

# %%
from mlxtend.frequent_patterns import fpgrowth

# %%
print(encoded.columns.tolist())

# %%
print(encoded.shape)

# %%
freq = fpgrowth(encoded, min_support=0.01, use_colnames=True)

# %%
print(encoded.columns.tolist())

# %% [markdown]
# Association rules

# %%
from mlxtend.frequent_patterns import association_rules

# %%
rules = association_rules(freq,metric='lift',min_threshold=1)
rules = rules[['antecedents','consequents','support','confidence','lift']]

# %%
filtering = rules[(rules['confidence']>0.4) & (rules['lift']>0.8)]

# %%
filtering.sort_values(by='lift', ascending=False).head(2)

# %% [markdown]
# Recommendation

# %%
from collections import Counter

def recommend_for_basket(user_basket, rules, top_n=5):
    user_basket = set(user_basket)
    scores = Counter()
    
    for _, row in rules.iterrows():
        antecedent = set(row['antecedents'])
        if antecedent.issubset(user_basket):
            for rec in row['consequents']:
                if rec not in user_basket:
                    scores[rec] += row['confidence'] * row['lift']
                    
    return [item for item, _ in scores.most_common(top_n)]

# %%
def recommend_for_basket(user_basket, rules, top_n=5):
    user_basket = set(user_basket)
    scores = Counter()
    
    for _, row in rules.iterrows():
        antecedent = set(row['antecedents'])
        if antecedent.issubset(user_basket):
            consequents = list(row['consequents'])   # convert frozenset to list
            for rec in consequents:
                if rec not in user_basket:
                    scores[rec] += row['confidence'] * row['lift']
                    
    return [item for item, _ in scores.most_common(top_n)]
import random
def get_item_recommendations(user_basket, rules, df, mapping, top_n=5):
    user_basket_mapped = [mapping.get(item.lower(), "other") for item in user_basket]
    recommendations_mapped = recommend_for_basket(user_basket_mapped, rules, top_n=top_n)
    recommendations = []
    for cat in recommendations_mapped:
        items_in_cat = [item for item in df[df['mapped'] == cat]['itemDescription'].unique()
                        if item not in user_basket]
        if items_in_cat:
            random.shuffle(items_in_cat)
            recommendations.extend(items_in_cat[:2]) 
    
    return recommendations[:top_n]

# %% [markdown]
# Sample usage

# %%
user_basket = ['dairy','fruits']
recommendations = recommend_for_basket(user_basket, filtering, top_n=3)

print("Your Basket:")
for item in user_basket:
    print(f"   - {item}")

print("\nRecommended Items:")
for idx, rec in enumerate(recommendations, 1):
    print(f"   {idx}. {rec}")

# %%
df1.to_csv("encoded_baskets.csv", index=False)
df.to_csv("cleaned_groceries.csv", index=False)
filtered_baskets.to_frame(name="basket").to_csv("filtered_baskets.csv")


