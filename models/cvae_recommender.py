# cvae_recommender.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =======================
# 1. Load & preprocess dataset
# =======================
df = pd.read_csv("clustered_dishes_new.csv")
required_cols = {"DishName", "Category", "Calories", "Price", "Ingredients"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Dataset must have columns: {required_cols}")

feature_cols = ["Calories", "Price"]
scaler = MinMaxScaler()
features = scaler.fit_transform(df[feature_cols])

# =======================
# 2. CVAE model
# =======================
class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, cond_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, 64)
        self.fc2_mu = nn.Linear(64, latent_dim)
        self.fc2_logvar = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim + cond_dim, 64)
        self.fc4 = nn.Linear(64, input_dim)

    def encode(self, x, c):
        h = torch.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = torch.relu(self.fc3(torch.cat([z, c], dim=1)))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def cvae_loss(recon_x, x, mu, logvar):
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

# =======================
# 3. Menu generation logic
# =======================
categories = ["main", "side", "soup", "dessert", "drinks"]
category_dict = {cat: df[df["Category"] == cat].to_dict("records") for cat in categories}

def pick_dish(cat, recent_list):
    available = [d for d in category_dict.get(cat, []) if d["DishName"] not in recent_list]
    if not available:
        # If all dishes recently used or category empty, reset
        available = category_dict.get(cat, [])
        if not available:
            # No dishes at all: return placeholder
            return {"DishName": f"No {cat} available", "Ingredients": "", "Calories": 0, "Price": 0}
    dish = np.random.choice(available)
    recent_list.append(dish["DishName"])
    if len(recent_list) > 5:
        recent_list.pop(0)
    return dish

def generate_day_menu(recent_dict):
    meals = ["breakfast", "lunch", "dinner"]
    day_plan = {}
    for meal in meals:
        for cat in categories:
            dish = pick_dish(cat, recent_dict[meal][cat])
            key_name = f"{meal}_{cat.capitalize()}" if cat != "drinks" else f"{meal}_Drink"
            key_ing = f"{key_name}_Ingredients"
            key_cal = f"{meal}_{cat.capitalize()}_Calories" if cat != "drinks" else f"{meal}_Drink_Calories"
            key_price = f"{meal}_{cat.capitalize()}_Price" if cat != "drinks" else f"{meal}_Drink_Price"

            day_plan[key_name] = dish["DishName"]
            day_plan[key_ing] = dish["Ingredients"]
            day_plan[key_cal] = dish["Calories"]
            day_plan[key_price] = dish["Price"]

    return day_plan

def generate_month_menu_with_meals(days=14):
    # Track recently used dishes per meal & category
    recent_dict = {meal: {cat: [] for cat in categories} for meal in ["breakfast", "lunch", "dinner"]}
    all_days = []

    for day in range(1, days + 1):
        day_plan = generate_day_menu(recent_dict)
        day_plan["Day"] = day
        all_days.append(day_plan)

    return pd.DataFrame(all_days)

# Optional example run
if __name__ == "__main__":
    df_menus = generate_month_menu_with_meals(days=14)
    print(df_menus.head())
