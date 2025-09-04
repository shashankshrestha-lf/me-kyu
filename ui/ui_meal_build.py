# ui_menu_planner_meal_builder.py
import gradio as gr
import pandas as pd
import re
from cvae_recommender import generate_month_menu_with_meals

# === Load inventory ===
inventory_df = pd.read_csv("inventory_list.csv")
inventory_df["Ingredient"] = inventory_df["Ingredient"].str.lower()
inventory_df.set_index("Ingredient", inplace=True)

generated_menus = None

# === Recalculate calories/cost from ingredients ===
def recalc_nutrition(ingredient_list):
    total_cal, total_cost = 0, 0
    for ing in ingredient_list:
        name = ing["ingredient"].strip().lower()
        grams = ing["grams"]
        if name in inventory_df.index:
            cal_per_g = inventory_df.loc[name, "Calories_per_g"]
            cost_per_g = inventory_df.loc[name, "Cost_per_g"]
            total_cal += cal_per_g * grams
            total_cost += cost_per_g * grams
    return total_cal, total_cost

# === Generate Menus ===
def generate_menus_ui(days, b_cal, b_cost, l_cal, l_cost, d_cal, d_cost):
    global generated_menus
    generated_menus = generate_month_menu_with_meals(days=days)
    return render_calendar(generated_menus)

# === Render Menu Calendar ===
def render_calendar(menus):
    html = "<div style='display:grid; grid-template-columns: repeat(7, 1fr); gap:10px;'>"
    for _, row in menus.iterrows():
        html += f"<div style='border:1px solid #ccc; padding:10px; border-radius:8px; background:#f9f9f9;'>"
        html += f"<h4>Day {int(row['Day'])}</h4>"
        for meal in ["breakfast", "lunch", "dinner"]:
            components = []
            ingredients_list = []
            cal, price = 0, 0
            for cat in ["Main", "Side", "Soup", "Dessert", "Drink"]:
                dish_name_col = f"{meal}_{cat}"
                ingredients_col = f"{meal}_{cat}_Ingredients"
                calories_col = f"{meal}_{cat}_Calories"
                price_col = f"{meal}_{cat}_Price"

                dish_name = row.get(dish_name_col, "")
                ingredients = row.get(ingredients_col, "")
                calories = row.get(calories_col, 0)
                cost = row.get(price_col, 0)

                if dish_name:
                    components.append(dish_name)
                if ingredients:
                    ingredients_list.append(ingredients)
                cal += calories
                price += cost

            html += f"<b>{meal.capitalize()}:</b> {' + '.join(components)}<br>"
            html += f"<small>Calories: {cal:.0f}, Price: {price:.2f}</small>"
            html += "<details><summary>📝 Ingredients</summary>"
            html += "<br>".join(ingredients_list)
            html += "</details><br>"

        html += "</div>"
    html += "</div>"
    return html

# === Fetch Ingredients for Editing Menus ===
def fetch_ingredients(day, meal):
    global generated_menus
    if generated_menus is None:
        return "⚠️ Please generate menus first."
    row = generated_menus.loc[generated_menus["Day"] == int(day)]
    if row.empty:
        return "⚠️ Day not found."
    dish_name = row.iloc[0][f"{meal}_Main"]
    ingredients = row.iloc[0].get(f"{meal}_Main_Ingredients", "")
    cal, price = recalc_nutrition([{"ingredient": i, "grams":50} for i in ingredients.split(",") if i.strip()])
    return f"🍽️ Dish: {dish_name}\n\n📝 Ingredients: {ingredients}\n🔥 Calories: {cal:.0f}, 💰 Price: {price:.2f}"

# === Edit Ingredients in Menus ===
def edit_menu(day, meal, ingredient, action):
    global generated_menus
    if generated_menus is None:
        return "⚠️ Please generate menus first.", None

    row_idx = generated_menus.index[generated_menus["Day"] == int(day)][0]
    meal_key = f"{meal}_Main_Ingredients"

    current_ing = generated_menus.loc[row_idx, meal_key]
    ingredients = [ing.strip() for ing in current_ing.split(",") if ing.strip()]

    if action == "add":
        if ingredient not in ingredients:
            ingredients.append(ingredient.strip())
    elif action == "remove":
        ingredients = [ing for ing in ingredients if ing.lower() != ingredient.strip().lower()]

    updated_list = ", ".join(ingredients)
    generated_menus.loc[row_idx, meal_key] = updated_list

    new_cal, new_cost = recalc_nutrition([{"ingredient": i, "grams":50} for i in ingredients])
    generated_menus.loc[row_idx, f"{meal}_Main_Calories"] = new_cal
    generated_menus.loc[row_idx, f"{meal}_Main_Price"] = new_cost

    return (
        f"✅ Updated {meal} on Day {day}\n📝 Ingredients: {updated_list}\n🔥 Calories: {new_cal:.0f}, 💰 Price: {new_cost:.2f}",
        render_calendar(generated_menus)
    )

# === Meal Builder ===
def build_meal(*args):
    max_len = len(args) // 2
    ingredients = args[:max_len]
    grams_list = args[max_len:]

    total_cal, total_cost, total_grams = 0, 0, 0
    used_ingredients = []

    for ing, g in zip(ingredients, grams_list):
        ing_name = str(ing).strip().lower()
        g = float(g) if g else 0
        if ing_name == "" or g <= 0:
            continue

        used_ingredients.append(f"{ing_name} ({g:.0f}g)")
        total_grams += g

        if ing_name in inventory_df.index:
            cal_per_g = float(inventory_df.loc[ing_name, "Calories_per_g"])
            cost_per_g = float(inventory_df.loc[ing_name, "Cost_per_g"])
            total_cal += cal_per_g * g
            total_cost += cost_per_g * g  # yen

    return (
        f"Ingredients: {', '.join(used_ingredients)}\n"
        f"Total grams: {total_grams:.0f}g\n"
        f"Calories: {total_cal:.0f}\n"
        f"Price: ¥{total_cost:.0f}"
    )


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("##  Me-kyu Menu Planner")

    with gr.Tab("📅 Menu Generation"):
        days = gr.Slider(1, 30, value=7, step=1, label="Number of Days")
        b_cal = gr.Slider([300, 500], minimum=200, maximum=800, step=50, label="Breakfast Calories Range")
        b_cost = gr.Slider([0.5, 1.0], minimum=0.3, maximum=2.0, step=0.1, label="Breakfast Cost Range")
        l_cal = gr.Slider([600, 900], minimum=400, maximum=1200, step=50, label="Lunch Calories Range")
        l_cost = gr.Slider([1.0, 1.5], minimum=0.5, maximum=3.0, step=0.1, label="Lunch Cost Range")
        d_cal = gr.Slider([500, 800], minimum=300, maximum=1000, step=50, label="Dinner Calories Range")
        d_cost = gr.Slider([0.8, 1.3], minimum=0.5, maximum=2.5, step=0.1, label="Dinner Cost Range")

        generate_btn = gr.Button("🚀 Generate Menus")
        output_html = gr.HTML()
        generate_btn.click(
            generate_menus_ui,
            inputs=[days, b_cal, b_cost, l_cal, l_cost, d_cal, d_cost],
            outputs=output_html
        )

    with gr.Tab("✏️ Edit Ingredients"):
        day = gr.Number(label="Day")
        meal = gr.Dropdown(["breakfast", "lunch", "dinner"], label="Meal")
        fetch_btn = gr.Button("🔍 Show Current Ingredients")
        current_list = gr.Textbox(label="Current Dish & Ingredients", interactive=False)
        ingredient = gr.Textbox(label="Ingredient Name")
        action = gr.Radio(["add", "remove"], label="Action")
        edit_btn = gr.Button("Update Ingredient")
        edit_output = gr.Textbox(label="Result")
        updated_calendar = gr.HTML()
        fetch_btn.click(fetch_ingredients, inputs=[day, meal], outputs=current_list)
        edit_btn.click(edit_menu, inputs=[day, meal, ingredient, action], outputs=[edit_output, updated_calendar])

    with gr.Tab("🥗 Meal Builder"):
        max_ingredients = 10
        ingredient_inputs = []
        grams_inputs = []
        for i in range(max_ingredients):
            with gr.Row():
                ingredient_inputs.append(
                gr.Dropdown(list(inventory_df.index), label=f"Ingredient {i+1}", value="", allow_custom_value=True)
            )
            grams_inputs.append(gr.Number(label="Grams", value=50, interactive=True))

        build_btn = gr.Button("Build Meal")
        meal_output = gr.Textbox(label="Meal Nutrition & Cost", interactive=False)

        # FIX: unpack the lists with *
        build_btn.click(build_meal, inputs=[*ingredient_inputs, *grams_inputs], outputs=meal_output)


demo.launch()
