# 🍽️ Mekyu Meal Recommender

A smart meal planning system that generates balanced menus with cost and calorie constraints, using:
- **Deterministic optimization (CP-SAT via OR-Tools)**
- Optional **CVAE model** for dish embeddings
- **Inventory-based meal builder**
- **Interactive UI with Gradio**

## Features
- Menu generation for multiple days with nutrition + cost targets
- Ingredient editing and recalculation
- Build custom meals from inventory
- Exportable technical documentation

## Project Structure
- `data/` → CSV datasets
- `models/` → Recommendation algorithms (CVAE + CP-SAT)
- `ui/` → Gradio-based user interfaces
- `docs/` → Technical notes & reports

## Run the UI
```bash
python ui/ui_meal_build.py
