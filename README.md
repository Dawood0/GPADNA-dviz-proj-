# Student Habits Dashboard

This Dash app compares student habits across high, average, and low grade
categories. The project separates data preparation, visualizations, and the app
shell so additional teammate visualizations can be added without changing the
existing radar visual.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 src/app.py
```

Open the local URL printed by Dash, usually `http://127.0.0.1:8050/`.

## Structure

```text
data/
  student_habits_performance.csv
src/
  app.py
  preprocessing.py
  all_in_one.py
  vizs_src/
    __init__.py
    radar.py
requirements.txt
README.md
HOW_TO_RUN.txt
```

- `src/preprocessing.py` loads, cleans, detects, and prepares dataset fields.
- `src/vizs_src/radar.py` contains the current radar visualization.
- `src/all_in_one.py` is the registry for current and future visualizations.
- `src/app.py` contains the Dash interface and callbacks.

## Add A Visualization

Create a module under `src/vizs_src/` that exposes this interface:

```python
def create_visual(df, **kwargs):
    return fig, title, explanation
```

Import and register that function in `src/all_in_one.py`. The `visual_name`
argument can then select it through the shared registry.
