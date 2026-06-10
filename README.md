# Student Habits Dashboard

This Dash app compares student habits across high, average, and low grade
categories. The project separates data preparation, visualizations, and the app
shell so additional teammate visualizations can be added without changing the
existing radar visual.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 server.py
```

Open the local URL printed by Dash, usually `http://127.0.0.1:8050/`.

## Structure

```text
data/
  student_habits_performance.csv
src/
  assets/
    style.css
  app.py
  preprocessing.py
  vizs_src/
    __init__.py
    radar.py
requirements.txt
README.md
HOW_TO_RUN.txt
```

- `src/preprocessing.py` loads, cleans, detects, and prepares dataset fields.
- `src/vizs_src/radar.py` contains the current radar visualization.
- `src/app.py` contains the compact Dash interface and callback.
- `src/assets/style.css` contains the page styling.

## Add A Visualization

Create a module under `src/vizs_src/` that exposes this interface:

```python
def create_visual(df, **kwargs):
    return fig, title, explanation
```

Import the function directly in `src/app.py` with a clear alias, then call it
from the appropriate callback. Comments in `src/app.py` show an example.
