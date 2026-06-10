import sys

sys.path.insert(0, "src")

from app import app


if __name__ == "__main__":
    app.run(debug=False)
