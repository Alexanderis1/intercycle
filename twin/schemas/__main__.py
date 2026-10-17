"""`python -m twin.schemas`: regenerate the committed JSON Schema export in twin/schemas/json/."""
from twin.schemas import export_json

if __name__ == "__main__":
    for p in export_json():
        print(p)
