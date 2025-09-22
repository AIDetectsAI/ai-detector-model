import importlib.util
import inspect
import sys
from pathlib import Path
from tqdm import tqdm
import typer
from config import MODELS_DIR
from loguru import logger
app = typer.Typer()

@app.command()
def load_class_instance(model_name: str):
    file_path = MODELS_DIR / 'pytorch' / f"{model_name}.py"
    logger.info(f'Loading model: {file_path}') 
    if not file_path.exists() or file_path.suffix != ".py":
        raise ValueError("Podany plik nie istnieje lub nie jest plikiem .py")

    module_name = file_path.stem

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    classes = [obj for name, obj in inspect.getmembers(module, inspect.isclass)
               if obj.__module__ == module_name]

    if not classes:
        raise ValueError("Nie znaleziono żadnej klasy w pliku.")
    if len(classes) > 1:
        print("Uwaga: znaleziono więcej niż jedną klasę, użyję pierwszej.")
    
    cls = classes[0]
    instance = cls()
    print(instance)

if __name__ == "__main__":
    app()