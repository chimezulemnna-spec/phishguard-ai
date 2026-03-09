# step2_examine_modules.py
import importlib
import inspect
import sys
sys.path.append('src')

modules_to_examine = ['url_features', 'content_features', 'ensemble', 'preprocessing']

for module_name in modules_to_examine:
    try:
        module = importlib.import_module(f'{module_name}')
        
        print(f"\n{'='*60}")
        print(f"Examining {module_name}.py")
        print(f"{'='*60}")
        
        # Get all classes
        classes = inspect.getmembers(module, inspect.isclass)
        if classes:
            print("\n📦 Classes:")
            for name, cls in classes:
                if cls.__module__ == module.__name__:
                    print(f"  {name}")
                    # Get methods
                    methods = inspect.getmembers(cls, inspect.isfunction)
                    for method_name, method in methods[:5]:
                        if not method_name.startswith('_'):
                            sig = str(inspect.signature(method))[:50]
                            print(f"    └─ {method_name}{sig}...")
        
        # Get all functions
        functions = inspect.getmembers(module, inspect.isfunction)
        if functions:
            print("\n🔧 Functions:")
            for name, func in functions[:10]:
                if func.__module__ == module.__name__ and not name.startswith('_'):
                    sig = str(inspect.signature(func))[:50]
                    print(f"  {name}{sig}...")
                    
    except ImportError as e:
        print(f"❌ Could not import {module_name}: {e}")
    except Exception as e:
        print(f"❌ Error examining {module_name}: {e}")