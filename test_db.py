import sys
import os
sys.path.append(os.getcwd())
from app.core.database import engine, Base
Base.prepare(autoload_with=engine)
for name, cls in Base.classes.items():
    if "file" in name.lower() or "similar" in name.lower():
        print(f"Table: {name}")
        for col in cls.__table__.columns:
            print(f"  - {col.name}")
