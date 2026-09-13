import pandas as pd, os
from pathlib import Path

FIBONACCI = {1, 2, 3, 5, 8, 13, 21, 34, 55}
raw_dir = Path("data/01_raw/neodataset/csv")

frames = []
for csv in raw_dir.glob("*.csv"):
    df = pd.read_csv(csv)
    df["project_id"] = csv.stem
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"Total crudo: {len(raw)}")

# Criterio 1: body no vacío
c1 = raw["description"].notna() & (raw["description"].str.strip() != "")
# Criterio 2: weight no nulo ni cero
c2 = raw["storypoints"].notna() & (raw["storypoints"] != 0)
# Criterio 3: weight mapeable a Fibonacci
c3 = raw["storypoints"].isin(FIBONACCI)

screened = raw[c1 & c2 & c3]
print(f"[N] tras quality screening: {len(screened)}")

# Aplicar lógica de smart_sampler para obtener N post-cap
screened['word_count'] = screened['user_story'].apply(lambda x: len(str(x).split())) \
    if 'user_story' in screened.columns else \
    (screened['title'].fillna('') + ' ' + screened['description'].fillna('')).apply(lambda x: len(x.split()))

# Snap a Fibonacci (cualquier valor positivo ya mapeó en c3, pero filtrar >40)
post_wc = screened[(screened['word_count'] >= 10) & (screened['word_count'] <= 300)]
post_sp = post_wc[post_wc['storypoints'] <= 40]

post_cap = post_sp.groupby('project_id').apply(
    lambda g: g.sample(min(len(g), 2500), random_state=42)
).reset_index(drop=True)

print(f"N post-cap 2500/proyecto: {len(post_cap)}")