def ann_model(df):
    target_column = [col for col in df.columns if 'score' in col.lower()]