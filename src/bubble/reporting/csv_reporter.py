import os
import pandas as pd
from typing import Dict, Any

class CsvReporter:
    def save_results(self, metrics: Dict[str, Any], model_name: str, output_file: str) -> None:
        new_row = {
            'Model': model_name,
            'Accuracy': round(metrics['accuracy'], 4),
            'Precision': round(metrics['precision'], 4),
            'Recall': round(metrics['recall'], 4),
            'F1-Score': round(metrics['f1_score'], 4)
        }
        
        df_new = pd.DataFrame([new_row])
        
        if os.path.exists(output_file):
            df_new.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df_new.to_csv(output_file, index=False)
            
        print(f'Results saved/updated in {output_file}')