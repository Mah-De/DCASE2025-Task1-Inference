## Evaluation Workflow

1. **Prepare the Evaluation Dataset**  
   We download the evaluation dataset from Kaggle and convert all audio files into mel spectrograms. For ease of use, all mel spectrograms are combined into a single `.npz` file:  
   👉 [DCASE2025 Evaluation Dataset](https://www.kaggle.com/datasets/mahdyr/dcase2025-evaluation-dataset )  
   ⚠️ **Note:** Due to limited computational resources, the dataset had to be split into two parts.

2. **Generate Predictions Using Student Models**  
   Using the trained student models available here:  
   👉 [Final Student Models](https://www.kaggle.com/models/mahdyr/dcase2025-task1-models/pyTorch/final_student_models )  
   we run inference and generate the final submission CSV file using this notebook:  
   👉 [`dcase2025-evaluation-notebook.ipynb`](https://www.kaggle.com/code/mahdyr/dcase2025-evaluation-notebook )
