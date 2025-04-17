# Augmenting Few-Shot Fault Diagnosis in Photovoltaic Arrays Using Generative Models

This repository contains the source code and related materials for the research paper titled "Augmenting Few-Shot Fault Diagnosis in Photovoltaic Arrays Using Generative Models".

## Description

The project focuses on improving fault diagnosis techniques for photovoltaic (PV) arrays, particularly in scenarios where only a limited amount of labeled fault data (few-shot) is available. We propose using generative models to augment the training data, thereby enhancing the performance and robustness of fault diagnosis models.

This codebase includes implementations of:
*   Data preprocessing steps for PV array data.
*   Generative models used for data augmentation.
*   Few-shot learning algorithms for fault diagnosis.
*   Evaluation scripts and metrics.

## Getting Started

To get started with this project, clone the repository and set up the Python environment.

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
# It is recommended to use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Prerequisites

This project requires Python 3.x and the following libraries:
*   NumPy
*   Pandas
*   Scikit-learn
*   PyTorch (or TensorFlow/Keras, depending on implementation)
*   Matplotlib (for visualizations)

Install all dependencies using pip:
```bash
pip install -r requirements.txt
```
Ensure you have access to the necessary PV array datasets. (Specify dataset source or access method if applicable).

### Usage

Follow these examples to run the core components of the project:

1.  **Data Preprocessing:**
    ```bash
    python preprocess_data.py --input_dir path/to/raw_data --output_dir path/to/processed_data
    ```

2.  **Train Generative Model:**
    ```bash
    python train_generator.py --data_path path/to/processed_data/train.csv --model_save_path path/to/save/generator.pth --epochs 100
    ```

3.  **Augment Data:**
    ```bash
    python augment_data.py --generator_path path/to/save/generator.pth --original_data path/to/processed_data/few_shot_train.csv --output_file path/to/augmented_data.csv --num_samples 1000
    ```

4.  **Train Fault Diagnosis Model:**
    ```bash
    python train_classifier.py --train_data path/to/augmented_data.csv --val_data path/to/processed_data/validation.csv --model_save_path path/to/save/classifier.pth
    ```

5.  **Evaluate Model:**
    ```bash
    python evaluate_model.py --model_path path/to/save/classifier.pth --test_data path/to/processed_data/test.csv
    ```
*(Adjust script names and arguments based on your actual implementation)*

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes relevant tests if applicable.

## License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


