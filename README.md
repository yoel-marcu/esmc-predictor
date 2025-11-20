\# ESMC-300m Model Predictions



This repository contains scripts to run predictions using ESMC-300m based models. It supports models utilizing \*\*Mean\*\* or \*\*Median\*\* pooling strategies.



\## Setup



1\. \*\*Clone the repository:\*\*

&nbsp;  ```bash

&nbsp;  git clone <your-repo-url>

&nbsp;  cd <your-repo-name>

&nbsp;  ```

2\. \*\*Install dependencies:\*\*

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

&nbsp;  ```



\*\*Usage\*\*

Place your input FASTA file in the directory (or anywhere accessible).

```bash

python predict.py \\

&nbsp; --fasta inputs.fasta \\

&nbsp; --output results.csv \\

&nbsp; --weights\_dir weights

```



\*\*Arguments\*\*

```bash

--fasta: Path to the sequences file.

--weights\_dir: Folder containing .pt model files (default: weights/).

--output: Name of the result file.

--batch\_size: Batch size for ESMC embedding (adjust based on GPU memory).



```











