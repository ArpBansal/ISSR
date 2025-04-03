## Project Proposal ISSR Task work
**AI-Powered Behavioural Analysis for Suicide Prevention, Substance Use, and Mental Health Crisis Detection with Longitudinal Geospatial Crisis Trend Analysis**

This repo contains the completed work for tasks given by ISSR(under The university of Alabama) for GSOC-2025: [Task-Docs](https://docs.google.com/document/d/e/2PACX-1vQfC8gkrSx_ycYkIOdae5sJ-fuqn2UA9nLtGqA5egBuwNKMNZpi_NBR0MRnnqdWt8WYqznE6x9_DIO0/pub)


### Setup
```sh
git clone https://github.com/ArpBansal/ISSR
cd ISSR
```

**ON Linux Ubuntu**

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
**On windows**
```sh
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Due to GitHub **LFS** limit, files are removed from the repo,

You can download them via:

```sh
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_classified.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_extracted_unbiased_locations.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_preprocessed.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postswith_comments.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_locations_extracted.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_locations_geocoded.csv
```

These Files are the output work, although they can be reproduced by scripts proivded, it will be a time consuming process.

**scripts/ folder contain the same code as script.ipynb file, Just divided into files or Cells and markdowns respectively.**