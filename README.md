## Project Proposal ISSR Task work
**AI-Powered Behavioural Analysis for Suicide Prevention, Substance Use, and Mental Health Crisis Detection with Longitudinal Geospatial Crisis Trend Analysis**

This repo contains the completed work for tasks given by Humane AI ISSR(under The university of Alabama) for GSOC-2025: [Task-Docs](https://docs.google.com/document/d/e/2PACX-1vQfC8gkrSx_ycYkIOdae5sJ-fuqn2UA9nLtGqA5egBuwNKMNZpi_NBR0MRnnqdWt8WYqznE6x9_DIO0/pub)


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

```
In the main directory create .env file and define these variables

- CLIENT_SECRET
- CLIENT_ID

you can get them for free via Reddit

These are required only for reddit extraction code, rest scripts can run smoothly,
Provided you have download the essential files.
```

Due to GitHub **LFS** limit, files are removed from the repo,

You can download all the files that were generated via:

```sh
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_classified.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_extracted_unbiased_locations.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_preprocessed.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postswith_comments.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_locations_extracted.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_locations_geocoded.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_terms_bert.csv

```

Advised If you want to analyze, download only these output file:

```sh
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postsV1_extracted_unbiased_locations.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_terms_bert.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/crisis_locations_geocoded.csv
wget https://huggingface.co/datasets/Arpit-Bansal/ISSR/resolve/main/mental_health_postswith_comments.csv

```

These Files are the output work, although they can be reproduced by scripts proivded, it will be a time consuming process.

**scripts/ folder contain the same code as script.ipynb file, Just divided into files or Cells and markdowns respectively.**

Plot and *.csv files may not end up with same name as due provided in Repo

### Plots
*plots/ folder contain plots in .html*
*plots_png/ folder contains same but in png - for better experience.*

Personal advise: Go via plots_png - save time
For more see Plots.md

## Better Conclusion/inference from work Reading
Rather than going via code.

It's advisable to go through markdowns in **script.ipynb** to get grasp of the conclusion we can draw from this analysis.
OR i was able to grasp from this analysis.

And What things to be Aware of/take care of in Project Work.

Don't forget to checkout Plots.md


