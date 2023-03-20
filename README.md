# NLP_trend_stocks

# Setup

Windows:

Fasttext is currently difficult to compile on Windows 10/11. This could limit the selection of python versions to ones which have precompiled versions of fasttext (currently python<3.10). Additionally, tensorflow==2.10.1 is the last supported native version on windows 10/11. If using linux, you could probably still use the latest versions of all packages.

cd {path_to_project_folder}

conda create --prefix {path_to_project_folder}/.env/  -c esri -c anaconda -c nvidia -c conda-forge fasttext python=3.9 cudnn cudatoolkit tensorflow=2.10.0 transformers sentencepiece tensorflow-probability

conda activate {path_to_project_folder}/.env

pip install -r requirements.txt

# Running

The first time NLP_stock_prediction.ipynb runs it will:

    1) Ask for a password to encrypt API keys with

    2) Ask for Quandle Nasdaq API key (student + some other account are free)

    3) Ask for Kagle username and API key. Type in username + space + API key. Don't include quotation marks or extra spaces. It will be space deliminated and split at spaces. On future runs, it will only ask for the password to decrypt the API keys.

    4) It will then download the data, preprocess it into a more space efficient format, and do tokenization on the text corpi using fasttext.

    5) Once complete, each step will not repeat itself again if sucesfull. If a step crashes, simply restart the program and it will likely work the 2nd time. Some of the steps are less resource intensive if loading from the cached results instead of continuing running after the initial computation.
