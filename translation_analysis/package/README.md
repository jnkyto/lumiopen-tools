# Translation analysis script package

---

### Setting up

Dataset layouts:
 - ELRC-Norden
   1) Download bilingual moses files 
        from https://opus.nlpl.eu/ELRC-www.norden.org/en&fi/v1/ELRC-www.norden.org
   2) Extract to <proj_root>/data/elrc-norden_en-fi/. Ensure no subfolder.
 - ELRC-fi-info
   1) Download bilingual moses files
        from https://opus.nlpl.eu/ELRC-Finnish_Information/en&fi/v1/ELRC-Finnish_Information
   2) Extract to <proj_root>/data/elrc-fi_info_en-fi/. Ensure no subfolder.
 - Europarl
   - Dataset should be downloaded automatically. If not, extract HF dataset files to 
        <proj_root>/data/europarl_en-fi/.
 - Tatoeba
   1) Download tatoeba-test-v2023-09-26.eng-fin.txt (any version with the same langs is probably fine)
   2) Put it in <proj_root>/data/tatoeba/. Ensure no subfolder.
 - Ted2020
   1) Download bilingual moses files
        from https://opus.nlpl.eu/TED2020/en&fi/v1/TED2020
   2) Extract to <proj_root>/data/ted2020_en-fi/. Ensure no subfolder.

### Running

**main.py** is the primary entrypoint, which can be run with `python3 main.py -d [dataset:str]
-b [bands:int] -p [per_band:int] -t [threshold:float] -m [min_len:int] --dry (only do sampling)`.

Setting the dataset argument is required, others are optional and have default values set
in the script.

I've migrated to running **main_presampled.py**, after running main.py with the dry-option.
This essentially means that I sample each dataset on my own computer and only do the translation 
on LUMI. Splitting the workload like this means that the resulting .json outputs have to be merged,
which is why **merge_jsons.py** exists. It takes `sampled_file` and `translated_file` as path 
arguments (in that order).

**scoring.py**-script's resource requirements grow linearly with the size of the files. With 
default values for any script in this package, I would say 16 GB of RAM per file is a good 
rule of thumb.

Python 3.12 was used during development so I suggest using that to run these scripts as well.