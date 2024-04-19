# cs-responses

First things first, lets get the dependancies sorted. 

```
cd cs-responses

pip install virtualenv

python3.9 -m venv env

source env/bin/activate

pip install -r requirements.txt
```
<br>
Directory structure:

```
| -- data
    | -- csv files
    | -- json files
| -- src
    | -- exp : temp experimental folder
    | -- data_handling : folder to hold code related to data processing/handling
    | -- metrics : folder to hold metrics computation
    | -- evaluations
```


Scoring Flow:
![Scoring](/data/scoring.png)