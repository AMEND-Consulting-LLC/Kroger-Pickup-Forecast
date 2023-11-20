# [ repo name here ]
[ project title here ]

## Project Overview

Project Background: 

Project/Problem Statement/Goal: 

## Setup

### Development
Development will take place in [ choose .py (python script), .ipynb (jupyter) ].

### Virtual Environment
A python virtual will be used for this project. To set this up, view the [AMEND Python Virtual Environment Guide](https://github.com/AMEND-Consulting-LLC/amend-standards/blob/main/python/python_virtualenv.md).

### Dependencies
- Pathlib for all filepath references
- Keyring for all passwords

### Requirements.txt
To install required dependencies from requirements.txt:
1. open CMD 
2. navigate to the directory of your github repo `cd github\your-repository`
3. activate the virtual environment `.\env\scripts\activate`
4. install via pip `pip install -r requirements.txt`

## Navigating this Repo

### data
Contains input data. All static data (csv, xlsx, etc) used in the project will come from here. Pickle (.pkl) files created from any Data Engineering steps will also be stored/exported here. The files in this folder are included in .gitignore and will not push to Github.

### exports
Contains all exported files created during this project. The files in this folder are included in .gitignore and will not push to Github.

### sql
Contains all .sql files used in this project.

## Contributing

### Python
This Python Project follows the style guide outlined in [PEP8](https://www.python.org/dev/peps/pep-0008/) as well as the [AMEND Python Standards](https://github.com/AMEND-Consulting-LLC/amend-standards/blob/main/python/python_standards.md) outlined in the [AMEND Standards Repo](https://github.com/AMEND-Consulting-LLC/amend-standards).

It is recommended to use Black as an autoformatter to ensure compliance with these standards. Documentation for autoformatting can be found [here](https://github.com/AMEND-Consulting-LLC/amend-standards/blob/main/python/python_autoformatting.md)

### Version Control
This GitHub repo follows the [AMEND Version Control Standards](https://github.com/AMEND-Consulting-LLC/amend-standards/tree/main/version_control) outlined in the [AMEND Standards Repo](https://github.com/AMEND-Consulting-LLC/amend-standards).