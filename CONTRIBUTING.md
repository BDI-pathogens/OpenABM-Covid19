# Contributing

This document outlines the basic workflow for contributing to the OpenABM-Covid19 model.  Please also see notes at the bottom of this document.  

1. Create a fork of the OpenABM-Covid19 repository and clone this repository.  

```bash
git clone https://github.com/USERNAME/OpenABM-Covid19.git
```

2. Add the upstream repository for OpenABM-Covid19

```bash
cd OpenABM-Covid19
git remote add upstream https://github.com/BDI-pathogens/OpenABM-Covid19.git
```

3. Sync upsteam with the master branch of your fork

```bash
git checkout master
git fetch upstream master
git merge master
```

4. Make changes for any additions to the model in a branch on your fork

```bash
git checkout -b feature:speed_up_fix
<change>
# Add any new files with 'git add' etc
git commit -m "Useful commit message"
```

5. Push changes to your fork
```bash
git push origin feature:speed_up_fix
```

6. Head to the upstream repository at https://github.com/BDI-pathogens/OpenABM-Covid19 using a browser and a "pull request" button should be available.  Click that and follow the prompts, tagging of one of the OpenABM-Covid19 core team in the pull request for review (roberthinch, p-robot, danielmonterocr, brynmathias).  


**Notes**

* Any changes to the model need to pass all test (see testing guidelines in the main README.md) and new features of the model need to provide new tests of the added functionality.  
* PRs are only merged in the master (release) branch after all tests have passed.  
* If the contribution/PR changes the model parameterisation please add the new parameter(s) to `tests/data/baseline_parameters_transpose.csv`, including a description, default value, and reference (if possible) for the new parameters.  Calling `python python/transpose_parameters.py` from the main project directory will create markdown documents of the parameter files and flat files `tests/data/baseline_parameters.csv` that can be included in the PR.  
* Parameters that wish to be updated via the Python interface need to be added to the list `PYTHON_SAFE_UPDATE_PARAMS` within `OpenABM-Covid19/src/COVID19/model.py`.  
* If the contribution/PR changes the Python/C interface please include documentation.  
