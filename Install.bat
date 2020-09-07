
CLS
@echo off

echo Installing components

CALL python -m pip install virtualenv
CALL python -m venv venv
cmd /c  "venv\Scripts\activate.bat &&  pip install -r requirementTest.txt && echo All done"
PAUSE