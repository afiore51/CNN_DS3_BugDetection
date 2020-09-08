
clear

echo Installing components...

pip install virtualenv

python3 -m venv ./venv


source venv/bin/activate

pip install -r requirement.txt

read -p "Press enter to continue"
