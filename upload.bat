
@echo on
set path=D:\Anaconda3;D:\Anaconda3\Library\bin;D:\Anaconda3\Scripts;D:\Anaconda3\condabin;%path%
path

python setup.py sdist

twine check dist/*

twine upload dist/*

rd /s /Q dist

rd /s /Q perovskite.egg-info

pause

pause

exit