# GOPH547-W2026-Lab01-stAM

##GOPH 547

*Semester:* W2026
*Instructor:* B. Karchewski
*Author:* Amelia Morris

A repository designed to for forward modelling corrected gravity data, along with contributing to a better understanding of the effects of gravity potential and gravity effects on grids of synthetic data radiating away from a central point mass.

To download or clone the repository, click on the green "< > Code" button above the repository files. You can either download entire repository as a ZIP file or single files if selected, or you can clone them using HTTPS, SSH, or GitHub CLI through Python, making sure to navigate to the directory you'd like to store the files to beforehand, using the *cd /path/to/directory/* function. You can clone these files through *git clone https://github.com/Ames-777/GOPH547-W2026-Lab01-stAM.git* so that you can automatically pull updates through *git pull*.

To set up a virtual environment, which is recommended when running these files, navigate into the directory that the downloaded files are stored to, using the *cd /path/to/directory/* function. Use either *python -m venv .venv* or *python -m virtualenv .venv* depending on if you are using virtualenv or not, to create the virtual environment. To run the virtual environment, run *source ./.venv/bin/activate* after which a (.venv) should appear on the left hand side of the Terminal, indicating that the virtual environment is now activated.

To ensure the script runs properly, make sure to install the following packages by running *pip install numpy matplotlib setuptools*

Once the packages are installed, you can run the files by inputting *python driver.py*

The script *driver_single_mass.py* (contained within the *examples/* folder) is designed to be used to better understand the effects of gravity potential and gravity effect of a single mass from different distances away from the survey point. To run the code, run *python driver_single_mass.py*

The expected output will contain two figures, each containing six subplots demonstrating the changes in gravity potential and gravity effects with varying distance from a point mass with different grid spacing between the two figures.

The script *driver_multi_mass.py* (contained within the *examples/* folder) is designed to be used to better understand the effects of gravity potential and gravity effects due to multiple objects at varying distances from the survey point. To run the code, run *python driver_multi_mass.py*

The script *driver_mass_anomaly.py* (contained within the *examples/* folder) is designed to forward model given distributed density anomaly data and explore the effects of gravity potential and gravity effects at varying distances from the survey point. To run the code, run *python driver_mass_anomaly.py*