USOPC Data Challenge Zip File 

This folder contains a subfolder titled "visuals" of RMD files and datasets that we used for running EDA to gain insight into constructing our model. This subfolder also contains files we used to generate the matplots in our Results section of our report. There is also a subfolder containing the analytical model for the data challenge titled "Model". We also included the Report pdf.

To run the model:
1. Unzip the folder
2. In your terminal or coding environment, navigate to the "Model" subfolder directory
3. Run the preprocessing.py file with the following command " python preprocessing.py "
4. Run the lmer.R file with the following command " Rscript lmer.R "
5. Run the simulation file with the following command " python simulation.py "

What the files do: 
- Preprocessing.py: This file edits the dataset for consistency and creates a csv file titled "preprocess.csv" containing the variables relevant to the LMER model. 
- LMER.R : This file constructs the linear mixed effects regression model and creates a new dataset in the output csv titled "distinct_apps_sim.csv"
- Simulation.py: This file contains the bulk of our analytical model. We first fix the players from the 11 non-USA qualifying countries and the 36 individuals, read in the LMER dataset, create the combinations of USA athletes, run 100 simulations of qualifying and final rounds for each USA team combination for both genders. At the end, the file prints out the most succesful women's and men's team and the average weighted medal counts for both of the teams respectively.

** Note that the simulation.py file may take a few hours to run. It runs through 168 combinations of athletes with approximately 2 minutes per combination.**


Libraries to download: 
Python: Matplot, NumPy, Pandas
R: Dplyr, lme4, tidyverse