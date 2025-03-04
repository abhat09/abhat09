library(dplyr)
library(lme4)
library(tidyverse)


data = read.csv("preprocess.csv")

# FullName, Country, and Gender are fixed effects with given intercepts
# Apparatus is a random effect
formula <- Score ~ (1|FullName) + (1|Country) + (1|Gender) + Apparatus
mixed_model <- lmer(formula, data = data)

distinct_players <- data %>%
  dplyr::select('Country', 'Apparatus', 'Gender', 'FullName') %>%
  distinct()

# generate 500 values for each athlete in an apparatus
athlete_sim <- simulate(mixed_model, nsim = 500, newdata = distinct_players,
                            # Tell simulate to use the random effects!
                            re.form = NULL)

# apply score bounds
athlete_sim_bound <- apply(athlete_sim, 2, #apply to each column
                               function(x) pmax(pmin(x, 16), 0))

distinct_apps = bind_cols(distinct_players, athlete_sim_bound)

distinct_apps_sim <- distinct_apps %>%
  pivot_longer(cols = sim_1:sim_100,
               names_to = "sim_index", values_to = "sim_value")

# output dataframe
write_csv(distinct_apps,
          "distinct_apps_sim.csv")


