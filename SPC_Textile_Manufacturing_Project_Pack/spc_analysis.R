# Statistical Process Control for Yarn Tensile Strength
# Dataset is synthetic/illustrative and should be replaced with plant measurements.
# Install once if needed: install.packages(c("readr", "dplyr", "ggplot2"))
library(readr)
library(dplyr)
library(ggplot2)

raw <- read_csv("yarn_tensile_strength_sample_data.csv", show_col_types = FALSE)
stopifnot(all(c("Subgroup", "Specimen", "Tensile_Strength_N") %in% names(raw)))

stats <- raw %>%
  group_by(Subgroup) %>%
  summarise(n = n(),
            xbar = mean(Tensile_Strength_N),
            R = max(Tensile_Strength_N) - min(Tensile_Strength_N),
            .groups = "drop")

# Constants for subgroup size n = 5
A2 <- 0.577; D3 <- 0; D4 <- 2.114
xbarbar <- mean(stats$xbar)
Rbar <- mean(stats$R)
stats <- stats %>%
  mutate(Xbar_CL = xbarbar, Xbar_UCL = xbarbar + A2*Rbar,
         Xbar_LCL = xbarbar - A2*Rbar,
         R_CL = Rbar, R_UCL = D4*Rbar, R_LCL = D3*Rbar)

print(stats)
write_csv(stats, "subgroup_control_chart_statistics_from_R.csv")

p1 <- ggplot(stats, aes(Subgroup, xbar)) +
  geom_line() + geom_point() +
  geom_hline(yintercept=xbarbar, linetype="solid") +
  geom_hline(yintercept=xbarbar + A2*Rbar, linetype="dashed", colour="red") +
  geom_hline(yintercept=xbarbar - A2*Rbar, linetype="dashed", colour="red") +
  labs(title="X-bar Control Chart", x="Subgroup", y="Mean tensile strength (N)") +
  theme_minimal()
ggsave("xbar_control_chart.png", p1, width=9, height=5, dpi=300)

p2 <- ggplot(stats, aes(Subgroup, R)) +
  geom_line() + geom_point() +
  geom_hline(yintercept=Rbar, linetype="solid") +
  geom_hline(yintercept=D4*Rbar, linetype="dashed", colour="red") +
  geom_hline(yintercept=D3*Rbar, linetype="dashed", colour="red") +
  labs(title="R Control Chart", x="Subgroup", y="Within-subgroup range (N)") +
  theme_minimal()
ggsave("r_control_chart.png", p2, width=9, height=5, dpi=300)

# Basic one-sided tabular CUSUM on subgroup means, k = 0.5 sigma, h = 5 sigma.
# sigma estimate based on R-bar / d2 for n=5 (d2 = 2.326).
sigma_hat <- Rbar / 2.326
k <- 0.5 * sigma_hat
h <- 5 * sigma_hat
cusum <- stats %>% arrange(Subgroup) %>%
  mutate(Cplus = 0, Cminus = 0)
cp <- 0; cm <- 0
for (i in seq_len(nrow(cusum))) {
  z <- cusum$xbar[i] - xbarbar
  cp <- max(0, cp + z - k)
  cm <- max(0, cm - z - k)
  cusum$Cplus[i] <- cp
  cusum$Cminus[i] <- cm
}
cusum$Signal <- ifelse(cusum$Cplus > h | cusum$Cminus > h, "Investigate", "No signal")
write_csv(cusum, "cusum_results.csv")
p3 <- ggplot(cusum, aes(Subgroup)) +
  geom_line(aes(y=Cplus, colour="Positive CUSUM")) +
  geom_line(aes(y=Cminus, colour="Negative CUSUM")) +
  geom_hline(yintercept=h, linetype="dashed") +
  labs(title="Tabular CUSUM Chart", x="Subgroup", y="Cumulative deviation", colour="Series") +
  theme_minimal()
ggsave("cusum_control_chart.png", p3, width=9, height=5, dpi=300)
