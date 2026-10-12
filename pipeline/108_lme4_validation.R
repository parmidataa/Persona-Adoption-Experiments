# 108_lme4_validation.R
# Refits the cells exported by 108_export_lme4_validation.py with lme4
# (crossed random intercepts for text and annotator). LRTs use ML fits;
# variance components for the ceiling use REML.
# Writes: results_lme4_validation_lme4.csv
library(lme4, lib.loc = "~/Rlibs")
fast <- read.csv("results_lme4_validation_fast.csv", stringsAsFactors = FALSE)
ctrl <- lmerControl(calc.derivs = FALSE)
prep <- function(d) {
  for (c in setdiff(names(d), c("label", "age_centered"))) d[[c]] <- factor(d[[c]])
  d
}
out <- list()
for (i in seq_len(nrow(fast))) {
  r <- fast[i, ]
  d <- prep(read.csv(file.path("/tmp/lme4val", paste0(r$cell, ".csv")), stringsAsFactors = FALSE))
  if (startsWith(r$cell, "ceil_")) {
    m <- lmer(as.formula(paste("label ~", r$base, "+ (1|text_id) + (1|annotator_id)")),
              data = d, REML = TRUE, control = ctrl)
    vc <- as.data.frame(VarCorr(m))
    out[[i]] <- data.frame(cell = r$cell,
                           var_text = vc$vcov[vc$grp == "text_id"],
                           var_ann = vc$vcov[vc$grp == "annotator_id"],
                           var_e = vc$vcov[vc$grp == "Residual"],
                           llf_base = NA, llf_full = NA, df = NA, lr = NA, p = NA)
  } else {
    fb <- as.formula(paste("label ~", r$base, "+ (1|text_id) + (1|annotator_id)"))
    ff <- as.formula(paste("label ~", r$base, "+", r$added, "+ (1|text_id) + (1|annotator_id)"))
    mb <- lmer(fb, data = d, REML = FALSE, control = ctrl)
    mf <- lmer(ff, data = d, REML = FALSE, control = ctrl)
    a <- anova(mb, mf)
    out[[i]] <- data.frame(cell = r$cell, var_text = NA, var_ann = NA, var_e = NA,
                           llf_base = as.numeric(logLik(mb)), llf_full = as.numeric(logLik(mf)),
                           df = a$Df[2], lr = a$Chisq[2], p = a$`Pr(>Chisq)`[2])
  }
  print(out[[i]])
}
write.csv(do.call(rbind, out), "results_lme4_validation_lme4.csv", row.names = FALSE)
