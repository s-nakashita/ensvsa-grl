library(tinytable)
options(tinytable_tt_digits=3)
options(tinytable_format_num_fmt="significant_cell")
options(tinytable_html_mathjax=TRUE)
library(webshot2)

figdir <- "."

nm <- 3
fname <- paste(figdir, "wgtcorr_D1-D2.csv", sep = "/")
data <- read.csv(fname)
cor12 <- data$corr[1:nm]
fname <- paste(figdir, "wgtcorr_D1-D3.csv", sep = "/")
data <- read.csv(fname)
cor13 <- data$corr[1:nm]
fname <- paste(figdir, "wgtcorr_D2-D3.csv", sep = "/")
data <- read.csv(fname)
cor23 <- data$corr[1:nm]
fname <- paste(figdir, "eig_D1.csv", sep = "/")
data <- read.csv(fname)
mode <- data$X[1:nm]
eigd1 <- data$eig[1:nm]
cond1 <- data$contrib[1:nm]
totd1 <- sum(data$eig)
fname <- paste(figdir, "eig_D2.csv", sep = "/")
data <- read.csv(fname)
eigd2 <- data$eig[1:nm]
cond2 <- data$contrib[1:nm]
totd2 <- sum(data$eig)
fname <- paste(figdir, "eig_D3.csv", sep = "/")
data <- read.csv(fname)
eigd3 <- data$eig[1:nm]
cond3 <- data$contrib[1:nm]
totd3 <- sum(data$eig)

data = data.frame( 
  mode = c('total',mode), 
  eig_D1 = c(totd1,eigd1), 
  con_D1 = c(NA,cond1), 
  eig_D2 = c(totd2,eigd2), 
  con_D2 = c(NA,cond2), 
  eig_D3 = c(totd3,eigd3), 
  con_D3 = c(NA,cond3), 
  corrD1_D2 = c(NA,cor12),
  corrD1_D3 = c(NA,cor13), 
  corrD2_D3 = c(NA,cor23) 
  )

tt1 <- data |>
  `colnames<-`(c("mode",rep(c("$\\lambda$","fraction"),3),"D1--D2","D1--D3","D2--D3")) |>
  tt(width=1.5) |>
    group_tt(j=list("D1"=2:3,"D2"=4:5,"D3"=6:7,"correlation"=8:10)) |>
  format_tt(
    i=c(2:4),
    j=c(1:10),
    math=TRUE,
    replace = ''
  ) |>
  format_tt(
    i=1,
    j=c(2,4,6),
    digits = 4,
    math=TRUE,
    replace = ''
  ) |>
  format_tt(
    i=c(2:4),
    j=c(3,5,7),
    fn = function(x) paste(sprintf('%.1f',x*100), "%"),
    escape = TRUE
  ) |>
    style_tt(
    i=c(1:4),
    j=c(1:10),
    align='r'
  ) |>
  style_tt(
    i=c(-1,0),
    align='c'
  )
tt1 |> save_tt(paste(figdir,"TableS1.png",sep="/"),overwrite=TRUE)
