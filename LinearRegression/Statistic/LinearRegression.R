library(readr)
Advertising <- read_csv("Statistical learning/Advertising.csv")
View(Advertising)

names(Advertising)

x <- Advertising$TV
y <- Advertising$sales

x_bar <- mean(x)
y_bar <- mean(y)

b1_hat <- sum((x-x_bar) * (y-y_bar)) / sum((x-x_bar)^2)
b0_hat <- y_bar - b1_hat*x_bar

y_hat <- b0_hat + b1_hat*x
e <- y - y_hat
se_b1_2 <- var(e)/sum((x-x_bar)^2)
se_b1 <- sqrt(se_b1_2)

c_inter <- c(b1_hat-2*se_b1, b1_hat+2*se_b1)

plot(Advertising$TV, Advertising$sales, col='red', pch=16)
abline(a=b0_hat, b=b1_hat, lwd=2, col='blue')
segments(Advertising$TV, Advertising$sales, Advertising$TV, y_hat, lty='dotted')


#plot(Advertising$radio, Advertising$sales, col='red')
#plot(Advertising$newspaper, Advertising$sales, col='red')