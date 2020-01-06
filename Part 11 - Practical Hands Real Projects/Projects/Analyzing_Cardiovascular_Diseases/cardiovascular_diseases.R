
files <- list.files(path="./sources", pattern="*.csv", full.names=TRUE, recursive=FALSE)
list_df <- list()

for (i in 1:length(files)){
  print(i)
  list_df[[i]] <- read.csv(files[[i]])
}
  
with(list_df[[1]]{
  summary(list_df[[1]])
})

for (i in list_df){
  i$Time <- factor(i$Time, ordered = TRUE)
}


# plot(list_df[[1]]$Time, list_df[[1]]$Ads)
# abline(lm(list_df[[1]]$Time ~ list_df[[1]]$Ads))

dose <- c(20,30,40,45,60)
drugA <- c(16,20,27,40,60)
drugB <- c(15,18,25,31,40)

#Graphical Elements in R.

# dev.new()
# plot(dose, drugA, type = "b")
# hist(dose)
# 
# dev.new()
# plot(dose, drugB, type = "b")
# 
# dev.new()
# pdf("time_ads.pdf")
#   plot(list_df[[1]])
#   title("Time vs Ads Plot")
# dev.off()                
#   
# dev.new()
#   hist(list_df[[1]]$Time)
# dev.new()
#   boxplot(list_df[[1]])
#   

##Setting options for plotting the graphs.
opar <- par(no.readonly=TRUE)
par(lty=2, pch=17)
plot(dose, drugA, type = "b")
par(cex=1.5)
plot(dose, drugA, type = "b")

dev.new()
par(cex=2.5)
plot(dose, drugA, type = "b")
dev.off()

dev.new()
plot(dose, drugA, type = "b", lty=3, pch = 16, lwd=2)

##Specifying Colors.

library(RColorBrewer)
n <- 7
mycolors <- brewer.pal(n, "Set1")
display.brewer.pal(n, "Set1")

n <- 12
mycolors2 <- brewer.pal(n, "Set3")
display.brewer.pal(n, "Set3")

barplot(rep(1,n), col=mycolors2)

n <- 10
mycolors <- rainbow(n)
pie(rep(1,n), labels=mycolors, col=mycolors)
mygrays <- gray(0:n/n)
pie(rep(1,n), labels=mygrays, col=mygrays)


#Text Characteristics. 
par(font.lab=3, cex.lab=1.5, font.main=4, cex.main=2)

windowsFonts(
  A=windowsFont("Arial Black"),
  B=windowsFont("Bookman Old Style"),
  C=windowsFont("Comic Sans MS")
  )

par(font.lab=3, cex.lab=1.5, font.axis = 2, font.main=4, cex.main=2, family="A")
pie(rep(1,n), labels=mycolors, col=mycolors)
plot(dose, drugA, type="b")

####Graph and margin dimensions

opar <- par(no.readonly=TRUE)
par(pin=c(2,3))
par(lwd=2, cex=1.5)
par(cex.axis=.75, font.axis=3)
plot(dose, drugA, type="b", pch=19, lty=2, col="red")
plot(dose, drugA, type="b", pch=23, lty=6, col="blue", bg="green")
par(opar)#We restore the previous settings. 

####Adding text, customized axes, and legends. 
plot(dose, drugA, type="b",
     col="red", lty=2, pch=2, lwd=2,
     main="Clinical Trials for Drug A", 
     sub="This is hypothetical data",
     xlab="Dosage", ylab="Drug Response",
     xlim=c(0,60), ylim=c(0,70))

dev.new()
plot(dose, drugB, type="b",
     col="red", lty=3, pch=3, lwd=2,
     main="Clinical Trials for Drug B",
     sub="This is hypothetical data",
     xlab="Dosage", ylab="Drug Response",
     xlim=c(0,60), ylim=c(0,70))

title(main="Clinical Trials for Drug A", col.main="red",
      sub="My subtitle", col.sub="blue",
      xlab = "My X label", ylab="My Y label",
      col.lab="green", cex.lab=0.75)

dev.off()

x <- c(1:10)
y <- x
z <- 10/x
opar <-par(no.readonly=TRUE)

par(mar=c(5,4,4,8) + 0.1)#Increases margins

plot(x,y, type="b",
     pch=21, col="red",
     yaxt="n", lty=3, ann=FALSE)#Disables default label display.

lines(x,z, type="b", pch=22, col="blue", lty=3)

axis(2,at=x, labels=x, col.axis="red", las=2)
axis(4, at=z, labels=round(z,digits=2),
     col.axis="blue", las=2, cex.axis=0.7, tck=-.01)

mtext("y=1/x", side=4, line=3, cex.lab=1, las=2, col="blue")
title("An Example of Creative axes",
      xlab="X values",
      ylab="Y=X")
par(opar)

dev.off()

##Reference Lines.
x = c(1:10)
y = x

par(pin=c(5,2), mai=c(1,1,1,1))
plot(x,y, type="b", xlab = "X label", ylab = "Y label", col.axis = "blue",
     main="Simple Plot x vs y", col="orange")

abline(h=c(1,3,5), lty=2)
abline(v=seq(3,7,2), lty=2, col="blue")

dev.off()


##Legend. 

opar <- par(no.readonly=TRUE)
par(pin=c(8,4), mai=c(1,1,1,1))
par(lwd=2, cex=0.75, font.lab=2)
axis(4,at=drugA, labels=drugA, col.axis="red", las=2)
plot(dose, drugA, type="b",
     pch=15, lty=1, col="red", ylim=c(0,60),
     main="Drug A vs Drug B",
     xlab="Drug Dosage", ylab="Drug Response")

axis(3, at=drugB, labels=drugB, col.axis="blue", las=2)
lines(dose, drugB, type="b",
      pch=17, lty=2, col="blue")

abline(h=c(30), lwd=1.5, lty=2, col="black")

library(Hmisc)
minor.tick(nx=3, ny=3, tick.ratio=0.5)
legend("topleft", inset=.05, title="Drug Type" ,c("A", "B"), lty=c(1,2),
       pch=c(15,17), col=c("red", "blue"))

par(opar)

###Text Annotations. 
attach(mtcars)
plot(wt,mpg, 
     main="Mileage vs Car Weight",
     xlab="Weight", ylab="Mileage",
     pch=18, col="blue")
text(wt,mpg,
     row.names(mtcars),
     cex=0.6, pos=4, col="red")
detach(mtcars)


###
opar <- par(no.readonly=TRUE)
par(cex=1.5, font.lab=4)
plot(1:7, 1:7, type="n")
text(3,3, "Example of default text")
text(4,4,family="mono", "Example of mono-spaced text")
text(5,5,family="serif", "Example of Serif Text.")
par(opar)


###Combining Graphs. 
attach(mtcars)
opar <- par(no.readonly=TRUE)
par(mfrow=c(2,2), mai=c(1,1,0.5,0.5), pin=c(1.5,1.5))
plot(wt, mpg, main="Scatterplot of wt vs mpg")
plot(wt, disp, main="Scatterplot of wt vs disp")
hist(wt, main="Histogram of wt")
boxplot(wt, main="Boxplot of wt")
par(opar)
detach(mtcars)


##More Combination of Graphs. 
attach(mtcars)
layout(matrix(c(1,1,2,3), 2, 2, byrow=TRUE))
hist(wt)
hist(mpg)
hist(disp)
detach(mtcars)

##Specify widths and heights to control the size of each figure. 
attach(mtcars)
layout(matrix(c(1,1,2,3),2, 2, byrow=TRUE),
       widths=c(3,1), heights=c(2,2))

hist(wt)
hist(mpg)
hist(disp)
detach(mtcars)

##Create a figure arrangement with fine control

opar <- par(no.readonly=TRUE)
par(fig=c(0,0.8,0,0.8))
plot(mtcars$mpg, mtcars$wt, xlab="Miles per Gallon", ylab="Car Weight")

par(fig=c(0,0.8,0.55,1), new=TRUE)
boxplot(mtcars$mpg, horizontal=TRUE, axes=FALSE)

par(fig=c(0.65,1,0,0.8), new=TRUE)
boxplot(mtcars$wt, axes=FALSE)

mtext("Enhanced Scatterplot", side=3, outer=TRUE, line=-3)
par(opar)
dev.off()

#Basic Data Management

manager <- c(1:5)
date <- c("10/24/14","10/28/14","10/01/14","10/12/14","05/01/14")
country <- c("US", "US", "UK", "UK", "UK")
gender <- c("M", "F", "F", "M", "F")
age <- c(32,45,25,39,99)

q1 <- c(5,3,3,3,2)
q2 <- c(4,5,5,3,2)
q3 <- c(5,2,5,4,1)
q4 <- c(5,5,5,NA,2)
q5 <- c(5,5,2,NA,1)

leadership <- data.frame(manager, date, country, gender, age, 
                         q1, q2, q3, q4, q5, stringsAsFactors = FALSE)



##Creating new variables. 
mydata <- data.frame(x1=c(2,2,6,4),
                     x2=c(3,4,2,8))
mydata$sumx <- mydata$x1 + mydata$x2
mydata$meanx <- (mydata$x1 - mydata$x2)/2

mydata <- transform(mydata, sumx=x1 + x2, meanx=(x1+x2)/2)


#Recoding variables.

leadership$age[leadership$age == 99] <- NA
leadership$agecat[leadership$age >=55 & leadership$age <=75] <- "Middle aged"
leadership$agecat[leadership$age < 55] <- "Young"

leadership <- within(leadership, {
                     agecat <- NA
                     agecat[age > 75] <- "Elder"
                     agecat[age >= 55 & age <= 75]  <- "Middle Aged"
                     agecat[age < 55] <- "Young"})   

#Renaming variables. 
names(leadership)
names(leadership)[2] <- "testDate"
names(leadership)[6:10] <- c("item1", "item2", "item3", "item4", "item5")
names(leadership)

install.packages("plyr")
library("plyr")
rename(leadership, c(manager="managerID", country="countryID"))

#Data values
mydates <- as.Date(c("2007-06-22", "2004-02-13"))


strDates <- c("01-05-1965","08-16-1975")
dates <- as.Date(strDates, "%m/%d/%y")
myformat <- "%m/%d/%y"
leadership$testDate <- as.Date(leadership$testDate, myformat)
leadership$testDate

#Sorting Data. 

attach(leadership)
newdata <- leadership[order(gender,age),]
detach(leadership)

attach(leadership)
newdata <- leadership[order(gender, -age),]
detach(leadership)

####Subsetting datasets.

newdata <- leadership[, c(6:10)]
myvars <- c("item1", "item2", "item3","item4", "item5")
newdata <- leadership[myvars]
myvars <- paste("item", 1:5, sep="")
newdata <- leadership[myvars]

myvars <- names(leadership) %in% c("item3", "item4")
newdata <- leadership[!myvars]

##This works the same...
newdata <- leadership[c(-8,-9)]
leadership$item3 <- leadership$item4 <- NULL


###Selecting observations. 
newdata <- leadership[1:3,]
newdata <- leadership[leadership$gender == "M" & leadership$age >30,]

##Use attach so you dont have to prepend variable names with data-frame names. 
attach(leadership)
newdata <- leadership[gender=="M" & age>30,]
detach(leadership)

leadership <- as.Date(leadership$testDate, "%m/%d/%y")

startdate <- as.Date("2009-01-01")
enddate <- as.Date("2009-10-31")

newdata <- leadership[which(leadership$testDate >= startdate 
                            & leadership$testDate <=enddate),]


##Using SQL Statements to manipulate data frames. 

install.packages("sqldf")
library("sqldf")

newdf <- sqldf("select * from mtcars where carb=1 order by mpg",
               row.names = TRUE)



sqldf("select avg(mpg) as avg_mpg, avg(disp) as avg_disp, gear
      from mtcars where cyl in (4,6) group by gear")

###Advanced data management. 

x <- c(1:8)
min(x)
max(x)
scale(x, center=TRUE, scale=TRUE)

###Standarize Data. 
newdata <- scale(mydata)
newdata <- scale(mydata)*SD + M


###Probability Functions. 

x <- pretty(c(-3,3), 30)
y <- dnorm(x)

plot(x, y, type="1", xlab="Normal deviate", ylab="Density", yaxs = "i")
pnorm(1.96)
qnorm(.9, mean=500, sd=100)
rnorm(50, mean=50, sd=10)

runiform(5)
runif(5)
set.seed(1234)
runif(5)
set.seed(1234)
runif(5)


library(MASS)
options(digits=3)
set.seed(1234)

mean <- c(230.7,146.7, 3.6)
sigma <- matrix(c(15260.7, 6721.2, -47.1,
                  6721.2, 4700.9, -16.5,
                  -47.1, -16.5, 0.3), nrow=3, ncol=3)

mydata <- mvrnorm(500, mean, sigma)
mydata <- as.data.frame(mydata)
names(mydata) <- c("y", "x1", "x2")
dim(mydata)
head(mydata)

#User-written functions. 
mystats <- function(x, parametric=TRUE, print=FALSE){
  if(parametric){
    center <- mean(x);
    spread <- sd(x);
    
  else{
    center <- median(x);
    spread <- mad(x);
    }
  if(print & parametric){
    cat("Mean=", center, "\n", "SD=", spread, "\n")
  }
  else if(print & !parametric) {
    cat("Median=", center, "\n", "MAD=", spread, "\n")
  }
  result <- list(center=center, spread=spread)
  return(result)
  }
}


set.seed(1234)
x <- rnorm(500)
y <- mystats(x)

y <- mystats(x)
y <- mystats(x, parametric=FALSE, print=TRUE)

mydate <- function(type=long) {
  switch(type, 
         long = format(Sys.time(), "%A %B %d %Y"),
         short = format(Sys.time(), "%m-%d-%y"),
         cat(type, "is not a recognized type\n") 
         #The function enters this third statement only if type is not short or long. 
         )
}

mydate("long")
mydate("short")
mydate()
mydate("medium")


#Transposing rows and columns in R. 

cars <- mtcars[1:5, 1:4]
t(cars)

#Aggregating data. 
options(digits=3)
attach(mtcars)
aggdata <- aggregate(mtcars, by=list(cyl, gear), FUN=mean, na.rm=TRUE)

install.packages("reshape2")
library("reshape2")

Time <- c(1,1,2,2)
ID <- c(1,2,1,2)
X1 <- c(5,3,6,2)
X2 <- c(6,5,1,4)

mydata <- data.frame(ID, Time, X1, X2)
md <- melt(mydata, id=c("ID", "Time"))
newdata <- dcast(md, formula, fun.aggregate)

##Casting
newdata <- dcast(md, formula, fun.aggregate)


##Basic Graphs. 
install.packages("vcd")
library("vcd")
counts <- table(Arthritis$Improved)

barplot(counts, 
        main="Simple Bar Plot",
        xlab="Improvement",
        ylab="Frequency")

dev.new()
barplot(counts, 
        main="Horizontal Bar Plot",
        xlab="Frequency",
        ylab="Improvement",
        horiz=TRUE)
dev.off()

plot(Arthritis$Improved, main="Simple Bar Plot",xlab="Improved",
     ylab="Frequency")

plot(Arthritis$Improved, horiz=TRUE, main="Horizontal Bar Plot",
     xlab="Frequency", ylab="Improved")


##Stacked and grouped bar plots. 
library(vcd)
counts <- table(Arthritis$Improved, Arthritis$Treatment)

barplot(counts, 
        main="Stacked Bar Plot", 
        xlab="Treatment",
        ylab="Frequency",
        col=c("red", "yellow", "green"),
        legend = rownames(counts))

barplot(counts, 
        main="Grouped Bar Plot", 
        xlab="Treatment",
        ylab="Frequency",
        col=c("red", "yellow", "green"),
        legend = rownames(counts), beside=TRUE)




####
states <- data.frame(state.region, state.x77)
means <- aggregate(states$Illiteracy, by=list(state.region), FUN=mean)
means <- means[order(means$x),]
barplot(means$x, names.arg=means$Group.1)
title("Mean Illiteracy Rate")

par(mar=c(5,8,4,2))
par(las=2)

counts <- table(Arthritis$Improved)
barplot(counts, main="Treatment Outcome", horiz=TRUE, cex.names=0.8,
        names.arg=c("No improvement", "Some improvement", "Marked Improvement"))


library(vcd)
attach(Arthritis)
counts <- table(Treatment, Improved)
spine(counts, main="Spinogram Example")
detach(Arthritis)

dev.off()
##Pie Charts. 
par(mfrow=c(2,2))
slices <- c(10,12,4,16,8)
lbls <- c("US", "UK", "Australia", "Germany", "France")
pie(slices, labels=lbls,
    main="Simple Pie Chart")


pct <- round(slices/sum(slices)*100)
lbls2 <- paste(lbls, "", pct, "%", sep="")
pie(slices, labels=lbls2, col=rainbow(length(lbls2)),
    main="Pie Chart with Percentages.")


####3D PIE PLOT
library(plotrix)
pie3D(slices, labels=lbls, explode=0.1, 
      main="3D Pie Chart")
mytable <- table(state.region)
lbls3 <- paste(names(mytable), "\n", mytable, sep="")
pie(mytable, labels=lbls3,
    main="Pie Chart from a Table \n (with sample sizes)")


###CHI-SQUARE TEST OF INDEPENDENCE
library(vcd)
mytable <- xtabs(~Treatment+Improved, data=Arthritis)
chisq.test(mytable)

mytable <- xtabs(~Improved+Sex, data=Arthritis)
chisq.test(mytable)

###FISHER TEST
mytable <- xtabs(~Treatment+Improved, data = Arthritis)
fisher.test(mytable)

###COCHRAN-MANTEL-HAENSZEL TEST
mytable <- xtabs(~Treatment+Improved+Sex, data=Arthritis)
mantelhaen.test(mytable)

####Measures of association. 
library(vcd)
mytable <- xtabs(~Treatment+Improved, data=Arthritis)
assocstats(mytable)


###Covariances and correlations
states <- state.x77[,1:6]
cov(states)

x <- states[,c("Population", "Income", "Illiteracy", "HS Grad")]
y <- states[,c("Life Exp", "Murder")]

cor(x,y)

###The use of partial correlation.

library(ggm)
colnames(states)
pcor(c(1,5,2,3,6), cov(states))


###Significance for correlations. 
cor.test(states[,3], states[,5])
library(psych)
corr.test(states, use="complete")

###Independent t-test
library(MASS)
t.test(Prob ~ So, data = UScrime)

###Dependent t-test
t.test(y1,y2,paired=TRUE)
library(MASS)
sapply(UScrime[c("U1","U2")], function(x)(c(mean=mean(x),sd=sd(x))))

###Wilcox and Kruskal Tests. 
wilcox.test(y ~ x, data)
wilcox.test(y1,y2)

with(UScrime, by(Prob, So, median))

kruskal.test(y ~ A, data)
friedman.test(y ~ A, data)

states <- data.frame(state.region, state.x77)
kruskal.test(Illiteracy ~ state.region, data=states)


###Regression Analysis. 

fit <- lm(weight ~ height, data=women)
summary(fit)
women$weight

fitted(fit)
residuals(fit)
plot(women$height, women$weight,
     xlab="Height (in inches)",
     ylab="Weight (in pounds)")
abline(fit)

###Polynomial Regression

fit2 <- lm(weight ~ height + I(height^2), data=women)
summary(fit2)

plot(women$height, women$weight, 
     xlab="Height (in inches)",
     ylab="Weight (in lbs)")

lines(women$height, fitted(fit2))


####nth-degree polynomial regression.
fit3 <- lm(weight ~ height + I(height^2) + I(height^3), data=women)
library(car)
scatterplot(weight ~ height, data=women, 
            spread=FALSE, smoother.args=list(lty=2), pch=19,
            main="Women Age 30-39",
            xlab="Height (inches)",
            ylab="Weight (lbs.)")


states <- as.data.frame(state.x77[,c("Murder", "Population",
                                     "Illiteracy", "Income", "Frost")])

cor(states)
library(car)
scatterplotMatrix(states, spread=FALSE, smoother.args=list(lty=2),
                  main="Scatter Plot Matrix")

states <- as.data.frame(state.x77[,c("Murder", "Population",
                                     "Illiteracy", "Income", "Frost")])

fit <- lm(Murder ~ Population + Illiteracy + Income + Frost, data=states)
summary(fit)



###Multiple Linear Regressions with interactions. 
fit <- lm(mpg ~ hp + wt + hp:wt, data=mtcars)
summary(fit)
##Visualize interactions between predictor variables. 
# plot(effect(term, mod, ,xlevels), multiline=TRUE)
install.packages("effects")
library(effects)
plot(effect("hp:wt", fit, ,list(wt=c(2.2, 3.2, 4.2)), multiline=TRUE))



##Regression diagnostics. 
fit <- lm(Murder ~ Population + Illiteracy + Income + Frost, data=states)
confint(fit)


##Evaluating the statistical assumptions in regression analysis. 
fit <- lm(weight ~ height, data=women)
par(mfrow=c(2,2))
plot(fit)

###Normality
library(car)
states <- as.data.frame(state.x77[,c("Murder", "Population",
                                     "Illiteracy", "Income", "Frost")])
fit <- lm(Murder ~ Population + Illiteracy + Income + Frost, data=states)

qqPlot(fit, labels=row.names(states), id.method="identify", 
       simulate=TRUE, main="Q-Q Plot")


###See the outlier's residual values. 
states["Nevada",]
fitted(fit)["Nevada"]
residuals(fit)["Nevada"]
rstudent(fit)["Nevada"]

###Distribution of Errors. 
residplot <- function(fit, nbreaks=10){
  z <- rstudent(fit)
  hist(z, breaks=nbreaks, freq=FALSE, 
       xlab="Studentized Residuals",
       main="Distribution of Errors")
}

###Linearity
library(car)
crPlots(fit)

###Homeosdacidity

library(car)
ncvTest(fit)
spreadLevelPlot(fit)

##Global Validation of linear model assumption.
install.packages("gvlma")
library(gvlma)
gvmodel <- gvlma(fit)
summary(gvmodel)

###Multicollinearity....
library(car)
vif(fit)

###Unusual observations. 
###Outliers
library(car)
outlierTest(fit)

dev.off()
##High-leverage Points. 

hat.plot <- function(fit) {
  p <- length(coefficients(fit))
  n <- length(fitted(fit))
  plot(hatvalues(fit), main="Index Plot of Hat Values")
  abline(h=c(2,3)*p/n, col="red", lty=2)
  identify(1:n, hatvalues(fit), names(hatvalues(fit)))
}
hat.plot(fit)

dev.off()
###Influential observations
cutoff <- 4/(nrow(states)-length(fit$coefficients)-2)
plot(fit, which=4, cook.levels=cutoff)
abline(h=cutoff, lty=2, col="red")

###Influential observations
library(car)
avPlots(fit, ask=FALSE, id.method="identify")

###Influential observations
library(car)
influencePlot(fit, id.method="identify", main="Influence Plot",
              sub="Circle size is proportional to Cook's Distance")


library(car)
boxTidwell(Murder~Population + Illiteracy, data=states)

###Corrective measures
library(car)
summary(powerTransform(states$Murder))

###Comparing Models. 
fit1 <- lm(Murder ~ Population + Illiteracy + Income + Frost, data=states)
fit2 <- lm(Murder ~ Population + Illiteracy, data=states)
anova(fit1,fit2)
AIC(fit1,fit2)


###Varible Selection
library(MASS)
states <- as.data.frame(state.x77[,c("Murder", "Population", "Illiteracy", "Income", "Frost")])
fit <- lm(Murder~Population+Illiteracy+Income + Frost, data=states)
stepAIC(fit, direction="backward")

install.packages("leaps")
library(leaps)
leaps <- regsubsets(Murder ~ Population + Illiteracy + Income + Frost, data=states, nbest=4)
plot(leaps, scale="adjr2")

library(car)
subsets(leaps, statistic="cp", 
        main="Cp Plot ofr All Subsets Regression")
abline(1,1,lty=2, col="red")

##Cross-Validation
shrinkage <- function(fit,k=10){
  require(bootstrap)  
  
  theta.fit <- function(x,y){lsfit(x,y)}
  theta.predict <- function (fit,x){cbind(1,x)%*%fit$coef}
  x <- fit$model[,2:ncol(fit$model)]
  y <- fit$model[,1]
  
  results <- crossval(x,y,theta.fit, theta.predict, ngroup=k)
  r2 <- cor(y,fit$fitted.values)^2
  r2cv <- cor(y,results$cv.fit)^2
  cat("Original R-Square=", r2, "\n")
  cat(k, "Fold Cross-Validated R-square=", r2cv, "\n")
  cat("Change=", r2-r2cv, "\n")
}

#Relative Importance. 
zstates <- as.data.frame(scale(states))
zfit <- lm(Murder~Population+Income+Illiteracy+Frost, data=zstates)
coef(zfit)


relweights <- function(fit,...){
  R <- cor(fit$model)
  nvar <- ncol(R)
  rxx <- R[2:nvar, 2:nvar]
  rxy <- R[2:nvar, 1]
  svd <- eigen(rxx)
  evec <- svd$vectors
  ev <- svd$values
  delta <- diag(sqrt(ev))
  lambda <- evec %*% delta %*% t(evec)
  lambdasq <- lambda ^2
  beta <- solve(lambda) %*% rxy
  rsquare <- colSums(beta^2)
  rawgt <- lambdasq %*% beta^2
  import <- (rawwgt/rsquare)*100
  import <- as.data.frame(import)
  row.names(import) <- names(fit$model[2:nvar])
  names(import) <- "Weights"
  import <- import[order(import),1,drop=FALSE]
  dotchart(import$Weights, labels=row.names(import),
    xlab="% of R-square", pch=19, main="Relative Importance of Predictor Variables",
    sub=paste("Total R-square=", round(rsquare, digits=3)),
    ...)
return (import)
}    


install.packages("gplots")
install.packages("HH")
install.packages("rrcov")
install.packages("multicomp")
install.packages("MASS")
install.packages("mvoutlier")



##ANOVA 

#One-Way Anova Example.
library(multcomp)
attach(cholesterol)
table(trt)
##Summary statistics. 
aggregate(response, by=list(trt), FUN=mean)
aggregate(response, by=list(trt), FUN=sd)
fit <- aov(response ~ trt)
summary(fit)

library(gplots)
plotmeans(response ~ trt, xlab="Treatment", ylab="Response",
          main="Mean Plot\n with 95% CI")
detach(cholesterol)

###ANOVA F TEST.
##WHICH TREATMENTS DIFFER FROM ONE ANOTHER. 
TukeyHSD(fit)

par(las=2)
par(mar=c(5,8,4,2))
plot(TukeyHSD(fit))

library(multcomp)
par(mar=c(5,4,6,2))
tuk <- glht(fit, linfct=mcp(trt="Tukey"))
plot(cld(tuk, level=.05),col="lightgrey")
            
###Assuming test assumptions. 
library(car)
qqPlot(lm(response ~ trt, data=cholesterol), 
       simulate=TRUE, main="Q-Q Plot", labels=FALSE)

bartlett.test(response ~ trt, data=cholesterol)
##Add more equality tests -> 223
library(car)
outlierTest(fit)

##One-Way Ancova. 
data(litter, package="multcomp")
attach(litter)
table(dose)
aggregate(weight, by=list(dose), FUN=mean)
fit <- aov(weight ~ gesttime + dose)
summary(fit)

##Multiple comparisons employing user-supplied contrasts. 
library(multcomp)
contrast <- rbind("no drug vs Drug" = c(3,-1,-1,-1))
summary(glht(fit, linfct=mcp(dose=contrast)))

##Testing for homogeneity of regression slopes.
library(multcomp)
fit2 <- aov(weight ~ gesttime*dose, data=litter)
summary(fit2)

##Visualizing the results. 
library(HH)
ancova(weight ~ gesttime + dose, data=litter)

##Two-way ANOVA
attach(ToothGrowth)
table(supp, dose)
aggregate(len, by=list(supp,dose), FUN=mean)
aggregate(len, by=list(supp,dose), FUN=sd)
dose <- factor(dose)
fit <- aov(len ~ supp*dose)
summary(fit)
detach(ToothGrowth)

interaction.plot(dose, supp, len, type="b", col=c("red", "blue"), pch=c(16,18),
                 main="Interaction between Dose and Supplement Type")

library(gplots)
plotmeans(len ~ interaction(supp, dose, sep=" "),
          connect=list(c(1,3,5), c(2,4,6)), #Connect with lines same group observations 
          col=c("red", "darkgreen"),
          main="Interaction Plot with 95% CIS",
          xlab="Treatment and Dose Combination")

library(HH)
interaction2wt(len ~ supp*dose)

###Repeated Measures ANOVA
CO2$conc <- factor(CO2$conc)  ##Why do we need to convert it into a factor?.
w1b1 <- subset(CO2, Treatment=="chilled")
fit <- aov(uptake ~ conc*Type + Error(Plant/(conc)),w1b1)
summary(fit)



##One-Way Anova

library(MASS)
attach(UScereal)
shelf <- factor(shelf)
y <- cbind(calories, fat, sugars)
agr <- aggregate(y, by=list(shelf), FUN=mean)

cov(y)
fit <- manova(y ~ shelf)
summary(fit)

summary.aov(fit)

##Assesing test assumptions. 
##Multivariate Normality 
center <- colMeans(y)
n <- nrow(y)
p <- ncol(y)
cov <- cov(y)
d <- mahalanobis(y, center, cov)
coord <- qqplot(qchisq(ppoints(n), df=p),
                d, main="Q-Q Plot Assesing Multivariate Normality",
                ylab="Mahalanobis D2")

abline(a=0, b=1)
identify(coord$x, coord$y, labels=row.names(UScereal))

##Identify outliers. 
library(mvoutlier)
outliers <- aq.plot(y)
outliers

##Robust Manova
##Robust One-Way Anova
library(multcomp)
levels(cholesterol$trt)
fit.aov <- aov(response ~ trt, data=cholesterol)
summary(fit.aov)
##A regression approach to the ANOVA problem. 
fit.lm <- lm(response ~ trt, data=cholesterol)
summary(fit.lm)

##If the predictors encountered are factors instead of nummerical 
##we can replace them with a set of numeric values representing contrasts among the levels. 
contrasts(cholesterol$trt)


##POWER ANALYSIS.
install.packages("pwr")
library(pwr)
pwr.t.test(d=.8, sig.level=.05, power=.9, type="two.sample",
           alternative="two.sided")

##ONE-WAY ANOVA 
pwr.anova.test(k=5, f=.25, sig.level=.05, power=.8)
##Correlations. 
pwr.r.test(r=.25, sig.level=.05, power=.90, alternative="greater")

#Tests on Linear Models. 
pwr.f2.test(u=3, f2=0.0769, sig.level=.05, power=0.90)

#Tests on proportions. 
pwr.2p2n.test(h=ES.h(.65, .6),sig.level=.05, power=.9,
              alternative="greater")
#Chi-Square Tests. 
prob <- matrix(c(.42, .28, .03, .07, .10, .10), byrow=TRUE, nrow=3)
ES.w2(prob)
pwr.chisq.test(w=.1853, df=2, sig.level=.05, power=.9)
##Sample sizes for detecting significant effects in one-way ANOVA. 
library(pwr)
es <- seq(.1,.5,.01)
nes <- length(es)

samsize <- NULL
for (i in 1:nes){
  result <- pwr.anova.test(k=5, f=es[i], sig.level=.05, power=.9)
  samsize[i] <- ceiling(result$n)
}
plot(samsize, es, type="1", lwd=2, col="red",
     ylab="Effect size",
     xlab="Sample Size(per cell)",
     main="One way ANOVA with power=.90 and Alpha=.05")

###Power Analysis Plots. 
library(pwr)
r <- seq(.1,.5,.01)
nr <- length(r)

p <- seq(.4,.9,.1)
np <- length(p)

samsize <- array(numeric(nr*np), dim=c(nr,np))
for (i in 1:np){
  for (j in 1:nr){
    result <- pwr.r.test(n=NULL, r=r[j], 
                         sig.level = 0.05, power = p[i],
                         alternative="two.sided")
    samsize[j,i] <- ceiling(result$n)
  }
}
##Setting up the Graph
xrange <- range(r)
yrange <- round(range(samsize))
colors <- rainbow(length(p))
plot(xrange, yrange, type="n",
     xlab="Correlation Coefficient(r)",
     ylab="Sample Size(n)")

##Adds power curves. 
for (i in 1:np){
  lines(r,samsize[,i], type="1", lwd=2, col=colors[i])
}
##Adds Grid Lines.
abline(v=0, h=seq(0,yrange[2],50), lty=2, col="grey89")
abline(h=0, v=seq(xrange[1], xrange[2], 0.2), lty=2, col="gray89")

##Adds annotations.
title("Sample Size Estimation for Correlation Studies \n Sig=.05(Two-Tailed)")
legend("topright", title="Power", as.character(p),fill=colors)


attach(mtcars)
plot(wt, mpg, 
     main="Basic Scatter Plot of MPG vs. Weight",
     xlab="Car Weight (lbs/100)",
     ylab="Miles per Gallon", pch=19)
abline(lm(mpg ~ wt), col="red", lwd=2, lty=1)
lines(lowess(wt,mpg), col="blue", lwd=2, lty=2)

library(car)
scatterplot(mpg ~ wt | cyl, data=mtcars, lwd=2, span=0.75, 
            main="Scatter Plot of MPG vs Weight by #Cylinders",
            xlab="Weight of Car(lbs/1000)",
            ylab="Miles per Gallon",
            legend.plot=TRUE,
            id.method="identify",
            labels=row.names(mtcars),
            boxplots="xy")

###Scatter-Plot Matrices.
pairs(~mpg+disp+drat+wt, data=mtcars, 
      main="Basic Scatter Plot Matrix")

library(car)
scatterplotMatrix(~mpg+disp+drat+wt, data=mtcars, 
                  spread=FALSE, smoother.args=list(lty=2),
                  main="Scatter Plot Matrix via car Package")

###High-Density Scatter Plots.
set.seed(1234)
n <- 10000
c1 <- matrix(rnorm(n, mean=0, sd=.5), ncol=2)
c2 <- matrix(rnorm(n, mean=3, sd=2), ncol=2)
mydata <- rbind(c1,c2)
mydata <- as.data.frame(mydata)
names(mydata) <- c("x", "y")

with(mydata, 
     plot(x,y,pch=19, main="Scatter Plot with 10.000 Observations."))

with(mydata, 
     smoothScatter(x,y,main="Scatter Plot Colored by Smoothed Densities."))
     
#Other Type of Plots. 
install.packages("hexbin")
library(hexbin)
with(mydata, {
  bin <- hexbin(x,y,xbins=50)
  plot(bin, main="Hexagonal Binning with 10.000 Observations.")
})

##3D ScatterPlot
library(scatterplot3d)
attach(mtcars)
scatterplot3d(wt, disp, mpg, main="Basic 3D Scatter Plot")
scatterplot3d(wt, disp, mpg,
              pch=16,
              highlight.3d=TRUE,
              type="h",
              main="3D Scatter Plot with Vertical Lines")


##3D ScatterPlot and a Regression Plane.
library(scatterplot3d)
attach(mtcars)
s3d <- scatterplot3d(wt, disp, mpg,
              pch=16,
              highlight.3d=TRUE,
              type="h",
              main="3D Scatter Plot with Vertical Lines and Regression Plane")
fit <- lm(mpg ~ wt + disp)
s3d$plane3d(fit)

##Spinning 3D Scatter Plots. 
library(rgl)
attach(mtcars)
plot3d(wt, disp, mpg, col="red", size=5)

##Buble Plots. 
symbols(x,y, circle=sqrt(z/pi))
attach(mtcars)
symbols(wt, mpg, circle=r, inches=0.30,
        fg="white", bg="lightblue",
        main="Buble Plot with point size proportional to displacement",
        ylab="Miles Per Gallon",
        xlab="Weight of Car(lbs/1000)")
text(wt, mpg, rownames(mtcars), cex=0.6)
detach(mtcars)

##Line Charts. 
opar <- par(no.readonly=TRUE)
par(mfrow=c(1,2))
t1 <- subset(Orange, Tree==1)
plot(t1$age, t1$circumference,
     xlab="Age(days)",
     ylab="Circumference(mm)",
     main="Orange Tree 1 Growth")
plot(t1$age, t1$circumference, 
     xlab="Age(days)",
     ylab="Circumference(mm)",
     main="Orange Tree 1 Growth",
     type="b")

##More Complex Line Chart. 
Orange$Tree <- as.numeric(Orange$Tree) ##Converts a factor to numeric for convenience. 
ntrees <- max(Orange$Tree)
##Setting up the Plot.
xrange <- range(Orange$age)
yrange <- range(Orange$circumference)
plot(xrange, yrange,
     type="n",
     xlab="Age (days)",
     ylab="Circumference (mm)"
)
colors <- rainbow(ntrees)
linetype <- c(1:ntrees)
plotchar <- seq(18, 18+ntrees, 1)

##Adds Lines
for (i in 1:ntrees) {
  tree <- subset(Orange, Tree==i)
  lines(tree$age, tree$circumference,
        type="b",
        lwd=2,
        lty=linetype[i],
        col=colors[i],
        pch=plotchar[i]
  )
}
title("Tree Growth", "example of line plot")
legend(xrange[1], yrange[2],
       1:ntrees,
       cex=0.8,
       col=colors,
       pch=plotchar,
       lty=linetype,
       title="Tree"
)

###Correlograms
# Correlograms
options(digits=2)
cor(mtcars)

library(corrgram)
corrgram(mtcars, order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="Corrgram of mtcars intercorrelations")

corrgram(mtcars, order=TRUE, lower.panel=panel.ellipse,
         upper.panel=panel.pts, text.panel=panel.txt,
         diag.panel=panel.minmax,
         main="Corrgram of mtcars data using scatter plots
         and ellipses")

cols <- colorRampPalette(c("darkgoldenrod4", "burlywood1",
                           "darkkhaki", "darkgreen"))
corrgram(mtcars, order=TRUE, col.regions=cols,
         lower.panel=panel.shade,
         upper.panel=panel.conf, text.panel=panel.txt,
         main="A Corrgram (or Horse) of a Different Color")


# Mosaic Plots
ftable(Titanic)
library(vcd)
mosaic(Titanic, shade=TRUE, legend=TRUE)

library(vcd)
mosaic(~Class+Sex+Age+Survived, data=Titanic, shade=TRUE, legend=TRUE)


# type= options in the plot() and lines() functions
x <- c(1:5)
y <- c(1:5)
par(mfrow=c(2,4))
types <- c("p", "l", "o", "b", "c", "s", "S", "h")
for (i in types){
  plottitle <- paste("type=", i)
  plot(x,y,type=i, col="red", lwd=2, cex=1, main=plottitle)
}


###Resampling statistics and bootstrapping. 
install.packages(c("coin"))

library(coin)
score <- c(40,57,45, 55, 58, 57, 64, 55, 62, 65)
treatment <- factor(c(rep("A",5), rep("B", 5)))
mydata <- data.frame(treatment, score)
t.test(score~treatment, data=mydata, var.equal=TRUE)

library(MASS)
UScrime <- transform(UScrime, So= factor(So))
wilcox_test(Prob ~ So, data = UScrime, distribution="exact")

library(multcomp)
set.seed(1234)
oneway_test(response~trt, data=cholesterol, distribution=approximate(B=9999))

###Independence in contengency tables. 
library(coin)
library(vcd)
Arthritis <- transform(Arthritis, 
                       Improved=as.factor(as.numeric(Improved)))
set.seed(1234)
chisq_test(Treatment ~ Improved,
           data=Arthritis, 
           distribution=approximate(B=9999))

states <- as.data.frame(state.x77)
set.seed(1234)
spearman_test(Illiteracy ~ Murder, data=states,
              distribution=approximate(B=9999))

###Dependant two-sample and k-sample tests. 
library(coin)
library(MASS)
wilcoxsign_test(U1~U2, data=UScrime, distribution="exact")

###Simple and Polynomial Regression.
library(lmPerm)
set.seed(1234)
fit <- lmp(weight ~ height, data=women, perm="Prob")
summary(fit)

##Permutation tests for Polynomial Regression. 
install.packages("lmPerm")
library(lmPerm)
set.seed(1234)
fit <- lmp(weight~height + I(height^2), data=women, perm="Prob")
summary(fit)

##Multiple Regression.
library(lmPerm)
set.seed(1234)
states <- as.data.frame(state.x77)
fit <- lmp(Murder ~ Population + Illiteracy + Income + Frost, 
           data=states, perm="Prob")

##Permutation test for one-way ANOVA
library(lmPerm)
library(multcomp)
set.seed(1234)
fit <- aovp(response~trt, data=cholesterol, perm="Prob")

##Permutation test for one-way ANCOVA. 
library(lmPerm)
set.seed(1234)
fit <- aovp(weight ~ gesttime + dose, data=litter, perm="Prob")

##Two-way ANOVA



