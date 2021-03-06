# Week 1
## What is the investment management process?
    1. Investment objects 
        1.1. Return
        1.1. Liability (cash flow)
        1.2. Expect of the risk

        What is benchment? 
    2. Investment policy
        Document describes all investment decisions
        Asset allocation decisions

    3. Measurement and evaluation of investment perfromance
        Monitoring portfolio, re-balancing


## Back to basics (Part 1.1): What is time value of money?

    How can compare Revenue in 1997 and 2009?
    I asked to lend me 100$ and I will pay 100$ in year

    Dollar today is worth more than a dollar tomorrow

    Why does money have time value?
        1. Postponed consumtion. You can use for your own consumtion. 
        2. Expected inflation
        3. Risk

    All above - opportunity cost - that's why it's positive rate (i.e. interest)


### How to find future value?
    Funds invested * opportunity cost
    After year = 100+100*0.05= 105;
    After 3 years = 100*(1.05)^3 - compounding

    Future value = interest rate;

### Finding the present value of a cash flow
    Presert value = 3 years value / (1.05)^3;

    Discount rate = Interest rate =  Opportunity cost of capital
    
## What are annuities?
    Fixed amount of cash flow within certain period of time

 Suppose you would like to have $50,000 in two years to start your new business idea.

### Stream of cash flow

![](img/SNAG_Program-0000.png)

### Annuity Compound Factor (Future value)
What if all cash flow is the same amount every year?

![](img/SNAG_Program-0002.png)
![](img/SNAG_Program-0003.png)

### Annuity Compound Factor (Present Value) - Retirement problem
What even annual savings or payments would you have to make to get to your goal if you can earn 6% per year? assuming that the interest rate is constant at 6%, expect to earn over the next 35 years

Future Value = 1 000 000 
Rate = 6%
Years = 35

![](img/SNAG_Program-0004.png)

### Finding the present value of annuities
Where we know the future value we know the constant interest rate and we know the number of payments and we try to find what the equal payments should be.

Knowns;
    1. Rate: 5%
    2. Years 2 years
    3. Expected annuity: 2000$

Question:
    1. What will be present value? 

![](img/SNAG_Program-0005.png)


### Annuities Example: Loan Problem
suppose the purchase price of the car that you would like to buy today is $37,150

Present value = 37150

you want to take out a loan, we're going to do a 100% financing with a maturity of 60 months, right, so there's going to be 60 payments.
first loan payment will come in one month's time

Interest Rate = 4%/y or 0.33%/mnth
Length = 5 years / 60 months

![](img/SNAG_Program-0006.png)

## Computing the effective annual interest rate

 credit card payments or car loan payments or mortgage payments are compounded not yearly, but typically monthly.

 what if the interest rate is being compounded every second instantaneously? Or continuously

 Interest rate = APR ( Annual Percentage Rate)

The problem: 
    APR is not what you really pay, Effective Rate is what you really pay

### What is Effective Rate?

![](img/SNAG_Program-0007.png)

## Computing an effective rate over any period
Say, APR 4%
Month rate = 4% / 12 = 0.0033%
Effective rate = (1.0033)^12 -1 = 4.032%

2month effective rate = (1.0033)^2 -1 = 0.6611%
...

## Computing continuously compounded rates

![](img/SNAG_Program-0008.png)

Question:
    What are your monthly payments?


## Back to basics (Part 2.1): What are annuities?

    Stream of cash flows, 
        how to find future value in 5 values having interest rate 5%? 

     FV = future cash value for every year and period and sum them up

     What if cash flow is equal every year than we can define ACF (r, n)

*Annuity Compound Factor (ACF)* 
 V(n) = C * ACF (R, n)

![](img/screenshot2018-11-18at22.22.33.png)



## Back to basics (Part 2.2): Annuities example: Retirement problem

   Target = 1 000 000$
   Years = 35
   Interest rate = 6%

*How much do we need to save every year?*

![](img/screenshot2018-11-18at22.38.37.png)

*Annuity Discount Factor (ADF)*  

What is Present value of cash flow?
 
![](img/screenshot2018-11-18at22.34.51.png)


## Back to basics (Part 2.4): Annuities Example: Loan Problem

Example:
    1. Car price today (present value): $37150
    2. Take loan (100%)
    3. Interest rate = 4% year or 4%/12 = 0.33% monthly
    4. Loan length = 60 monthers
    
    What car payments will be? 

    V(0) = 37150 = C X ADF(-.33%, 60) = 683.5

![](img/screenshot2018-11-18at22.48.29.png)


##  Computing the effective annual interest rate

- compounded monthly, effective rate over the year

![](img/SNAG_Program-0009.png)


##Back to basics (Part 4.1): Valuing perpetuities and growing perpetuities

What is perpetuities?
    It's series of payments (cash flows) over idefinate period

What is maturity?
    End date of last cash flow payment 

How to calculate V0? 
    V0 = C / r = 1000/0.01 = 10000

Payment growth - g
g = 5%

V0 = C / ( r-g) = 1000 / 0.1 - 0.05 = 20000

## Back to basics (Part 4.2): Valuing growing annuities

![](img/SNAG_Program-0010.png)


Which one would you prefer if your opportunity cost of capital is 6 percent per year?


-Receiving $150,000 today


-Receiving $100,000 today and a stream of cash flows every month for the next 36 months starting next month with $1250 every month and growing by 0.125% every month


-Receiving $750 every month forever starting today


+Receiving $25000 today and a stream of cash flows every month forever starting with $500 next month growing by 0.125%

Correct 
In order to compare these alternatives, we need to find the present value of each cash flow stream. This is a growing perpetuity that starts immediately.

The present value of the growing perpetuity is given by:

V0 = C/(r−g)

where C = 500 , r = 6%/12 = 0.5% and g = 0.125%

V0 = 500/(0.5%−0.125%) = 133,333.333

Total value = 133,333 + 25,000 = 158,333.333













