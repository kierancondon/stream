"""
title: Option Valuation & Hedging
date: 2015-01-01
tags: python, finance
"""

import streamlit as st

def display():

    st.write("""
    # Option Valuation and Hedging in Python

    ## Introduction

    Let's assume you own an option, how much is it worth? This notebook shows a number of ways to model and price that option using python. 
    It makes a comparison between a simulation approach and a closed form approach. The unique thing is that both methods of valuing the option can be 
    understood as two different real world hedging strategies for the option.

    For example, the simulation approach is akin to hedging your option using the underlying product (i.e. stock or forward contracts). 
    The closed form approach is akin to hedging your option with another option, priced using Black and Scholes theorem.

    The expected payoff of both hedging strategies will be the same and will give the correct value of the option.

    Most importantly, I want to make it easy to see and therefore understand:

    - Lognormal prices
    - The Black Scholes equation
    - The power and simplicity of python and pandas
    
    And with respect to Python modelling, the following is done:

    - Simulation of lognormal forward price scenarios
    - Calculation of an option premium using Black Scholes closed form solution
    - Simulation of option payoffs using a montecarlo approach
    - Simulation of the in-the-money volume of an option and its evolution over time

    ## Creating Price Scenarios

    ### Importing what's needed from Python and setting assumptions
    """)

    with st.echo():
        from scipy.special import ndtri
        from math import exp, log, pi
        import matplotlib.pyplot as plt
        from random import random, seed
        import pandas as pd
        import inspect
        import numpy as np
        plt.style.use('ggplot')

        # Market price settings
        S0 = 20.0  # Current underlying price
        sd_p = 0.05  # Standard deviation per *period*
        mu_p = 0.0  # Percent drift per *period*

        # Option settings
        T = 36  # Number of *periods* until delivery
        K = 20.0  # Strike price
        notional = 100.0  # Volume of the underlying
        r = 0.000000000001  # risk free rate of return per period
        strat_type = 'instant'  # 'instant' 'spot' or 'progressive'

        # Variables dealing with steps
        dt = 1  # Size of a time step relative to one period
        steps = int(T / dt)  # Total number of time steps in all periods

        # Simulation settings
        scens = 10000  # The number of scenarios to run

    st.write("""
        In this simulation, TIME is defined in periods, say months or years, as well as steps within those periods. 
        There could be 10 or 2 or 1 step for every time period. This is done to stay general to some well known formulas. 
        For the most part the total number of steps is what matters. Actions, such as sell a forward or sell an option can be taken in each step. 
        Prices change in every step. However, the standard deviation and drift of price are defined per period, 
        and so you will see them converted to be relevant to steps in the following section.
        
        #### Prices are modelled using the following formula:
        
        $$ 
        S_{t+Δt} = S_t.μ.Δt + S_t.ϑ.ε.\sqrt{Δt}
        $$

        """)

    st.write("""
    This says that the price at the next moment in time depends on the current price affected by an unpredictable, random shock as well as a 
    predictable change known as a drift. It's easiest to demonstrate the formula through code:
    """)

    with st.echo():
        def price_evo(S0, sd_p, mu_p, steps):
            prices = [S0]
            for i in range(steps):
                drift = prices[i] * mu_p * dt
                uncertainty = prices[i] * ndtri(random()) * sd_p * dt ** 0.5
                prices.append(prices[i] + drift + uncertainty)
            return prices

    st.write("""
    This function creates a full forward evolution of lognormal prices for the amount of time steps specified. I set the drift to 0 in the settings above. 
    Uncertainty is the only part affecting price, so I explain that here:

    We look firstly at the part of the formula given by epsilon, 'e'. This states by how many standard deviations we will move the price by. 
    In Python it is given by ndtri(random()). It is a random draw from a standard normal distribution's cumulative distribution function. 
    It is equivalent to this in Excel: "=NORMINV(rand(), 0, 1)". On average this term will be 0 and it may go as low as 
    -infinity and +infinity but in practice it will range between -3 and 3.

    Epsilon is multiplied by the standard deviation of the stock given by 'sigma'. It is correct to use the standard deviation of one time step. 
    In this formula I calculate that by taking the standard deviation of a full period of time and adapt it using the square root of time rule 
    to represent one time step. In this example a step equals one period as dt is set to 1.

    So far these three terms combine to give the change in price as a proportion of price. Multiply by price to get the actual change. 
    Add this to the current price to have the new price and this is a random walk
    """)


    def strategy_maker(steps):
        strategies = {'instant': [1] * (steps + 1),
                      'delivery': [0] * (steps) + [1],
                      'progressive': list(np.linspace(0, 1, (steps + 1)))}
        return strategies

    def dyn_trade_evo(K, notional, prices, strategy):
        int_vol = []
        for i in range(len(prices)):
            int_vol.append(notional if K <= prices[i] else 0)
        hedge_tot = [a * b for a, b in zip(int_vol, strategy)]
        return actions(hedge_tot)

    def dyn_payoff(prices, dyn_trades):
        hedge_payoff = sum([a * b for a, b in zip(prices, dyn_trades)])
        asset_exercise = -K * notional if K < prices[-1] else 0
        return hedge_payoff + asset_exercise

    def black_scholes_call(S, K, t, r, sd):
        if t == 0:
            return max(S - K, 0)
        else:
            d1 = (log(S / K) + t * (r + (sd * sd) / 2.0)) / (sd * t ** 0.5)
            d2 = d1 - sd * t ** 0.5
            Nd1 = norm_cdf(d1)
            Nd2 = norm_cdf(d2)
            return S * Nd1 - K * exp(-r * t) * Nd2

    def opt_payoff(prices, K, r, sd, option_trades):
        opt_pay = 0
        t = len(prices)
        for i in range(t):
            opt_pay += black_scholes_call(prices[i], K, t - i - 1, r, sd) * option_trades[i]
        return opt_pay

    def actions(p):
        ap = []
        for i in range(len(p)):
            ap.append(p[i] if i == 0 else p[i] - p[i - 1])
        return ap

    def norm_cdf(x):
        k = 1.0 / (1.0 + 0.2316419 * x)
        k_sum = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + 1.330274429 * k))))
        if x >= 0.0:
            return (1.0 - (1.0 / ((2 * pi) ** 0.5)) * exp(-0.5 * x * x) * k_sum)
        else:
            return 1.0 - norm_cdf(-x)

    def simulate(S0, sd_p, mu_p, steps, K, notional, r, strat_type, dt):
        strategy = strategy_maker(steps)[strat_type]
        option_trades = actions([notional * e for e in strategy])
        sd = sd_p * dt ** 0.5  # Standard deviation of a time step

        dyn_payoffs = []
        opt_payoffs = []

        for scen in range(scens):
            if scen % 1000 == 0: print
            scen,
            prices = price_evo(S0, sd_p, mu_p, steps)
            opt_payoffs.append(opt_payoff(prices, K, r, sd, option_trades))
            dyn_trades = dyn_trade_evo(K, notional, prices, strategy)
            dyn_payoffs.append(dyn_payoff(prices, dyn_trades))
        print
        'Done!'
        return pd.DataFrame({'Stock': np.array(dyn_payoffs),
                             'Option': np.array(opt_payoffs)})

    st.code(inspect.getsource(simulate))
    if st.button('Run the simulation'):
        payoff_i = simulate(S0, sd_p, mu_p, steps, K, notional, r, 'instant', dt)
        payoff_p = simulate(S0, sd_p, mu_p, steps, K, notional, r, 'progressive', dt)
        payoff_d = simulate(S0, sd_p, mu_p, steps, K, notional, r, 'delivery', dt)
        full = pd.concat([payoff_i, payoff_p, payoff_d], axis=1, keys=['i', 'p', 'd'])
        desc = np.round(full.describe())
        st.write(desc)