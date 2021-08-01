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

    This notebook shows a number of ways to model and price an option using python. It makes a comparison between a simulation approach and a closed form approach. 
    Both methods of valuing the option can be understood as two different real world hedging strategies for the option:

    - the simulation approach is akin to hedging your option using the underlying product (i.e. stock or forward contracts), *but we simulate +1000 outcomes of this*.
    - The closed form approach is akin to hedging your option with another option, priced using Black Scholes theorem.
    
    In both cases, hedging actions can take place at any time in the life of the contract. Assuming no drift in prices, the expected payoff of all hedging strategies 
    will be the same and equal to the value of the current option.

    The aim is to make it easy to see and therefore understand:

    - Simulation of log normal forward price scenarios
    - Calculation of an option premium using Black Scholes closed form solution
    - Simulation of option payoffs using a monte carlo approach
    - Simulation of the in-the-money volume of an option and its evolution over time
    - The power and simplicity of python and pandas


    ### Firing up
    """)

    with st.echo():
        from scipy.special import ndtri
        from math import exp, log, pi
        import matplotlib.pyplot as plt
        from random import random, seed
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import inspect
        plt.style.use('ggplot')


    st.write('#### Market price settings')
    S0 = st.number_input(label="S0: Current underlying price", value=20.0)
    sd_p = st.number_input(label="sd_p: Standard deviation per period", value=0.05)
    mu_p = st.number_input(label="mu_p: Percent drift per period", value=0.0)

    st.write('#### Time variables')
    T = st.number_input(label="T: Number of *periods* until delivery", value=36)
    dt = st.number_input(label="dt: Size of a time step relative to one period", value=1.0, min_value = 0.01, max_value=1.0)
    steps = int(T / dt)
    st.write("steps: Total number of time steps over all periods = " + str(steps))

    st.write('#### Option variables')
    K = st.number_input(label="K: Strike price", value=20.0)  # Strike price
    notional = st.number_input(label="notional: Volume of the underlying", value=100.0)
    r = 0.000000001  # risk free rate of return per period
    scens = st.number_input(label="scens: The number of scenarios to run", value=1000, min_value=2, max_value=10000)

    st.write("""
        ## Creating Price Scenarios
        In this simulation, time is defined in periods, say months or years, as well as steps within those periods. 
        There can be any number of steps for every time period. This is done to stay general to some well known formulas. 
        For the most part the total number of steps is what matters. Prices change in every step. Hedging Actions, such as sell a forward or sell an option can be taken in each step. 
        However, the standard deviation and drift of price are defined per period, and so you will see them converted to be relevant to steps later.
        
        ### Prices are modelled using random walk with drift
        
        $$ 
        S_{t+Δt} = S_t.μ.Δt + S_t.ϑ.ε.\sqrt{Δt}
        $$

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
    Uncertainty is the only part affecting price.

    We look firstly at the part of the formula given by epsilon, 'e'. This states by how many standard deviations we will move the price by. 
    In Python it is given by ndtri(random()). It is a random draw from a standard normal distribution's cumulative distribution function. 
    It is equivalent to this in Excel: "=NORMINV(rand(), 0, 1)". On average this term will be 0 and it may go as low as 
    -infinity and +infinity but in practice it will range between -3 and 3.

    Epsilon is multiplied by the standard deviation of the stock given by 'sigma'. It is correct to use the standard deviation of one time step. 
    In this formula I calculate that by taking the standard deviation of a full period of time and adapt it using the square root of time rule 
    to represent one time step. In this example a step equals one period as dt is set to 1.

    So far these three terms combine to give the change in price as a proportion of price. Multiply by price to get the actual change. 
    Add this to the current price to have the new price and this is a random walk
    
    ### The following function simulates price evolutions and charts them
    """)

    with st.echo():
        def chart(scens):
            prices = [price_evo(S0, sd_p, mu_p, steps) for scen in range(scens)]
            prices = pd.DataFrame(prices).transpose()
            prices.index.name = 'Time'

            fig = go.Figure()
            for col in prices.columns:
                fig.add_trace(go.Scatter(x=prices.index, y=prices[col]))

            fig.update_layout(
                xaxis_title='time', yaxis_title='price',
                title={'text': 'Lognormal price scenarios', 'y': 0.85, 'x': 0.12, 'xanchor': 'left', 'yanchor': 'top'})
            return fig

    st.plotly_chart(chart(st.number_input(label="Price evolutions to chart", value=20, min_value=1, max_value=50)))

    st.write("""
        ## Simulating a Call Option
        The goal of this section is to determine the value of a Call option. This can be done by assuming we already have this option and seeing how much money we can expect to 
        make with it by executing various hedging strategies. A buyer of the option should reward us for this expected value, at least.
        
        The first strategy is simple. We have an option, let's sell the same option instantly and earn the value as given by the Black Scholes equation. This is a perfect hedge 
        because we will not be exposed to future price changes at all.
        
        We could also sell the stock instantly (a forward contract, as we don't yet hold it) instead of selling the option, as long as the call is currently in-the-money. After that, 
        if the option goes out of the money we should buy the stock back again, as we can no longer expect to receive it via our option. If the price goes up and we again expect to
        receive it via our Call we can again sell a forward on the stock. This back and forth could happen multiple times and will generate a positive cashflow. It will 
        also (imperfectly) hedge the outcome of the option being exercised or not.
        
        Instead of selling either the option or the stock **instantly** we could do it **progressively**, selling a little bit over a long period of time. In the case of the stock we would 
        still need to do buy backs when the option comes OTM. This progressive approach is also a hedge although it is less effective (more risky) as we do not know where prices will evolve.
        
        The last strategy is so extreme it can't really be called hedging. We sell the option on the stock, or the stock itself, the very moment the option is exercised. Let's call 
        this **at delivery**. If the option is ITM, we buy the stock at price K and sell it at price S. If it is OTM we do nothing. If we sell an option at delivery, it is exactly the same outcome
        as for an outright stock because the premium we will get will be exactly equal to S-K if ITM or 0 otherwise.
        
        To put a value on all of these methods of hedging we need to simulate them many times and see the average value, or the expected value.
        
        To summarise: 
        1. we could sell stock or an option, and,
        2. we can do either of those two things instantly, progressively or in delivery (depending on if our existing option is exercised / expected to be exercised or not).
        
        We simulate all of these many times and see the expected payoff of each. That is 6 strategies in total.
        
        Firstly, the timing of the strategy is defined, instant progressive or in delivery.
        """)

    with st.echo():
        def strategy_maker(steps):
            strategies = {'instant': [1] * (steps + 1),
                          'delivery': [0] * (steps) + [1],
                          'progressive': list(np.linspace(0, 1, (steps + 1)))}
            return strategies

    st.write("""
    ### Simulating the "sell stock" strategies
    The next definitions calculate when the option is in the money or not and will sell the correct amount as stock at the right time 
    at the price given for the correct date (step) in the price scenarios we generated earlier. Currently we follow the 'instant' strategy. 
    The payoff sums the value of all selling and buying deals done.
    """)

    with st.echo():
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

    st.write("""
    ### Simulating the "sell options" strategies
    The next definitions simulate the sale of options at the three given speeds using the price scenarios per step that we generate as above, and using the closed form Black Scholes equation. 
    
    The last function is the norm_cdf formula used earlier. I have coded this manually because it is far faster than the built in scipy method, however it is an 
    approximation (sufficiently accurate).
    """)

    with st.echo():
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

    st.write("""
    #### Finally, putting it all together!
    
    The following function executes both the stock method and the option method, given one strategy type ('instant', 'progressive' or 'delivery'). 
    It returns all 10 000 payoffs (for example) for both so we can see the full distribution.
    """)

    with st.echo():
        def simulate(S0, sd_p, mu_p, steps, K, notional, r, strat_type, dt):
            strategy = strategy_maker(steps)[strat_type]
            option_trades = actions([notional * e for e in strategy])
            sd = sd_p * dt ** 0.5  # Standard deviation of a time step

            dyn_payoffs = []
            opt_payoffs = []

            for scen in range(scens):
                if scen % 1000 == 0:
                    print(scen)
                prices = price_evo(S0, sd_p, mu_p, steps)
                opt_payoffs.append(opt_payoff(prices, K, r, sd, option_trades))
                dyn_trades = dyn_trade_evo(K, notional, prices, strategy)
                dyn_payoffs.append(dyn_payoff(prices, dyn_trades))
            print('Done!')
            return pd.DataFrame({'Stock': np.array(dyn_payoffs),
                                 'Option': np.array(opt_payoffs)})

        # st.code(inspect.getsource(simulate))    this code is able to display the function code on screen. Alternative to st.echo

        if st.button('Run the simulation'):
            payoff_i = simulate(S0, sd_p, mu_p, steps, K, notional, r, 'instant', dt)
            payoff_p = simulate(S0, sd_p, mu_p, steps, K, notional, r, 'progressive', dt)
            payoff_d = simulate(S0, sd_p, mu_p, steps, K, notional, r, 'delivery', dt)
            full = pd.concat([payoff_i, payoff_p, payoff_d], axis=1, keys=['i', 'p', 'd'])
            desc = np.round(full.describe())
            st.write(desc)

    st.write("""
    ## Conclusion
    We see that the mean of all methods gives approximately the same value. The differences come about from not having infinite scenarios, 
    but this isn't needed to see the point. The conclusions we can draw are:
    
    It doesn't matter if you sell stock or options, you still expect to earn the same amount+
    It doesn't matter when you execute your hedging strategy, or even if you do nothing, you still expect to earn the same amount++
    However, both of these things do matter for the risk you take, as shown by the standard deviation.
    
    +Only valid with lognormal price scenarios 
    ++Only valid with lognormal price scenarios with no drift (brownian motion)
    """)