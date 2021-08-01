"""
title: How logistic regression works
date: 2015-01-01
tags: python, econometrics
"""

import streamlit as st

def display():

    st.write("""
    # How Logistic Regression Works
    
    ## Introduction
    
    What is Logistic Regression (LR)? First consider an example from linear regression: predict the life expectancy of a person based on their current age, where they live, 
    their habits and so on. In this case the explained variable, life expectency is practically continuous. It takes on a large set of possible value.

    In LR, the possible values are a binary set. A question relevant for LR would be: "Will a person still be alive in 10 years or not?". The outcome is yes or no, 1 or 0.
    The explanatory variables could be the same as above.

    The hypothesis of linear regression, i.e. $ŷ = θ_0 + θ_1.X_1 + θ_2.X_2$, or **θTX** in matrix notation, is not suitable because it could give values greater than 1 as it
    depicts a continuous line. Interestingly, the same structure can be used but it must be transformed into a function that lies between 0 and 1. 
    
    This is done using a Sigmoid function **1 / ( 1 + e^Z)**, which equals 0.5 when Z = 0. 
    
    The hypothesis of logistic regression equals **1 / (1 + e^Z)** where **z = θTX**. 
    
    So, for a single explanatory x variable, the $hypothesis = 1 / (1 + e^{θ_0 + θ_1.X_1})$, and we say that ŷ = 1 when hypothesis is greater 
    than 0.5 and ŷ = 0 when hypothesis is less than 0.5.
    
    This notebook aims to show the meaning behind this.
    
    ### Firing up
    
    """)

    with st.echo():
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        import math


        def sigmoid(z):
            return 1 / (1 + math.exp(-z))

        curve = np.linspace(-5, 5, 11) # integers between -5 and 5
        test = pd.Series(curve).apply(sigmoid)
        test.index = curve

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test))
        fig.update_layout(xaxis_title='Integer input', yaxis_title='Sigmoid value')

        st.plotly_chart(fig)

    st.write("""
    Yes, it's really as easy as this to turn an array into a sigmoid. It looks like an S, and it equals 0.5 when z equals 0.
    
    This is useful. Our prediction for y can be represented by the sigmoid as it is always conveniently between 0 and 1. 
    As the prediction for y must be either 0 or 1 we will say y is 0 if the sigmoid is less than 0.5 and we will say it is 1 if the sigmoid is greater than 0.5.

    The purpose of a logistic regression is to define theta so that Z equals 0, i.e. the sigmoid equals 0.5, where we believe y goes from 'probably 0' to 'probably 1'.

    (I won't discuss how we solve for that, which can be done in different ways. That involves minimising a cost function which 
    penalises situations where we predict y poorly. In this notebook we solve the regression visually.)

    Let's see this.

    ## Will You Still Be Alive in 10 Years?
    We predict this question using one explanatory variable: *How many red traffic lights do you run per year?*

    Create artificial dataset where x is number of lights jumped per year over last 10 years and y is whether the person is still alive following this period or not,
    Making a prediction for y denoted by y^
    """)

    with st.echo():
        data = pd.DataFrame({'x': [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             'y': [1.0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.x, y=data.y, mode='markers'))
        fig.update_layout(xaxis_title='Lights jumped per year', yaxis_title='Still alive after 10 years?')
        st.plotly_chart(fig)

    st.write("""
    The goal is to predict when y equals 0 (dead) or 1 (alive) based on information given by x. In this data it seems that x above 5 is more or less where y goes from 
    probably 1 (alive) to probably 0 (dead). Let's assume for now that 5 is the correct solution to this problem.

    How would we get to this result using a sigmoid? We need to define an equation for z, a function of x, so that: 
    - z = 0 where X = 5.  
    - z is positive when X is less than 5
    - z is negative when X is greater than 5
    
    Why is that important? 
    It is so that the #sigmoid applied on z# will be greater than 0.5 when x is less than 5 and it will be less than 0.5 when it x is greater than 5. 
    Therefore predicted y equals 1 when x is less than 5 and 0 when x is greater than 5.

    A very simple function for z(x) leads to this result: z = 5 − X1 is a good way to define z so that it is 0 in the correct place, and positive / negative 
    in the correct way too. In other terms, a good hypothesis is that `θ0 = 5` and `θ1 = −1`.

    #### Why did we go to the trouble of using a sigmoid?
    The sigmoid function is a continuous and differentiable function between 0 and 1. The differentiable nature of the sigmoid is what allows 
    the definition of a cost function that is 'convex' and can be minimised non-iteratively (and iteratively) with a unique result. 
    It allows a 'closed form', 'best fit' solution.

    I have not demonstrated that optimisation process here but I hope I have made it clear what the purpose of that process is.

    #### Time to chart this result
    """)

    with st.echo():
        data['z'] = 5 - data.x
        data['sigmoid'] = data['z'].apply(sigmoid)
        data['yhat'] = 0
        data.loc[data['sigmoid'] >= 0.5, 'yhat'] = 1
        st.dataframe(data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.x, y=data.sigmoid, mode='lines', name="sigmoid"))
        fig.add_trace(go.Scatter(x=data.x, y=data.yhat, mode='markers', name="predicted y"))
        fig.add_trace(go.Scatter(x=data.x, y=data.y, mode='markers', name="actual y"))
        fig.update_layout(xaxis_title='Red lights run per year', yaxis_title='Will you still be alive in 10 years?')
        st.plotly_chart(fig)