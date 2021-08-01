"""
title: How gradient descent works
date: 2015-01-01
tags: python, econometrics
"""

import streamlit as st

def display():
    st.write("""
    # How does Gradient Descent works
    ## Introduction
    
    This notebook presents an example of gradient descent to perform linear regression. Put simply gradient descent is a smart way of finding a 
    linear relationship between variables through iteration. It is smart because every iteration will be better than the last (if done correctly). 
    The computer will place an intitial line and improve it until it represents a best fit. It is not a closed form solution, like Ordinary Least Squares, 
    which finds the unique result instantly through linear algebra.

    I was inspired by the first lesson of the Coursera MOOC, 'Machine Learning' and have used the dataset from that course. 
    The answers to tutorial questions should not be posted online - in this case the content of this notebook is sufficiently different and does not give away answers. 
    If you understand this notebook, you understand what was taught in the lesson. I have used Python instead of Octave, the language of the course.
    """)

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    with st.echo():
        data = pd.read_csv('https://raw.githubusercontent.com/kierancondon/Sandbox/master/datasets/ex1data2.txt', header=None)
        X = np.array(data.iloc[:, 0:2])
        y = np.array(data.iloc[:, 2:3])

    st.write(
        "Load the data we need using Pandas. This is a dataset describing house characteristics and house price. X represents "
        "the multiple explanatory variables (square feet, number of bedrooms), y represents the variable we want to predict (house price).")

    with st.echo():
        plt.scatter(X[:, 0], y)
        plt.xlabel('Square Feet')
        plt.ylabel('House Price')
        st.pyplot(plt)

        plt.figure()
        plt.scatter(X[:, 1], y)
        plt.xlabel('Number of Bedrooms')
        plt.ylabel('House Price')
        st.pyplot(plt)

    st.write("""
    Plots indicate that house price is positively correlated with the size of the house and the number of bedrooms.

    #### Define functions
    """)

    with st.echo():
        def onesX(X):  # Append ones to the explanatory variables
            return np.concatenate((np.ones((len(X), 1)), X), axis=1)

    st.write("""Here a column of 1s is added to the explanatory variables. This will act as an intercept term. 
    The intercept attempts to account for the price of a home assuming it has no bedrooms and has no size. Yes, this is a theoretical notion, but it allows to achieve a better fit.
    """)

    with st.echo():
        def normaliseFeatures(X):
            mu = np.mean(X, axis=0)
            sigma = np.std(X, axis=0)
            X = (X - mu) / sigma
            X = onesX(X)
            return X, mu, sigma

        def gradientDescent(X, y, alpha, iters):
            m = len(X)
            theta = np.zeros((3, 1))
            J_history = np.zeros((iters, 1))
            X, mu, sigma = normaliseFeatures(X)

            for i in range(iters):
                deriv = X.T.dot((X.dot(theta) - y))
                theta = theta - alpha / m * deriv
                J_history[i] = computeCost(X, y, theta, m)
            return theta, J_history, mu, sigma,

        def computeCost(X, y, theta, m):
            return 1.0 / (2.0 * m) * (X.dot(theta) - y).T.dot((X.dot(theta) - y))

        def computeYGD(X, theta, mu, sigma):
            if isinstance(X, list):
                X = np.array([[x] for x in X]).T
            X = (X - mu) / sigma
            return onesX(X).dot(theta)

    st.write("""All of the code needed for gradient descent is written above! It is a neat procedure because linear algebra is 
    used to perform something complicated easily. This 'vectorised' way of coding is also faster than using loops.
    """)

    with st.echo():
        def ols(X, y):
            X = onesX(X)
            theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
            return theta

        def computeYOLS(X, theta):
            if isinstance(X, list):
                X = np.array([[x] for x in X]).T
            return onesX(X).dot(theta)

    st.write("""And everything needed for OLS is written above. It is even more succinct. Generally OLS should be used as long as there 
    aren't thousands of explanatory variables. In this case there are only two, but gradient descent will be used anyway. Gradient descent is very 
    general in that it allows for different cost functions and so it is more versatile. The cost function of linear regression, as in this example, 
    is the sum of (difference between the predicted y and the actual y for each observation)^2.
    """)

    with st.echo():
        thetaGD, J_history, mu, sigma = gradientDescent(X, y, 0.3, 400)
        thetaOLS = ols(X, y)

    st.write('Gradient Descent: ' + str(computeYGD([1650, 3], thetaGD, mu, sigma)))
    st.write('Ordinary Least Squares: ', str(computeYOLS([1650, 3], thetaOLS)))

    st.write(""" 
    The code above executes both Gradient Descent and OLS. GD needs both a step term, 0.3, and the number of iterations to keep improving the fit of the line, 400. 
    In the end both methods predict the same price for a house of 1650 square feet and 3 bedrooms: 293 081 US dollars. GD worked perfectly.
    
    #### Actual vs predicted
    Plotting again the price vs size relationship we can see the actual data vs the predicted. The predicted is not a perfectly straight line because here we 
    ignore the number of bedrooms. If seen in 3d space it would be a perfect line.
    """)

    with st.echo():
        yhat = computeYGD(X, thetaGD, mu, sigma)
        plt.figure()
        plt.scatter(X[:, 0], y, label='Actual price')
        plt.scatter(X[:, 0], yhat, c='red', label='Predicted price')
        plt.xlabel('Square Feet')
        plt.ylabel('House Price')
        plt.legend()
        st.pyplot(plt)

    st.write("""
    #### Plotting the convergence graph
    How long did it take GD to find the lowest 'cost' i.e. the best fit? It made it in about 8 steps, way quicker than the 400 for which it was run!
    """)

    with st.echo():
        def plotGDCost(J_history, xlim, label):
            plt.figure()
            plt.plot(J_history, label=label)
            plt.title('Cost Function Evolution')
            plt.xlabel('Number of iterations')
            plt.ylabel('Cost: sum of squared error')
            plt.xlim([0, xlim])
            plt.legend()
            return plt

        st.pyplot(plotGDCost(J_history, 50, 0.3))

    st.write("""
    The step size was determined by the number 0.3. It would have taken longer to find a good solution if we had used a smaller step term, such as 0.1. 
    A solution would have been found even more quickly if we had used a step of 1, but it may then have been inaccurate or 
    we **may have never found a solution**. Picking the step term is crucial to GD.
    """)

    with st.echo():
        alphas = [0.01, 0.03, 0.1, 0.3, 1.0]

        for alpha in alphas:
            thetaGD, J_history, mu, sigma = gradientDescent(X, y, alpha, 400)
            plotGDCost(J_history, 50, alpha)
            plt.title('Cost Evolution for Different Alphas')

    st.write("""
    ### Gradient descent cost function in 3D space
    The following plots make it easy to see that the cost function is bowl shaped around the optimal solution. We found the minimum cost at the center of the bowl, 
    indicated by the red cross. OLS is able to work because there is always a unique solution to this type of cost function, which makes linear regression 
    like this popular. It gets trickier when the cost surface has multiple minima.
    """)

    with st.echo():
        def plotThetaContour(X, y, theta, J_history, t1, t2, a):
            m = len(X)
            X = normaliseFeatures(X)[0]

            thetat1_vals = np.linspace(theta[t1] * (1 - a), theta[t1] * (1 + a), 100)
            thetat2_vals = np.linspace(theta[t2] * (1 - a), theta[t2] * (1 + a), 100)
            J_vals = np.zeros((len(thetat1_vals), len(thetat2_vals)))

            for i, v0 in enumerate(thetat1_vals):
                for j, v1 in enumerate(thetat2_vals):
                    wtheta = np.array(theta)
                    wtheta[t1] = thetat1_vals[i]
                    wtheta[t2] = thetat2_vals[j]
                    J_vals[i, j] = computeCost(X, y, wtheta, m)

            R, P = np.meshgrid(thetat1_vals, thetat2_vals)

            fig1 = plt.figure()
            ax = fig1.gca(projection='3d')
            ax.plot_surface(R, P, J_vals.T)
            plt.plot(theta[t1], theta[t2], J_history[-1], 'rx', markersize=10)
            plt.title('3D Cost Surface')
            plt.xlabel('Variable ' + str(t1))
            plt.ylabel('Variable ' + str(t2))
            #plt.show()

            fig2 = plt.figure()
            plt.contour(R, P, J_vals.T)
            plt.plot(theta[t1], theta[t2], 'rx', markersize=10)
            plt.title('Cost Contour')
            plt.xlabel('Variable ' + str(t1))
            plt.ylabel('Variable ' + str(t2))
            #plt.show()

            fig3 = plt.figure()
            ax = fig3.gca(projection='3d')
            plt.contour(R, P, J_vals.T)
            plt.plot(theta[t1], theta[t2], J_history[-1], 'rx', markersize=10)
            plt.title('3D Cost Contour')
            plt.xlabel('Variable ' + str(t1))
            plt.ylabel('Variable ' + str(t2))
            #plt.show()
            return fig1, fig2, fig3

        fig1, fig2, fig3 = plotThetaContour(X, y, thetaGD, J_history, 0, 1, 0.3)
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)