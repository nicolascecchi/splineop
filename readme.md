# SPOP: Time Series Compression With Quadratic Splines

Spline Optimal Partitioning (SPOP) is an algorithm for estimating changes in the second derivative of quadratic splines (i.e. continuous and continuously differentiable piece-wise quadratic polynomials).

# Notes on usage

Note that the cost functions and the algorithms are implemented using Numba. For this reason, when calling a method you shall **not use named arguments, and pass them positionally**. 

For example, you should call `model.predict(13)` and **not** `model.predict(K=13)`. 

Otherwise you will get ```TypeError: some keyword arguments unexpected```  



# References
SPOP: Time Series Compression With Quadratic Splines
Under non-disclosure agreement until publication. 

Authors: Nicolás E. Cecchi, Vincent Runge, Charles Truong and Laurent Oudre

