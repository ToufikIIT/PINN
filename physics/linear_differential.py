def ode_residual(x, params, forward):
    """
    Compute the residual for the linear ODE: y' + 3y = 2x
    
    Args:
        x: Input of shape (1, n)
        params: List of parameters [w, b, v]
        forward: Forward function (forward_ode)
    
    Returns:
        Residual of shape (1, n)
    """
    y, y_x, *_ = forward(x, params)
    return y_x + 3*y - 2*x
