__all__ = ["concrete_expand_log"]


def concrete_expand_log(expr, first_call=True):
    """Expand log explcitely

    Parameters
    ----------
    expr :  `~sympy.log`
        Sympy expression
    first_call : bool
        First call

    Returns
    -------
    expr : `~sympy.Sum`
        Expanded sum
    """
    import sympy as sp

    if first_call:
        expr = sp.expand_log(expr, force=True)

    func = expr.func
    args = expr.args

    if args == ():
        return expr

    if func == sp.log:
        if args[0].func == sp.concrete.products.Product:
            Prod = args[0]
            term = Prod.args[0]
            indices = Prod.args[1:]
            return sp.Sum(sp.log(term), *indices)

    return func(*map(lambda x: concrete_expand_log(x, False), args))
