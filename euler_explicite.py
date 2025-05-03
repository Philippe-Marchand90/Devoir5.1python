def euler_explicite(F,t0,y0,h,tf):
 
    t_vals = np.arange(t0, tf + h, h)
    y_vals = np.zeros((len(y0), len(t_vals)))
    
    y = y0.copy()
    for i, t in enumerate(t_vals):
        y_vals[:,i] = y
        y = y + h * F(t, y)
    
    return t_vals, y_vals
    