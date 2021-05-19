

def enthalpy(avg_tt, avg_mt):
    enth = ((1.006 * avg_tt) + (avg_mt / 1000 * (2501 + (1.805 * avg_tt))))
    return enth
