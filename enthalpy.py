def enthalpy(avg_tt, avg_mt):
    return (1.006 * avg_tt) + (avg_mt / 1000 * (2501 + (1.805 * avg_tt)))
