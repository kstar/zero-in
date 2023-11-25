import re
def convert_ra(ra_string):
    if type(ra_string) is not str:
        assert type(ra_string) is float
        return ra_string

    match = re.match(r' *(\d+)(?:[: ]|[hH] ?)(\d{2})(?:[: ]|[mM] ?)(\d{2}(?:\.\d+)?)[sS]? *$', ra_string)
    if not match:
        match = re.match(r'(\d{2})(\d{2})(\d{2}) *', ra_string)
    if match:
        rah, ram, ras = list(map(float, match.groups()))
        if rah >= 24 or ram >= 60 or ras >= 60:
            raise ValueError('Invalid (out-of-range) RA: {}'.format(ra_string))
        return 15.0 * (rah + ram / 60.0 + ras / 3600.0)
    try:
        ra = float(ra_string)
    except:
        raise ValueError('Gave up trying to interpret RA string: {}'.format(ra_string))

    return ra

def convert_dec(dec_string):
    if type(dec_string) is not str:
        assert type(dec_string) is float
        return dec_string

    match = re.match(r'([+-]?)(\d+)(?:[: ]|[dD] ?)(\d{2})(?:[: ]|[mM] ?)(\d{2}(?:\.\d+)?)[sS]? *$', dec_string)
    if not match:
        match = re.match(r'([+-]?)(\d{2})(\d{2})(\d{2}) *', dec_string)
    if match:
        sgn, *parts = match.groups()
        decd, decm, decs = map(float, parts)
        if decd >= 90 or decm >= 60 or decs >= 60:
            raise ValueError('Invalid (out-of-range) Dec: {}'.format(dec_string))
        assert sgn in ('+', '-', '')
        if sgn == '-':
            sgn = -1
        else:
            sgn = +1
        return sgn * (decd + decm / 60.0 + decs / 3600.0)

    try:
        dec = float(dec_string)
    except:
        raise ValueError('Gave up trying to interpret Dec string: {}'.format(dec_string))

    return dec

def convert_dms(dms_string):
    """Like convert_dec, but has no constraints on value of the angle, so
    it can be used for longitude

    """
    match = re.match(r'([+-]?)(\d+)(?:[: ]|[dD] ?)(\d{2})(?:[: ]|[mM] ?)(\d{2}(?:\.\d+)?)[sS]? *$', dms_string)
    if not match:
        match = re.match(r'([+-]?)(\d{2})(\d{2})(\d{2}) *', dms_string)
    if match:
        sgn, *parts = match.groups()
        dms_d, dms_m, dms_s = map(float, parts)
        if dms_m >= 60 or dms_s >= 60:
            raise ValueError('Invalid (out-of-range) DMS: {}'.format(dms_string))
        assert sgn in ('+', '-', '')
        if sgn == '-':
            sgn = -1
        else:
            sgn = +1
        return sgn * (dms_d + dms_m / 60.0 + dms_s / 3600.0)

    try:
        deg = float(dms_string)
    except:
        raise ValueError('Gave up trying to interpret DMS string: {}'.format(dms_string))

    return deg

def pretty_ra(ra):
    if type(ra) is str:
        ra = convert_ra(ra)
    assert type(ra) in (float, int)
    while ra < 0:
        ra += 360.0
    while ra >= 360.0:
        ra -= 360.0
    assert ra >= 0.0 and ra <= 360.0

    _ = ra / 15.0
    rah = int(_)
    _ = (_ - rah)*60.0
    ram = int(_)
    ras = (_ - ram)*60.0
    return '{:02}:{:02}:{:04.1f}'.format(rah, ram, round(ras, 1))

def pretty_dec(dec):
    if type(dec) is str:
        dec = convert_dms(dec)

    assert type(dec) in (float, int)

    sgn = '+'
    if dec < 0:
        sgn = '-'
        _ = -dec
    else:
        _ = dec

    ded = int(_)
    _ = (_ - ded) * 60.0
    dem = int(_)
    des = (_ - dem) * 60.0

    return '{}{:02}:{:02}:{:04.1f}'.format(sgn, ded, dem, round(des, 1))

def pretty_icrs(icrs):
    return pretty_ra(icrs.ra) + ' ' + pretty_dec(icrs.dec)

def pretty_short(angle: float) -> str:
    """
    angle: degrees

    For angles 1 degree or larger, print XX.X°.
    For angles 1 arcminute to 59 arcminutes, print XX.X'
    For angles less than an arcminute, print XX.X"
    """

    if angle > 1.0:
        return f'{angle:.1f}°'
    angle *= 60
    if angle > 1.0:
        return f'{angle:.1f}\''
    angle *= 60
    return f'{angle:.1f}"'
