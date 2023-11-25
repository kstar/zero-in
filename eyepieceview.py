from typing import Optional
import math
import logging

logger = logging.getLogger('eyepieceview')

from coordinates import EquatorialCoordinates, LiteAltAzFrame
from dms import pretty_ra, pretty_dec

def _compute_eyepiece_rotation(
        ra: float, dec: float,
        rap: float, decp: float,
        lst: float, lat: float,
        focuser_angle: float,
        display_angle: float,
        display_on_scope: bool,
):
    """
      ra: Current epoch (JNow) right ascension of the plate center in degrees
     dec: Current epoch (JNow) declination of the plate center in degrees
     rap: Current epoch (JNow) right ascension of the NCP at the plate epoch (usually ICRS NCP) in degrees
    decp: Current epoch (JNow) declination of the NCP at the plate epoch (usually ICRS NCP) in degrees
     lst: Current local sidereal time in degrees
     lat: Latitude of the observer in degrees
    focuser_angle: Telescope focuser offset angle in degrees (0° for Obsession telescopes)
    """

    ##
    # The plate has the up direction corresponding to the direction of
    # the NCP at the plate epoch (typically, J2000.0). We must
    # therefore first rotate the plate so that the top points towards
    # the current (JNow) NCP before adding further rotations
    ##

    logger.debug(f'Plate center coordinates JNow: {pretty_ra(ra)}, {pretty_dec(dec)}')
    logger.debug(f'J2000 NCP coordinates in JNow: {pretty_ra(rap)}, {pretty_dec(decp)}')
    logger.debug(f'LST {pretty_ra(lst)}, latitude {pretty_dec(lat)}')
    logger.debug(f'Focuser angle: {focuser_angle}')
    # The formula used for this precessionOffset is derived as follows:
    #
    # Consider the spherical triangle formed by the Current NCP, the
    # Plate NCP and the Object. Let us denote the Object's current
    # coordinates by (α, δ), the Object's plate epoch coordinates by
    # (α₀, δ₀) and the current coordinates of Plate NCP by (α_p, δ_p)
    #
    # The angle at the Object vertex is the desired angle PO. Applying
    # cosine rule to the arc opposite this angle,
    #
    # [1] cos(PO) = [sin(δ_p) - sin(δ₀) sin(δ)]/[cos(δ₀) cos(δ)]
    #
    # Applying the sine rule between this angle and the angle at the
    # current NCP,
    #
    # [2] sin(PO) = sin(α_p - α) cos(δ_p) / cos(δ₀)
    #
    # Finally, using the cosine rule for the arc opposite current NCP
    # to eliminate sin(δ₀),
    #
    # [3] sin(δ₀) = sin(δ_p) sin(δ) + cos(δ_p) cos(δ) cos(α_p - α)
    #
    # Substituting the above in [1] and dividing [2] by it, one
    # obtains
    #
    # PO = atan2(sin(α_p - α), tan(δ_p) cos(δ) - sin(δ) cos(α_p - α))
    #
    # We can double-check by visualization that if the plate NCP is
    # east of the object (i.e. α_p > α), the plate needs to be rotated
    # counterclockwise, and if the plate NCP is west of the object,
    # the plate needs to be rotated clockwise. Therefore, sin(PO) has
    # the same sign as sin(α_p - α)
    #
    # However, chances are δ_p is very close to 90°, and tan(δ_p) is
    # ill-conditioned and hard to compute. So we will re-multiply both
    # numerator and denominator by cos(δ_p) which is always
    # positive. Thus,
    #
    # PO = atan2(
    #    sin(α_p - α) cos(δ_p),
    #    sin(δ_p) cos(δ) - sin(δ) cos(δ_p) cos(α_p - α)
    # )

    dRA_rad = math.radians(rap - ra)
    dec_rad = math.radians(dec)
    decp_rad = math.radians(decp)
    sin_decp, cos_decp = math.sin(decp_rad), math.cos(decp_rad)
    sin_dRA, cos_dRA = math.sin(dRA_rad), math.cos(dRA_rad)
    sin_dec, cos_dec = math.sin(dec_rad), math.cos(dec_rad)

    precessionOffset = math.atan2( # CCW +ve
        sin_dRA * cos_decp,
        sin_decp * cos_dec - sin_dec * cos_decp * cos_dRA
    )

    logger.debug(f'Precession offset {math.degrees(precessionOffset)}')

    ##
    # This block of code calculates the angle that the (JNow) NCP
    # makes with respect to "vertical up" at a given point in the
    # observer's sky
    ##
    # The formula used for northAngle is derived as follows:
    #
    # Consider the spherical triangle formed by the NCP, Object and Zenith
    #
    # Apply cosine rule for the arc opposite to the unknown angle (NA); we obtain
    #
    # [1] cos(NA) = [sin(φ) - sin(h) sin(δ)] / [cos(h) cos(δ)]
    #
    # (δ denotes declination, h denotes altitude, and φ denotes latitude)
    #
    # Apply sine rule between the angle at the object and the angle at NCP
    #
    # [2] sin(NA) = cos(φ) sin(-HA) / cos(h)
    #
    # Apply cosine rule for the arc opposite the NCP to eliminate sin(h):
    #
    # [3] sin(h) = sin(φ) sin(δ) + cos(φ) cos(δ) cos(HA)
    #
    # Substitute the above for sin(h) in the first equation, and then
    # divide the second equation by it, to obtain
    #
    # tan(NA) = sin(-HA) / [tan(φ) cos(δ) - sin(δ) cos(HA)]
    #
    # By visualization, we can double-check that sin(NA) has the same
    # sign as sin(-HA) -- looking at an eastern object, NCP is
    # counterclockwise from vertically up, i.e. positive NA, but HA is
    # negative; similarly looking at a western object, NCP is
    # clockwise from vertically up, i.e. negative NA, but HA is
    # positive.
    #
    # But it turns out we anyway need to compute the altitude for the
    # next step, so we might as well compute [3] first, and then use
    # [2] / [1] directly. We double-check that we have preserved signs
    # to the arguments of atan2.

    lat_rad = math.radians(lat)
    HA_rad = math.radians(lst - ra)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_HA, cos_HA = math.sin(HA_rad), math.cos(HA_rad)

    sin_alt = sin_lat * sin_dec + cos_lat * cos_dec * cos_HA
    alt = math.asin(sin_alt) # Angle is between -90° to 90° so no need for atan2
    logger.info('Altitude of the plate center as computed by eyepiece rotation {}'.format(math.degrees(alt)))
    cos_alt = math.sqrt(1 - sin_alt ** 2) # always positive, saving a trig call

    northAngle = math.atan2( # CCW +ve
        -sin_HA * cos_lat * cos_dec,
        sin_lat - sin_alt * sin_dec
    )

    # This gives the clockwise angle that the plate must be rotated by
    print('Precession offset: {}'.format(math.degrees(precessionOffset)))
    print('North angle: {}'.format(math.degrees(northAngle)))
    print('altitude: {}'.format(math.degrees(alt)))

    alt_correction = 0.0 if display_on_scope else alt

    rotation = math.degrees(
        -precessionOffset - northAngle - alt_correction
    ) + 180 - focuser_angle + display_angle # CW +ve

    return rotation

    # Ye olde code
    # sinlat = math.sin(math.radians(lat))
    # sindec = math.sin(math.radians(dec))
    # cosdec = math.cos(math.radians(dec))
    # sinalt = math.sin(math.radians(alt))
    # cosalt = math.cos(math.radians(alt))
    # sinaz = math.sin(math.radians(az))
    # cosNorthAngle = (sinlat - sinalt * sindec) / (cosalt * cosdec)
    # northAngle = math.degrees(math.acos(cosNorthAngle))
    # if sinaz > 0:
    #     northAngle = -northAngle

    # # Clockwise rotation
    # rotation = northAngle - alt + 180 - focuser_angle

def compute_eyepiece_rotation(
        center: EquatorialCoordinates,
        plate_ncp: EquatorialCoordinates,
        frame: LiteAltAzFrame,
        focuser_angle: float,
        display_angle: float,
        display_on_scope: bool,
):
    """ A better typed method """
    if abs(center.epoch_jd - plate_ncp.epoch_jd) > 30.4: # More than an average Julian month
        raise RuntimeError(
            f'Epoch incompatibility: The epoch of the plate NCP '
            f'({center.epoch_jd}) differs significantly from that of the plate '
            f'center ({center.epoch_jd})'
        )
    return _compute_eyepiece_rotation(
        center.ra, center.dec,
        plate_ncp.ra, plate_ncp.dec,
        frame.LST, frame.lat,
        focuser_angle,
        display_angle,
        display_on_scope,
    )
