import pandas as pd
import numpy as np


def get_area_ratio(areas, sample_info, counts, ref_class):
    """
    Estimate area from a ratio estimator using indicator observations

    args:
      areas (1-D array): list of stratum areas
      sample_info (3-D array):
          dim1: strata code
          dim2: map label
          dim3: reference label
      counts (1-D list): sample counts per stratum
      ref_class (int): class to estimate

    returns:
      df (pandas DataFrame):
        area_prop (col): proportional area of map of ref_class
        area_pix (col): area of ref class (Landsat pixels)
        area_ha (col): area of ref_class (hectares)
        area_km2 (col): area of ref_class (km^2)
        se_prop (col): proportional standard error of estimate of area
        se_pix (col): standard error of estimate of area (Landsat pixels)
        se_ha (col): standard error of estimate of area (hectares)
        se_km2 (col): standard error of estimate of area (km^2)

    """

    st1 = len(np.where((sample_info[0,:] == 1) & (sample_info[2,:] == ref_class))[0])
    st2 = len(np.where((sample_info[0,:] == 2) & (sample_info[2,:] == ref_class))[0])
    st3 = len(np.where((sample_info[0,:] == 3) & (sample_info[2,:] == ref_class))[0])
    st4 = len(np.where((sample_info[0,:] == 4) & (sample_info[2,:] == ref_class))[0])
    st5 = len(np.where((sample_info[0,:] == 5) & (sample_info[2,:] == ref_class))[0])
    st6 = len(np.where((sample_info[0,:] == 6) & (sample_info[2,:] == ref_class))[0])
    st7 = len(np.where((sample_info[0,:] == 7) & (sample_info[2,:] == ref_class))[0])

    counts = counts.astype(np.float)

    prop_st1 = st1 / counts[0]
    prop_st2 = st2 / counts[1]
    prop_st3 = st3 / counts[2]
    prop_st4 = st4 / counts[3]
    prop_st5 = st5 / counts[4]
    prop_st6 = st6 / counts[5]
    prop_st7 = st7 / counts[6]

    # Class area and proportion from stehman 2014 eq. 27
    total_class = (areas[0] * prop_st1) + (areas[1] * prop_st2) + (areas[2] * prop_st3)\
                + (areas[3] * prop_st4) + (areas[4] * prop_st5) + (areas[5] * prop_st6) + (areas[6] * prop_st7)

    # Calculate area proportion
    area_proportion = total_class / areas.sum()

    # Now variance from Stehman 2014 eq. 26
    st1_var1 = ((1 - prop_st1)**2) / (counts[0] - 1) * st1
    st1_var2 = ((0 - prop_st1)**2) / (counts[0] - 1) * (counts[0] - st1)
    st1_var = st1_var1 + st1_var2
    st2_var1 = ((1 - prop_st2)**2) / (counts[1] - 1) * st2
    st2_var2 = ((0 - prop_st2)**2) / (counts[1] - 1) * (counts[1] - st2)
    st2_var = st2_var1 + st2_var2
    st3_var1 = ((1 - prop_st3)**2) / (counts[2] - 1) * st3
    st3_var2 = ((0 - prop_st3)**2) / (counts[2] - 1) * (counts[2] - st3)
    st3_var = st3_var1 + st3_var2

    st4_var1 = ((1 - prop_st4)**2) / (counts[3] - 1) * st4
    st4_var2 = ((0 - prop_st4)**2) / (counts[3] - 1) * (counts[3] - st4)
    st4_var = st4_var1 + st4_var2

    st5_var1 = ((1 - prop_st5)**2) / (counts[4] - 1) * st5
    st5_var2 = ((0 - prop_st5)**2) / (counts[4] - 1) * (counts[4] - st5)
    st5_var = st5_var1 + st5_var2

    st6_var1 = ((1 - prop_st6)**2) / (counts[5] - 1) * st6
    st6_var2 = ((0 - prop_st6)**2) / (counts[5] - 1) * (counts[5] - st6)
    st6_var = st6_var1 + st6_var2

    st7_var1 = ((1 - prop_st7)**2) / (counts[6] - 1) * st7
    st7_var2 = ((0 - prop_st7)**2) / (counts[6] - 1) * (counts[6] - st4)
    st7_var = st7_var1 + st7_var2

    # Standard error from class
    term1 = 1.0 / (areas.sum()**2)
    term2a = areas[0]**2 * (1 - (counts[0]/areas[0])) * (st1_var/counts[0])
    term2b = areas[1]**2 * (1 - (counts[1]/areas[1])) * (st2_var/counts[1])
    term2c = areas[2]**2 * (1 - (counts[2]/areas[2])) * (st3_var/counts[2])
    term2d = areas[3]**2 * (1 - (counts[3]/areas[3])) * (st4_var/counts[3])
    term2e = areas[4]**2 * (1 - (counts[4]/areas[4])) * (st5_var/counts[4])
    term2f = areas[5]**2 * (1 - (counts[5]/areas[5])) * (st6_var/counts[5])
    term2g = areas[6]**2 * (1 - (counts[6]/areas[6])) * (st7_var/counts[6])

    term2 = term2a + term2b + term2c + term2d + term2e + term2f + term2g
    variance = term1 * term2
    standard_error = np.sqrt(variance)

    # Calculate for pixel counts, hectares, and km2

    df = pd.DataFrame()


    df['se_prop'] = [standard_error]
    df['se_pix'] = se_total = [standard_error * areas.sum()]
    df['se_ha'] = se_ha = [se_total[0] * (30**2 / 100**2)]
    df['se_km2'] = [se_ha[0] * .01]


    df['area_prop'] = [area_proportion]
    df['area_pix'] = area_total = [area_proportion * areas.sum()]
    df['area_ha'] = area_ha = [area_total[0] * (30**2 / 100**2)]
    df['area_km2'] = [area_ha[0] * .01]

    return df

def get_year_dist(ref_year1, ref_year2, interp_df):
    """ Get inputs for time period for area estimation using ratio estimator

    args:
        ref_year1 (int): first year of period
        ref_year2 (int): second year of period
        interp_df (dataframe): reference observations
    returns:

    """
    strata = []
    reference = []
    mapl = []
    id_list = []

    for index, row in interp_df.iterrows():
        ID = row['ID']
        strata_label = row['Strata_Code']

        reference_label = 1 # Non-change observation

        # Get disturbance info for that year
        year1 = row['Year1']
        year2 = row['Year2']
        year3 = row['Year3']

        if year1 != 0:
            year1 = int(year1)

        if year2 != 0:
            year2 = int(year2)

        if year3 != 0:
            year3 = int(year3)

        if year1 >= ref_year1 and year1 <= ref_year2:
            reference_type = row['Type1']
            if reference_type == 'D/ND':
                reference_label = 3
            elif reference_type == 'Deforestation':
                reference_label = 4
        elif year2 >= ref_year1 and year2 <= ref_year2:
            reference_type = row['Type2']
            if reference_type == 'D/ND':
                reference_label = 3
            elif reference_type == 'Deforestation':
                reference_label = 4
        elif year3 >= ref_year1 and year3 <= ref_year2:
            reference_type = row['Type3']
            if reference_type == 'D/ND':
                reference_label = 3
            elif reference_type == 'Deforestation':
                reference_label = 4

        strata.append(int(strata_label))
        reference.append(reference_label)
        mapl.append(reference_label)
    return strata, reference, mapl


def get_area_ratio_paper(areas, sample_info, counts, ref_class):
    """
    Estimate areas from Stehman 2014 using a ratio estimator using indicator observations

    args:
      areas (1-D array): list of stratum areas
      sample_info (3-D array):
          dim1: strata code
          dim2: map label
          dim3: reference label
      counts (1-D list): sample counts per stratum
      ref_class (int): class to estimate

    returns:
      df (pandas DataFrame):
        area_prop (col): proportional area of map of ref_class
        area_pix (col): area of ref class (Landsat pixels)
        area_ha (col): area of ref_class (hectares)
        area_km2 (col): area of ref_class (km^2)
        se_prop (col): proportional standard error of estimate of area
        se_pix (col): standard error of estimate of area (Landsat pixels)
        se_ha (col): standard error of estimate of area (hectares)
        se_km2 (col): standard error of estimate of area (km^2)

    """

    st1 = len(np.where((sample_info[0,:] == 1) & (sample_info[2,:] == ref_class))[0])
    st2 = len(np.where((sample_info[0,:] == 2) & (sample_info[2,:] == ref_class))[0])
    st3 = len(np.where((sample_info[0,:] == 3) & (sample_info[2,:] == ref_class))[0])
    st4 = len(np.where((sample_info[0,:] == 4) & (sample_info[2,:] == ref_class))[0])

    counts = counts.astype(np.float)

    prop_st1 = st1 / counts[0]
    prop_st2 = st2 / counts[1]
    prop_st3 = st3 / counts[2]
    prop_st4 = st4 / counts[3]

    # Class area and proportion from stehman 2014 eq. 27
    total_class = (areas[0] * prop_st1) + (areas[1] * prop_st2) + (areas[2] * prop_st3)\
                + (areas[3] * prop_st4)

    # Calculate area proportion
    area_proportion = total_class / areas.sum()

    # Now variance from Stehman 2014 eq. 26
    st1_var1 = ((1 - prop_st1)**2) / (counts[0] - 1) * st1
    st1_var2 = ((0 - prop_st1)**2) / (counts[0] - 1) * (counts[0] - st1)
    st1_var = st1_var1 + st1_var2
    st2_var1 = ((1 - prop_st2)**2) / (counts[1] - 1) * st2
    st2_var2 = ((0 - prop_st2)**2) / (counts[1] - 1) * (counts[1] - st2)
    st2_var = st2_var1 + st2_var2
    st3_var1 = ((1 - prop_st3)**2) / (counts[2] - 1) * st3
    st3_var2 = ((0 - prop_st3)**2) / (counts[2] - 1) * (counts[2] - st3)
    st3_var = st3_var1 + st3_var2

    st4_var1 = ((1 - prop_st4)**2) / (counts[3] - 1) * st4
    st4_var2 = ((0 - prop_st4)**2) / (counts[3] - 1) * (counts[3] - st4)
    st4_var = st4_var1 + st4_var2

    # Standard error from class
    term1 = 1.0 / (areas.sum()**2)
    term2a = areas[0]**2 * (1 - (counts[0]/areas[0])) * (st1_var/counts[0])
    term2b = areas[1]**2 * (1 - (counts[1]/areas[1])) * (st2_var/counts[1])
    term2c = areas[2]**2 * (1 - (counts[2]/areas[2])) * (st3_var/counts[2])
    term2d = areas[3]**2 * (1 - (counts[3]/areas[3])) * (st4_var/counts[3])

    term2 = term2a + term2b + term2c + term2d
    variance = term1 * term2
    standard_error = np.sqrt(variance)

    # Calculate for pixel counts, hectares, and km2

    df = pd.DataFrame()

    df['se_prop'] = [standard_error]
    df['se_pix'] = se_total = [standard_error * areas.sum()]
    df['se_ha'] = se_ha = [se_total[0] * (30**2 / 100**2)]
    df['se_km2'] = [se_ha[0] * .01]


    df['area_prop'] = [area_proportion]
    df['area_pix'] = area_total = [area_proportion * areas.sum()]
    df['area_ha'] = area_ha = [area_total[0] * (30**2 / 100**2)]
    df['area_km2'] = [area_ha[0] * .01]

    return df
