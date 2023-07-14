import numpy as np

def convert_float_to_three_digit(number):
        three_digit_number = '{:06d}'.format(int(number * 1000))
        half_length = len(three_digit_number) // 2
        first_half = str(three_digit_number[:half_length])
        return(first_half)

def convert_float_to_six_digit(number):
    six_digit_number = '{:06d}'.format(int(np.round(number * 1000)))
    half_length = len(six_digit_number) // 2
    first_half = str(six_digit_number[:half_length])
    second_half = str(six_digit_number[half_length:])
    return(first_half, second_half)

def redshift(a):
    z = (1 - a) / a
    z[z == np.inf] = 0
    return(z)