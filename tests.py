import math
#import scipy as sp
import numpy as np
#import sympy as smp
#from scipy.integrate import quad
import random
import time

def fabs(x):
    return sqrt(pow(x,2))

def deg(radians): #turns radians into degrees, needs math library
    degrees = radians*180/math.pi
    return degrees

def rad(degrees): #turns degrees into radians, needs math library
    radians = degrees/180*math.pi
    return radians

def squared_add(x): #repeated addition
    result = 0
    for i in range(x):
        result += x
    return result

""" needs to add features"""
"""
def pow(x, power, modulo = 1): #works only for positive powers + slow
    result2 = 1
    result = 1
    while (power >= 1):        
        result *= x
        power -= 1
    if (power > 0):
        y = 2
        for i in range(64):
            if(power > 1/y):
                result2 *= root2(x,y)
                power -= 1/y
            y *= 2
            result2 = round(result2, 24)
    result = result*result2
    if (modulo == 0 or modulo == 1):
        return result
    else:
        return (result % modulo)
"""

def pow_mod(x, y, z):
    "Calculate (x ** y) % z efficiently."
    number = 1
    while y:
        if y & 1:
            number = number * x % z
        y >>= 1
        x = x * x % z
    return number

def root(x,root): #works only for based 2 roots
    if (fmod(root,2) > 0):
        print("invalid root")
    while (root > 1):
        x = sqrt(x)
        root /= 2
    return x

def root2(x,root): #works, root without checking if its divisible by 2
    while (root > 1):
        x = sqrt(x)
        root /= 2
    return x

def sqrt(x): #works
    aprx = 8
    for i in range(64):
        aprx = (aprx+x/aprx)/2
        if aprx*aprx == x:
            break
    return aprx 

def factorial(x): #works
    result = 1
    while (x > 0):
        result *= x
        x -= 1
    return result

def fmod(numerator, denomenator): #works
    while (numerator >= denomenator):
        numerator -= denomenator
    return numerator

def sin(theta): #works, input is in radians
    theta = fmod(theta + pi(), 2 * pi()) - pi()
    result = 0
    termsign = 1
    power = 1

    for i in range(64):
        result += (pow(theta, power) / factorial(power)) * termsign
        termsign *= -1
        power += 2
    return result

def cos(theta): #works, input is in radians
    theta = fmod(theta + pi(), 2 * pi()) - pi()
    result = 0
    termsign = 1
    power = 0

    for i in range(64):
        result += (pow(theta, power) / factorial(power)) * termsign
        termsign *= -1
        power += 2
    return result

def tan(theta): #works, input is in radians
    return sin(theta) / cos(theta)

def log(base, product, it = 64): #works but slow
    if base <= 0 or base == 1 or product <= 0:
        return float('nan')
    if base == product:
        return 1
    if product == 1:
        return 0

    current_pow = 1
    closest_int = False
    while not closest_int:
        current_err = fabs(product - pow(base, current_pow))
        if current_err > fabs(product - pow(base, current_pow + 1)):
            current_pow += 1
        elif current_err > fabs(product - pow(base, current_pow - 1)):
            current_pow -= 1
        else:
            closest_int = True

    iterations = it
    pow_change = 1
    for i in range(iterations):
        pow_change /= 2
        current_err = fabs(product - pow(base, current_pow))
        if current_err > fabs(product - pow(base, current_pow + pow_change)):
            current_pow += pow_change
        elif current_err > fabs(product - pow(base, current_pow - pow_change)):
            current_pow -= pow_change
    return current_pow

def atan(y,x=1,iterations = 32): #works, output is in degrees
    if x <= 1:
        x = round(x * 1000000)
        y = round(y * 1000000)
    elif x < 100:
        x = x * 262144
        y = y * 262144
    elif x < 1000:
        x = x * 32768
        y = y * 32768
    elif x < 100000:
        x = x * 256
        y = y * 256
    else:
        x = x * 32
        y = y * 32

    positive = True
    angle = 45
    sum_angle = 0
    ang_table = [
    45, 26.5650512, 14.0362435, 7.12501635, 
    3.57633437, 1.78991061, 0.89517371, 0.447614171, 
    0.2238105, 0.111905677, 0.0559528919, 0.0279764526, 
    0.0139882271, 0.00699411368, 0.00349705685, 0.00174852843,
    0.000874264, 0.000437132, 0.000218566, 0.000109283,
    0.0000546415, 0.0000273208, 0.0000136604, 0.00000683019,
    0.00000341509, 0.00000170755, 0.000000853774, 0.000000426887,
    0.000000213443, 0.000000106722, 0.0000000533609, 0.0000000266804]
    for i in range(iterations):
        if positive:
            X = (x) + (y >> i)
            Y = (y) - (x >> i)
            sum_angle = sum_angle + ang_table[i]
        else:
            X = (x) - (y >> i)
            Y = (y) + (x >> i)
            sum_angle = sum_angle - ang_table[i]

        x = X
        y = Y

        if y > 0:
            positive = True
        elif y < 0:
            positive = False
        else:
            break
        
        angle = angle/2
    return sum_angle

def asin(y, x = 1): #works, output is in degrees
    y = y/x
    if y > 1:
        print("[asin] Error: Invalid input")
        return float('nan')
    n = y/sqrt(1-(y*y))
    n = round(n * 2048)
    return atan(n)  

def acos(y, x = 1): #works, output is in degrees
    y = y/x
    if y > 1:
        print("[acos] Error: Invalid input")
        return float('nan')
    n = sqrt(1-(y*y))/y
    n = round(n * 2048)
    return atan(n)  

def next_power_of_2(n):
    if n == 0:
        return 1
    if n & (n - 1) == 0:
        return n
    while n & (n - 1) > 0:
        n &= (n - 1)
    return n << 1

def FFT(P):
    # P is [p^0,p^1,...,p^n-1] coefficient representation
    length = len(P)
    n = next_power_of_2(length) # n is a power of 2
    half_n = int(n/2)
    for i in range(n-length):
        P.append(0)
    if n == 1:
        return P
    omega = e() ** (2j*pi()/n)
    #omega = round(omega.real, 10) + round(omega.imag, 10) * 1j
    Pe, Po = P[::2], P[1::2]
    Ye, Yo = FFT(Pe), FFT(Po)
    y = [0] * n
    for k in range(half_n):
        y[k] = Ye[k] + ((omega ** k)*Yo[k])
        y[k + half_n] = Ye[k] - ((omega ** k)*Yo[k])
    for j in range(n):
        y[j] = round(y[j].real, 12) + round(y[j].imag, 12) * 1j    
    return y

def IFFT(P):
    # P is [p(omega^0),p(omega^1),...,p(omega^n-1)] value representation
    length = len(P)
    n = next_power_of_2(length) # n is a power of 2
    half_n = int(n/2)
    for i in range(n-length):
        P.append(0)
    if n == 1:
        return P
    omega = e() ** (-2j*pi()/n)
    #omega = round(omega.real, 10) + round(omega.imag, 10) * 1j
    Pe, Po = P[::2], P[1::2]
    Ye, Yo = IFFT_rec(Pe), IFFT_rec(Po)
    y = [0] * n
    for k in range(half_n):
        y[k] = Ye[k] + ((omega ** k)*Yo[k])
        y[k + half_n] = Ye[k] - ((omega ** k)*Yo[k])
    for j in range(n):
        y[j] /= n
        y[j] = round(y[j].real, 12) + round(y[j].imag, 12) * 1j
    return y

def IFFT_rec(P):
    # P is [p(omega^0),p(omega^1),...,p(omega^n-1)] value representation
    length = len(P)
    n = next_power_of_2(length) # n is a power of 2
    half_n = int(n/2)
    for i in range(n-length):
        P.append(0)
    if n == 1:
        return P
    omega = e() ** (-2j*pi()/n)
    #omega = round(omega.real, 8) + round(omega.imag, 8) * 1j
    Pe, Po = P[::2], P[1::2]
    Ye, Yo = IFFT_rec(Pe), IFFT_rec(Po)
    y = [0] * n
    for k in range(half_n):
        y[k] = Ye[k] + ((omega ** k)*Yo[k])
        y[k + half_n] = Ye[k] - ((omega ** k)*Yo[k])
    return y

def value_multiplication(a, b): #must be used with same array sizes from Nth roots of unity (roots from a complex circle around 0)
    len_a, len_b = len(a), len(b)
    if len_a != len_b:
        print("invalid size")
        return 0
    c = [0] * len_a
    for i in range(len_a):
        c[i] = a[i] * b[i]
    return c

def polynomial_multiplication(a, b = [1]): #default b is [1,1,...,1]
    if (len(a) > len(b)):
        b.extend([0] * (len(a)-len(b)))
    if (len(a) < len(b)):
        a.extend([0] * (len(b)-len(a)))
    a.extend([0] * len(a)), b.extend([0] * len(b))
    result  = IFFT(value_multiplication(FFT(a), FFT(b)))
    for j in range(len(result)):
        result[j] = round(result[j].real, 5) + round(result[j].imag, 5) * 1j
    return result

def pi(): #atan(1)*4
    return rad(atan(1))*4

def pi2(iterations = 16):
    sum_a = 0
    for i in range(iterations):
        a = factorial(4*i) * (1103 + (26390 * i))
        b = pow(factorial(i), 4) * pow(396, (4 * i))
        sum_a += a/b
    sum_b = 2*sqrt(2) * sum_a
    sum = 9801/sum_b
    return sum

def pi3(iterations = 262144):
    result = 0
    for i in range(iterations):
        result += 1/pow((i+1), 2)
    pi = sqrt(result * 6)
    return pi

def pi4(iterations = 65536): #John Wallis
    a = 2
    b = 1
    sum_a = 1
    for i in range(iterations):
        sum_a *= a/b
        a += 2
        b += 2
    
    a = 2
    b = 3
    sum_b = 1
    for i in range(iterations):
        sum_b *= a/b
        a += 2
        b += 2
    
    result = sum_a * sum_b
    return (result*2)

def pi5(iterations = 2048): #Gregory-Leibniz
    sum = 0
    for i in range(iterations):
        a = pow(-1, i)
        b = (2*i) + 1
        sum += a/b
    return (sum * 4)

def pi6(iterations = 64): #Francois Viete
    a = 0
    sum = 1
    for i in range(iterations):
        a = sqrt(2 + a)
        sum *= (a/2)
    pi = (1/sum) * 2
    return pi

def pi7(): #John Machin
    pi = (4 * rad(atan(1,5)) - rad(atan(1,239))) * 4
    return pi

def pi8(iterations = 2048): #Gregory-Leibniz
    a = 1
    pi = 0
    for i in range(iterations):
        pi += 4 * pow(-1, i)/a
        a += 2
    return pi

def pi9(iterations = 1024): #Nilakantha
    multiplier = 1
    start_denominator = 2
    pi = 3
    for i in range(iterations):
        pi += (4 /(start_denominator * (start_denominator + 1) * (start_denominator + 2)) * multiplier)
        start_denominator += 2
        multiplier *= -1
    return pi

def e(iterations = 64):
    result = 0
    for i in range(iterations):
        result += 1/(factorial(i))
    return result

"""
Note: uncomment the files at the top
=======================
SYMBOLIC REPRESENTABLE:
=======================

To calculate integrals start by defining x such as:
>>> x = smp.symbols('x', real = True)
This means that x is real. You can also add things like "postive = True" etc.

Then define your function for example:
>>> f = smp.sin(x)**3 * smp.exp(-5*x)
or
>>> f = smp.cos(b*x)* smp.exp(-a*x)
Note: smp.exp uses e as base so "smp.exp(-a*x)" is actually e^-a*x

To integrate, use:
>>> smp.integrate(f, x)
or 
>>> print(smp.integrate(f, x))
to print to console

To simplify add .simplify():
>>> smp.integrate(f, x).simplify()

If you have any constants in the equation define using:
>>> a, b = smp.symbols('a b', real = True)
Note: define before defining the function

If you have any fractions or sqrts in your function, use smp to define them otherwise they are calculated as floats:
>>> f = (1+smp.sqrt(x))**smp.Rational(1,3) / smp.log(x)
Note: log is base e

If you have a definitive integral (a range to calculate the integral) use:
>>> smp.integrate(f, (x, 0, smp.oo))
to calculate from 0 - infinty (oo looks like infinity symbol)

If you would like the actual value use ".evalf" (evals as float):
>>> smp.integrate(f, (x, 0, smp.log(4))).evalf()

TO USE WITH DERIVATIVES USE smp.diff(f, x)
To get the nth derivative use smp.diff(f, x, n)
To compute value of derivative:
>>> deriv = smp.diff(f, x, n)
>>> deriv.subs([(x,8), (a,2), (b,3)]).evalf()

======================
INTEGRATE NUMERICALLY:
======================

Note: must be definite integral (must have a range)
Define the function as a lambda function
>>> f = lambda x: np.exp(-np.sin(x))
Then call quad for your range:
>>> quad(f, 1, 2)
The range there is from 1 to 2
Note: will output 2 numbers, first is the integral and the second is the error. To get just integral use:
>>> quad(f, 1, 2)[0]

Another example:
>>> f = lambda x: 1/((a-np.cos(x))**2 + b-np.sin(x)**2)
>>> a, b = 2, 3
>>> quad(f, 0, 2*np.pi)

If I want to solve for many a's and b's:
>>> def(x, a, b):
>>>     return 1/((a-np.cos(x))**2 + b-np.sin(x)**2)
>>> a_array = np.arrange(2,10,1) #will be [2,3,4,5,6,7,8,9,10], from 2 to 10 in incraments of 1
>>> b_array = np.arrange(2,10,1)
>>> integrals = [[a, b, quad(f, 0, 2*np.pi, args=(a,b))[0]] for a in a_array for b in b_array]
To get just the integrals:
>>> np.array(integrals).T[2]
"""

def nBitRandom(n):
    # Returns a random number
    # between 2**(n-1)+1 and 2**n-1'''
    return(random.randrange(2**(n-1)+1, 2**n-1))

def getLowLevelPrime(n,low_level_primes = 500): #low primality test
    '''Generate a prime candidate not divisible
      by first primes'''
    while True: 
        # Obtain a random number
        prime_candidate = nBitRandom(n) 
   
        for divisor in prime_finder(low_level_primes): 
            if (prime_candidate % divisor == 0 and divisor**2 <= prime_candidate):
                break
            # If no divisor found, return prime_candidate
            else: 
                return prime_candidate

def isMillerRabinPassed(mrc, iterations = 20): # works for all primes
    '''Run 20 iterations of Rabin Miller Primality test'''
    maxDivisionsByTwo = 0
    ec = mrc-1
    while ec % 2 == 0:
        ec >>= 1
        maxDivisionsByTwo += 1
    assert(2**maxDivisionsByTwo * ec == mrc-1)
 
    def trialComposite(round_tester):
        if pow(round_tester, ec, mrc) == 1:
            return False
        for i in range(maxDivisionsByTwo):
            if pow(round_tester, 2**i * ec, mrc) == mrc-1:
                return False
        return True
 
    # Set number of trials here
    for i in range(iterations):
        round_tester = random.randrange(2, mrc)
        if trialComposite(round_tester):
            return False
    return True

def prime_finder(n):
    #initializes array with True values
    prime = [True] * (n + 1)
    for i in range(0, n + 1):
        prime[i] = True

    p = 3
    while (p * p <= n):
        if (prime[p] == True):
            for i in range(p + p, n+1, p):
                prime[i] = False
        p += 1
    
    result = []
    
    result.append(2)
    for p in range(3, n+1, 2):
        if prime[p]:
            result.append(p)         
    return result

def random_prime(size, accuracy = 20):
    while True:
        prime_candidate = getLowLevelPrime(size)
        if not isMillerRabinPassed(prime_candidate, accuracy):
            continue
        else:
            break
    return prime_candidate

def LuscasLehmerSeries(n):
    term = 4
    result = [4]
    while (len(result) < n):
        term = (term*term)-2
        result.append(term)
    return result

def isPrime(p, low_level = 500):
    num = 2 ** p - 1
    term = 4 % num

    for divisor in prime_finder(low_level): 
        if (num % divisor == 0 and divisor**2 <= num):
            return False
        # If no divisor found, return prime_candidate

    for i in range(1, p - 1):
        term = (term * term - 2) % num

    if (term == 0): 
        return True
    else: 
        return False

start = time.time()
print(pi4())
end = time.time()
print(end-start)
print(np.pi)
