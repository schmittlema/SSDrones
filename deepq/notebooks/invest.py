import math

def concatenate_int(x, y):

  try:
     a = math.floor(math.log10(y))
  except ValueError:
     a = 0
  return int(x * 10 ** (1 + a) + y)

print(concatenate_int(1,2))
print(concatenate_int(12,3))
print(concatenate_int(123,4))
print(concatenate_int(423,1))
